"""Rehearse a recorded handoff using isolated installed EMHASS, never live endpoints."""
import argparse
import json
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from energy_pipeline.solver_replay import prepare_request, validate_result, forecast_summary


def container_command(image, workspace):
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', image):
        raise ValueError('require an immutable local Docker image ID')
    return ['docker', 'run', '--rm', '--pull', 'never', '--network', 'none', '--read-only',
            '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
            # Image's uv interpreter lives below /root; non-root cannot execute it.
            # Root remains inside an unprivileged, cap-free, read-only container.
            '--user', '0:0', '--cpus', '1', '--memory', '2g',
            '--pids-limit', '64', '--tmpfs', '/tmp:rw,nosuid,nodev,size=128m,mode=1777',
            '--mount', f'type=bind,src={workspace}/worker.py,dst=/work/worker.py,readonly',
            '--mount', f'type=bind,src={workspace}/request.json,dst=/work/request.json,readonly',
            '--env', 'PYTHONDONTWRITEBYTECODE=1', '--env', 'OMP_NUM_THREADS=1',
            '--env', 'OPENBLAS_NUM_THREADS=1', '--env', 'LP_SOLVER=HIGHS',
            '--entrypoint', '/app/.venv/bin/python', image,
            '/work/worker.py', '/work/request.json']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--optimization-sha256', required=True)
    parser.add_argument('--kind', choices=('dh', 'mpc'), default='dh')
    parser.add_argument('--record-index', type=int, default=-1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; choose a new path')
    with sqlite3.connect(args.journal.resolve().as_uri()+'?mode=ro', uri=True) as db:
        rows = db.execute('SELECT record FROM handoffs ORDER BY rowid').fetchall()
    record = json.loads(rows[args.record_index][0])
    config = json.loads(args.config.read_text())
    request = prepare_request(record, config, kind=args.kind,
                              optimization_sha256=args.optimization_sha256)
    with tempfile.TemporaryDirectory(prefix='energy-solve-replay-') as workspace:
        folder = Path(workspace)
        shutil.copyfile(ROOT/'scripts/emhass_solver_worker.py', folder/'worker.py')
        (folder/'request.json').write_text(json.dumps(request, allow_nan=False))
        # Bind files individually: host directory remains private (0700), while
        # cap-free image root can read the mounted files without DAC override.
        (folder/'worker.py').chmod(0o444)
        (folder/'request.json').chmod(0o444)
        command = container_command(args.image, workspace)
        # Worker thread/solver budgets < docker stop timeout < client timeout.
        # A named container is needed for reliable timeout cleanup; use request identity.
        name = 'energy-replay-'+uuid.uuid4().hex[:16]
        command[2:2] = ['--name', name]
        try:
            completed = subprocess.run(command, text=True, capture_output=True, timeout=90)
        except subprocess.TimeoutExpired:
            subprocess.run(['docker', 'stop', '--time', '1', name], capture_output=True, timeout=15)
            raise
        if completed.returncode:
            raise RuntimeError(f'isolated solver failed: {completed.stderr[-4000:]}')
        result = json.loads(completed.stdout)
    frame = validate_result(request, result)
    artifact = {'schema': 1, 'mode': 'historical_solver_replay',
                'publication_authorized': False, 'image': args.image,
                'request': request, 'result': result, 'summary': forecast_summary(request, frame)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output), 'status': result['status'],
                      'solve_seconds': result['solve_seconds'], 'summary': artifact['summary']}))


if __name__ == '__main__':
    main()
