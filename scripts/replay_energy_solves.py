"""Rehearse a recorded handoff using isolated installed EMHASS, never live endpoints."""
import argparse
import hashlib
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
from energy_pipeline.solver_replay import digest, prepare_request, validate_result, forecast_summary
from energy_pipeline.solver_chain import build_chained_handoff


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


def run_request(request, image):
    """Run one identity-checked request in the bounded network-free core worker."""
    identity = dict(request)
    request_id = identity.pop("request_id", None)
    if request_id != digest(identity):
        raise ValueError("request identity does not match frozen request")
    with tempfile.TemporaryDirectory(prefix='energy-solve-replay-') as workspace:
        folder = Path(workspace)
        shutil.copyfile(ROOT/'scripts/emhass_solver_worker.py', folder/'worker.py')
        (folder/'request.json').write_text(json.dumps(request, allow_nan=False))
        # Bind files individually: host directory remains private (0700), while
        # cap-free image root can read the mounted files without DAC override.
        (folder/'worker.py').chmod(0o444)
        (folder/'request.json').chmod(0o444)
        command = container_command(image, workspace)
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
    return {'schema': 1, 'mode': 'historical_solver_replay',
                'publication_authorized': False, 'image': image,
                'request': request, 'result': result, 'summary': forecast_summary(request, frame)}


def run_batch(bundle, worker, files):
    """Stage selected replay code; one bounded network-free disposable worker."""
    if worker not in ('eval/sequential_core_replay.py', 'eval/minute_core_replay.py',
                       'eval/dh_feedback_replay.py', 'eval/audit_mpc_formatter.py') or worker not in files:
        raise ValueError('unsupported batch worker')
    if any(Path(filename).is_absolute() or '..' in Path(filename).parts or
            Path(filename).parts[0] not in ('scripts', 'eval', 'energy_pipeline') for filename in files):
        raise ValueError('unsupported staged code path')
    with tempfile.TemporaryDirectory(prefix='energy-batch-') as folder:
        staging = Path(folder)
        for filename in files:
            path = staging/filename
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/filename, path)
            path.chmod(0o444)
        hashes = {filename: hashlib.sha256((staging/filename).read_bytes()).hexdigest() for filename in files}
        (staging/'worker.py').write_text('# unused entrypoint placeholder\n')
        (staging/'request.json').write_text(json.dumps(bundle, allow_nan=False))
        for path in staging.iterdir():
            path.chmod(0o755 if path.is_dir() else 0o444)
        command = container_command(bundle['image'], folder)
        for directory in sorted({Path(filename).parts[0] for filename in files}):
            position = command.index('--env')
            command[position:position] = ['--mount', f'type=bind,src={folder}/{directory},dst=/work/{directory},readonly']
        command[-2:] = ['/work/'+worker, '--worker', '/work/request.json']
        name = 'energy-batch-'+uuid.uuid4().hex[:16]
        command[2:2] = ['--name', name]
        try:
            completed = subprocess.run(command, text=True, capture_output=True, timeout=180)
        except subprocess.TimeoutExpired:
            subprocess.run(['docker', 'stop', '--time', '1', name], capture_output=True, timeout=15)
            raise
        if completed.returncode: raise RuntimeError(completed.stderr[-4000:])
        result = json.loads(completed.stdout)
    return result, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--optimization-sha256', required=True)
    parser.add_argument('--kind', choices=('dh', 'mpc'), default='dh')
    parser.add_argument('--record-index', type=int, default=-1)
    parser.add_argument('--dh-result', type=Path,
                        help='for MPC: replace recorded DH parent with a matching historical DH result')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; choose a new path')
    if args.dh_result and args.kind != 'mpc':
        parser.error('--dh-result requires --kind mpc')
    with sqlite3.connect(args.journal.resolve().as_uri()+'?mode=ro', uri=True) as db:
        rows = db.execute('SELECT record FROM handoffs ORDER BY rowid').fetchall()
    record = json.loads(rows[args.record_index][0])
    config = json.loads(args.config.read_text())
    if args.dh_result:
        parent = json.loads(args.dh_result.read_text())
        if parent['image'] != args.image:
            parser.error('DH/MPC chain must use the same pinned image')
        if parent['request']['config_revision'] != digest(config):
            parser.error('DH/MPC chain must use the same frozen configuration')
        if parent['request']['optimization_sha256'] != args.optimization_sha256:
            parser.error('DH/MPC chain must use the same pinned solver source')
        record = build_chained_handoff(record, parent)
    request = prepare_request(record, config, kind=args.kind,
                              optimization_sha256=args.optimization_sha256)
    artifact = run_request(request, args.image)
    if args.dh_result:
        artifact['chained_handoff'] = record
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output), 'status': artifact['result']['status'],
                      'solve_seconds': artifact['result']['solve_seconds'], 'summary': artifact['summary']}))


if __name__ == '__main__':
    main()
