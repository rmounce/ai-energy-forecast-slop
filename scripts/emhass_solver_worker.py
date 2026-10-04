"""One core EMHASS solve in a disposable container; no HA/web/CLI wrappers."""
import hashlib
from copy import deepcopy
import inspect
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from emhass.optimization import Optimization
from emhass.retrieve_hass import RetrieveHass
from emhass import utils


def solve_request(request):
    source_hash = hashlib.sha256(Path(inspect.getfile(Optimization)).read_bytes()).hexdigest()
    if source_hash != request['optimization_sha256']:
        raise ValueError('installed solver source hash mismatch')
    # EMHASS's parser replaces minutes/timezone strings with Python objects.
    # Keep the frozen request immutable when the worker handles multiple solves.
    conf = deepcopy(request['configuration'])
    payload = request['payload']
    logger = logging.getLogger('isolated_emhass')
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stderr))
    logger.setLevel(logging.WARNING)
    retrieve, optim, plant = utils.get_yaml_parse(conf, logger)
    # Same day-ahead runtime endpoint clamping as utils.treat_runtimeparams.
    soc_final = min(max(payload['soc_final'], plant['battery_minimum_state_of_charge']),
                    plant['battery_maximum_state_of_charge'])
    if soc_final != payload['soc_final']:
        raise ValueError('recorded endpoint needs runtime clamping; unsupported replay')
    with tempfile.TemporaryDirectory(prefix='emhass-solver-') as workspace:
        paths = {key: Path(workspace) for key in ('root_path', 'data_path', 'config_path')}
        solver = Optimization(retrieve, optim, plant, 'unit_load_cost', 'unit_prod_price',
                              optim['costfun'], paths, logger,
                              num_timesteps=payload['prediction_horizon'])
        index = pd.date_range(request['forecast_start'], periods=payload['prediction_horizon'],
                              freq=f"{payload['optimization_time_step']}min")
        data = pd.DataFrame(index=index)
        data['unit_load_cost'] = payload['load_cost_forecast']
        data['unit_prod_price'] = payload['prod_price_forecast']
        started = time.monotonic()
        times = {}
        frame = solver.perform_optimization(data,
            np.array(payload['pv_power_forecast'], dtype=float),
            np.array(payload['load_power_forecast'], dtype=float),
            np.array(payload['load_cost_forecast'], dtype=float),
            np.array(payload['prod_price_forecast'], dtype=float),
            soc_init=payload['soc_init'], soc_final=soc_final, stage_times=times)
        columns = [col for col in frame if pd.api.types.is_numeric_dtype(frame[col])]
        result = {'schema': 1, 'request_id': request['request_id'],
                  'optimization_sha256': source_hash, 'status': solver.optim_status,
                  'solve_seconds': time.monotonic()-started, 'stage_times': times,
                  'targets': [at.isoformat() for at in frame.index], 'columns': columns,
                  'values': frame[columns].to_numpy().tolist(),
                  'temporary_files': sorted(p.name for p in Path(workspace).rglob('*'))}
        if request['kind'] == 'dh':
            # Exercise only the pure static formatter: never instantiate the HA
            # client or call post_data/publish_data, even with dont_post flags.
            projected = {}
            for column, entity, attribute, scale, device, unit in (
                    ('P_Load', 'sensor.dh_p_load_forecast', 'forecasts', 1, 'power', 'W'),
                    ('P_PV', 'sensor.dh_p_pv_forecast', 'forecasts', 1, 'power', 'W'),
                    ('SOC_opt', 'sensor.dh_soc_batt_forecast', 'battery_scheduled_soc', 100,
                     'battery', '%')):
                values = frame[column]*scale
                data = RetrieveHass.get_attr_data_dict(values, 0, entity, device, unit,
                    entity, attribute, np.round(values.iloc[0], 2))
                projected[entity] = {'state': data['state'],
                                     'attributes': {attribute: data['attributes'][attribute]}}
            result['projected_dh_entities'] = projected
            result['projection_source_sha256'] = hashlib.sha256(
                Path(inspect.getfile(RetrieveHass)).read_bytes()).hexdigest()
        return result


def main():
    os.nice(19)
    request = json.loads(Path(sys.argv[1]).read_text())
    print(json.dumps(solve_request(request), allow_nan=False))


if __name__ == '__main__':
    main()
