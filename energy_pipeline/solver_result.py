"""Admission for isolated DH solve results; no HA projection or publication."""
from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from energy_pipeline.accepted_store import RecoveredBundle


@dataclass(frozen=True)
class AcceptedDHSolve:
    revision: str
    price_parent: str
    frame: pd.DataFrame


def accept_dh_result(plan, frame, *, now, status):
    """Accept only a complete optimal result belonging to a still-current price parent."""
    bundle = RecoveredBundle(plan['bundle'])
    bundle.validate()
    if not bundle.time_current(now):
        raise ValueError('DH solve price parent expired')
    if status != 'Optimal':
        raise ValueError('DH solve is not optimal')
    required = pd.date_range(bundle.payload['forecast_start'], periods=144, freq='30min')
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError('DH result requires an aware dataframe')
    if not frame.index.tz_convert('UTC').equals(required):
        raise ValueError('DH result target intervals differ from price parent')
    columns = ('P_Load', 'P_PV', 'SOC_opt')
    if not set(columns) <= set(frame):
        raise ValueError('DH result missing required output columns')
    values = frame.loc[:, list(columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, :2] < 0).any() or (values[:, 2] < 0).any() or (values[:, 2] > 1).any():
        raise ValueError('DH result power/SoC values invalid')
    owned = frame.loc[:, list(columns)].copy(deep=True)
    owned.index = owned.index.tz_convert('UTC')
    content = {'price_parent': plan['id'], 'targets': [at.isoformat() for at in owned.index],
               'values': owned.to_numpy(dtype=float).tolist()}
    revision = hashlib.sha256(json.dumps(content, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return AcceptedDHSolve(revision, plan['id'], owned)
