import pandas as pd
import pytest
from energy_pipeline.solver_result import accept_dh_result
from test_accepted_store import NOW
from test_publication import plan


def output(plan):
    return pd.DataFrame({'P_Load': [1000.]*144, 'P_PV': [2000.]*144, 'SOC_opt': [.5]*144},
        index=pd.date_range(plan['bundle']['forecast_start'], periods=144, freq='30min'))


def test_result_identity_and_owned_values(plan):
    frame = output(plan)
    accepted = accept_dh_result(plan, frame, now=NOW, status='Optimal')
    assert accepted.price_parent == plan['id']
    frame.iloc[0, 0] = 9000
    assert accepted.frame.iloc[0, 0] == 1000
    assert accepted.revision == accept_dh_result(plan, output(plan), now=NOW, status='Optimal').revision


@pytest.mark.parametrize('bad', ['expired', 'infeasible', 'short', 'shift', 'nan', 'soc', 'negative'])
def test_unusable_solver_output_rejected(plan, bad):
    frame = output(plan); now = NOW; status = 'Optimal'
    if bad == 'expired': now += pd.Timedelta(minutes=4)
    elif bad == 'infeasible': status = 'Infeasible'
    elif bad == 'short': frame = frame.iloc[:-1]
    elif bad == 'shift': frame.index += pd.Timedelta(minutes=30)
    elif bad == 'nan': frame.iloc[0, 0] = float('nan')
    elif bad == 'soc': frame.iloc[0, 2] = 50
    else: frame.iloc[0, 0] = -100
    with pytest.raises(ValueError): accept_dh_result(plan, frame, now=now, status=status)
