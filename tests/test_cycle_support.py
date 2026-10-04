import pandas as pd
import pytest

from eval.audit_cycle_support import audit,helper_at


def test_balanced_energy_and_stale_zero_strings_never_fill_missing_pv():
    start,end = pd.Timestamp('2026-09-27T10:00Z'),pd.Timestamp('2026-09-27T10:05Z')
    stale = [{'time':'2026-09-27T09:00Z','value':0}]
    fresh = lambda value:[{'time':at.isoformat(),'value':value} for at in pd.date_range(start,end,freq='1min')]
    history = {'pv':stale,'pv1':stale,'pv2':stale,'battery':fresh(-1000),
        'inverter_ac':fresh(900),'loss':fresh(100),'last_full':[]}
    result = audit(history,start,end,'Australia/Adelaide')
    assert result['string_recoverable_intervals']==0
    assert result['unresolved_intervals']==1
    assert result['unresolved_derived_balance']['maximum_abs_w']==0
    assert result['replay_admission_authorized'] is False


def test_fresh_strings_recover_only_observed_intervals():
    start,end = pd.Timestamp('2026-09-27T10:00Z'),pd.Timestamp('2026-09-27T10:05Z')
    fresh = lambda value:[{'time':at.isoformat(),'value':value} for at in pd.date_range(start,end,freq='1min')]
    history = {key:fresh(1) for key in ('pv1','pv2','battery','inverter_ac','loss')}
    history['pv'] = []
    result = audit(history,start,end,'Australia/Adelaide')
    assert result['string_recoverable_intervals']==1 and result['unresolved_intervals']==0


def test_full_helper_local_time_and_expiry_use_only_causal_receipts():
    rows = [{'time':'2026-09-27T05:35:04Z','state':'2026-09-27 15:05:04'},
        {'time':'2026-09-28T05:19:21Z','state':'2026-09-28 14:49:21'}]
    before = helper_at(rows,pd.Timestamp('2026-09-27T08:40Z'),'Australia/Adelaide')
    after = helper_at(rows,pd.Timestamp('2026-09-28T05:15Z'),'Australia/Adelaide')
    assert before['last_full_utc']=='2026-09-27T05:35:04+00:00'
    assert before['holdoff_active'] and not after['holdoff_active']
    assert after['receipt']==rows[0]['time']


def test_future_full_value_rejects():
    with pytest.raises(ValueError,match='later than receipt'):
        helper_at([{'time':'2026-09-27T05:00Z','state':'2026-09-28T05:00Z'}],
                  pd.Timestamp('2026-09-27T06:00Z'),'Australia/Adelaide')
