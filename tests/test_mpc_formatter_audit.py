from copy import deepcopy

import pytest

from eval.audit_mpc_formatter import CHANNELS, compare


def evidence():
    case = {'id':'frozen','accepted_at':'2026-10-03T16:25:20Z',
            'targets':['2026-10-03T16:25:00+00:00']}
    values = {'battery':1000.12,'grid':-400.13,'load':550.,'pv':0.,'hybrid':950.12,'curtailment':0.}
    expected = {'accepted_at':case['accepted_at'],'points':[{'target':case['targets'][0]} | values]}
    result = {'id':'frozen','projected':{key:{'state':str(value),'rows':[
        {'date':case['targets'][0],CHANNELS[key][0].removeprefix('sensor.'):str(value)}]}
        for key,value in values.items()}}
    return case,result,expected


def test_complete_evidence_matches_all_six_consumed_fields():
    assert compare(*evidence()) == 6


@pytest.mark.parametrize('mutation',['identity','target','coverage','state','sign'])
def test_altered_formatter_evidence_cannot_pass(mutation):
    case,result,expected = deepcopy(evidence())
    if mutation == 'identity': result['id'] = 'other'
    elif mutation == 'target': result['projected']['grid']['rows'][0]['date'] = '2026-10-03T16:30Z'
    elif mutation == 'coverage': result['projected']['grid']['rows'] = []
    elif mutation == 'state': result['projected']['grid']['state'] = '400.13'
    else:
        result['projected']['grid']['state'] = '400.13'
        result['projected']['grid']['rows'][0]['mpc_p_grid_forecast'] = '400.13'
    with pytest.raises(ValueError): compare(case,result,expected)
