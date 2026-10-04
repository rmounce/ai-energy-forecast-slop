import numpy as np
import pandas as pd
import pytest

from eval.screen_inventory_cycles import screen


def measured(soc):
    frame = pd.DataFrame({'soc_pct_end':soc},index=pd.date_range('2026-10-01',periods=len(soc),freq='5min',tz='UTC'))
    for key in ('pv_dc_w','load_site_w','grid_import_w',
                'battery_charge_w_charge_kwh','battery_charge_w_discharge_kwh'):
        frame[key] = 1.
    return frame


def test_cycle_uses_last_full_before_material_departure_and_end_labels():
    report = screen(measured([100,100,96,94,20,80,100,100]))
    row, = report['cycles']
    assert row['start']=='2026-10-01T00:10:00+00:00'
    assert row['end']=='2026-10-01T00:35:00+00:00'
    assert row['minimum_observed_soc_pct']==20
    assert row['expected_intervals']==5
    assert row['observed_charge_kwh']==5


def test_missing_power_never_becomes_zero_energy_or_admitted_cycle():
    frame = measured([100,30,np.nan,100])
    frame.iloc[2,frame.columns.get_loc('battery_charge_w_charge_kwh')] = np.nan
    row, = screen(frame)['cycles']
    assert row['observed_charge_kwh'] is None
    assert not row['complete_measured_power']
    assert row['finite_intervals']['soc_pct_end']==2


def test_missing_index_and_soc_split_scarce_episodes():
    frame = measured([14,14,np.nan,14,14,14]).drop(pd.Timestamp('2026-10-01T00:20Z'))
    report = screen(frame)
    assert len(report['scarce_episodes'])==3
    assert report['observed_intervals_at_or_below_10_pct']==0
    assert report['publication_authorized'] is False


@pytest.mark.parametrize('column,value',[('soc_pct_end',101),('pv_dc_w',np.inf),('pv_dc_w',-1),('battery_charge_w_charge_kwh',-1)])
def test_invalid_physical_targets_reject(column,value):
    frame = measured([100,20,100])
    frame.loc[frame.index[1],column] = value
    with pytest.raises(ValueError): screen(frame)


def test_out_of_order_intervals_reject():
    with pytest.raises(ValueError): screen(measured([100,20,100]).iloc[::-1])
