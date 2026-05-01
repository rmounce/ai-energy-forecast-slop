import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from eval.dispatch_simulator import solve_lp_dispatch


def test_solve_lp_dispatch_netload_import_only_when_idle_is_best():
    solve = solve_lp_dispatch(
        prices_mwh=np.array([0.0]),
        soc_init=20.0,
        import_prices_mwh=np.array([0.0]),
        export_prices_mwh=np.array([0.0]),
        net_load_forecast_kw=np.array([3.0]),
    )

    assert solve["success"] is True
    assert np.isclose(solve["charge_kw"][0], 0.0)
    assert np.isclose(solve["discharge_kw"][0], 0.0)
    assert np.isclose(solve["grid_import_kw"][0], 3.0)
    assert np.isclose(solve["grid_export_kw"][0], 0.0)
    assert np.isclose(solve["curtail_kw"][0], 0.0)


def test_solve_lp_dispatch_netload_export_surplus_when_feed_in_is_positive():
    solve = solve_lp_dispatch(
        prices_mwh=np.array([0.0]),
        soc_init=0.0,
        import_prices_mwh=np.array([200.0]),
        export_prices_mwh=np.array([100.0]),
        net_load_forecast_kw=np.array([-2.5]),
    )

    assert solve["success"] is True
    assert np.isclose(solve["charge_kw"][0], 0.0)
    assert np.isclose(solve["discharge_kw"][0], 0.0)
    assert np.isclose(solve["grid_import_kw"][0], 0.0)
    assert np.isclose(solve["grid_export_kw"][0], 2.5)
    assert np.isclose(solve["curtail_kw"][0], 0.0)


def test_solve_lp_dispatch_netload_curtails_surplus_when_feed_in_is_negative():
    solve = solve_lp_dispatch(
        prices_mwh=np.array([0.0]),
        soc_init=20.0,
        import_prices_mwh=np.array([0.0]),
        export_prices_mwh=np.array([-100.0]),
        net_load_forecast_kw=np.array([-2.5]),
    )

    assert solve["success"] is True
    assert np.isclose(solve["charge_kw"][0], 0.0)
    assert np.isclose(solve["discharge_kw"][0], 0.0)
    assert np.isclose(solve["grid_import_kw"][0], 0.0)
    assert np.isclose(solve["grid_export_kw"][0], 0.0)
    assert np.isclose(solve["curtail_kw"][0], 2.5)


def test_solve_lp_dispatch_netload_cannot_curtail_when_no_surplus_pv():
    solve = solve_lp_dispatch(
        prices_mwh=np.array([0.0]),
        soc_init=20.0,
        import_prices_mwh=np.array([100.0]),
        export_prices_mwh=np.array([0.0]),
        net_load_forecast_kw=np.array([2.5]),
    )

    assert solve["success"] is True
    assert np.isclose(solve["curtail_kw"][0], 0.0)


def test_solve_lp_dispatch_split_load_pv_can_turn_down_pv_behind_load():
    solve = solve_lp_dispatch(
        prices_mwh=np.array([0.0]),
        soc_init=40.0,
        import_prices_mwh=np.array([-100.0]),
        export_prices_mwh=np.array([-100.0]),
        load_forecast_kw=np.array([3.0]),
        pv_forecast_kw=np.array([2.0]),
    )

    assert solve["success"] is True
    assert np.isclose(solve["charge_kw"][0], 0.0)
    assert np.isclose(solve["discharge_kw"][0], 0.0)
    assert np.isclose(solve["curtail_kw"][0], 2.0)
    assert np.isclose(solve["grid_import_kw"][0], 3.0)
    assert np.isclose(solve["grid_export_kw"][0], 0.0)
