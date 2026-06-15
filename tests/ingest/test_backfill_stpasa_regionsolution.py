import io

import pandas as pd
import pytest

from ingest.backfill_stpasa_regionsolution import (
    OUTPUT_COLUMNS,
    _parse_aemo_csv,
    list_current_stpasa_files,
    horizon_summary,
    normalise_regionsolution,
    validate_horizon,
)


def test_normalise_regionsolution_filters_region_and_converts_nem_time_to_utc():
    raw = pd.DataFrame(
        {
            "RUN_DATETIME": ["2026/04/30 00:00:00", "2026/04/30 00:00:00"],
            "INTERVAL_DATETIME": ["2026/05/03 00:00:00", "2026/05/03 00:00:00"],
            "REGIONID": ["SA1", "VIC1"],
            "INTERVENTION": ["0", "0"],
            "UIGF": ["1200.5", "9999"],
            "SS_WIND_UIGF": ["800", "9999"],
            "SS_SOLAR_UIGF": ["400.5", "9999"],
        }
    )

    out = normalise_regionsolution(raw, region_id="SA1")

    assert out.columns.tolist() == OUTPUT_COLUMNS
    assert len(out) == 1
    assert out.loc[0, "run_time"] == pd.Timestamp("2026-04-29T14:00:00Z")
    assert out.loc[0, "interval_dt"] == pd.Timestamp("2026-05-02T14:00:00Z")
    assert out.loc[0, "uigf"] == 1200.5
    assert out.loc[0, "ss_wind_uigf"] == 800.0
    assert out.loc[0, "ss_solar_uigf"] == 400.5


def test_parse_aemo_csv_can_select_regionsolution_from_multi_table_file():
    raw = "\n".join(
        [
            "I,STPASA,CASESOLUTION,3,RUN_DATETIME,PASAVERSION",
            'D,STPASA,CASESOLUTION,3,"2026/06/15 09:00:00",7.4.4.0',
            "I,STPASA,REGIONSOLUTION,3,RUN_DATETIME,INTERVAL_DATETIME,REGIONID,UIGF",
            'D,STPASA,REGIONSOLUTION,3,"2026/06/15 09:00:00","2026/06/18 09:00:00",SA1,123.4',
            'D,STPASA,REGIONSOLUTION,3,"2026/06/15 09:00:00","2026/06/18 09:00:00",VIC1,999.0',
        ]
    ).encode()

    out = _parse_aemo_csv(io.BytesIO(raw), table_name="REGIONSOLUTION")

    assert out.columns.tolist() == ["RUN_DATETIME", "INTERVAL_DATETIME", "REGIONID", "UIGF"]
    assert out["REGIONID"].tolist() == ["SA1", "VIC1"]
    assert out["UIGF"].tolist() == ["123.4", "999.0"]


def test_list_current_stpasa_files_filters_month_range(monkeypatch):
    html = """
    <a href="/Reports/CURRENT/Short_Term_PASA_Reports/PUBLIC_STPASA_202604302300_1.zip">old</a>
    <a href="/Reports/CURRENT/Short_Term_PASA_Reports/PUBLIC_STPASA_202605010000_2.zip">may</a>
    <a href="/Reports/CURRENT/STPASA_DUIDAvailability/PUBLIC_STPASA_DUIDAVAILABILITY_202605010000_3.zip">duid</a>
    <a href="/Reports/CURRENT/Short_Term_PASA_Reports/PUBLIC_STPASA_202606302300_4.zip">jun</a>
    <a href="/Reports/CURRENT/Short_Term_PASA_Reports/PUBLIC_STPASA_202607010000_5.zip">future</a>
    """

    class Response:
        text = html

        def raise_for_status(self):
            return None

    monkeypatch.setattr(
        "ingest.backfill_stpasa_regionsolution.requests.get",
        lambda *args, **kwargs: Response(),
    )

    out = list_current_stpasa_files(start=(2026, 5), end=(2026, 6))

    assert [run_key for run_key, _url in out] == ["2026050100", "2026063023"]


def test_validate_horizon_requires_at_least_one_run_to_reach_threshold():
    df = pd.DataFrame(
        {
            "run_time": pd.to_datetime(
                ["2026-04-30T14:00:00Z", "2026-04-30T14:00:00Z"],
                utc=True,
            ),
            "interval_dt": pd.to_datetime(
                ["2026-05-01T18:30:00Z", "2026-05-03T14:00:00Z"],
                utc=True,
            ),
        }
    )

    validate_horizon(df, min_horizon_hours=72.0)
    summary = horizon_summary(df)
    assert summary["min_horizon_hours"].iloc[0] == 28.5
    assert summary["max_horizon_hours"].iloc[0] == 72.0

    with pytest.raises(ValueError, match="1/1 STPASA runs stop before 73.0h"):
        validate_horizon(df, min_horizon_hours=73.0)
