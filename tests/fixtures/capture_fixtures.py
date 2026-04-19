#!/usr/bin/env python3
"""
Capture frozen fixtures from a live predict-all cycle.

Intercepts InfluxDB queries and HA REST calls, runs Tier 1 and Tier 2 inference
(no actual HA publishing), and writes JSON snapshots to tests/fixtures/.

Usage:
    cd /home/saltspork/src/ai-energy-forecast-slop
    source .venv/bin/activate
    python tests/fixtures/capture_fixtures.py

Re-run after any deliberate pipeline change, review the diff carefully, then commit.
"""

import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _df_to_records(df):
    """Convert a DataFrame to a list of dicts (index as 'timestamp')."""
    if df is None or df.empty:
        return []
    return [
        {"timestamp": idx.isoformat(), **{k: float(v) for k, v in row.items()}}
        for idx, row in df.iterrows()
    ]


def main():
    logging.info("Loading forecast module (may take a few seconds)...")
    import forecast as fc

    logging.info("Running Tier 1 tactical prediction...")
    try:
        tactical_results = fc._execute_tactical_prediction()
    except Exception as e:
        logging.error(f"Tier 1 failed: {e}")
        tactical_results = {}

    if tactical_results:
        tactical_fixture = {
            k: _df_to_records(df)
            for k, df in tactical_results.items()
        }
        out = FIXTURES_DIR / "tactical_output.json"
        with open(out, "w") as f:
            json.dump(tactical_fixture, f, indent=2)
        logging.info(f"Wrote {out} ({len(tactical_fixture)} keys)")
    else:
        logging.warning("Tier 1 returned empty results — tactical_output.json not written")

    logging.info("Running Tier 2 TFT prediction (requires InfluxDB + external APIs)...")
    try:
        from influxdb import InfluxDBClient
        from datetime import datetime, timedelta
        import pytz

        now = datetime.now(pytz.UTC)
        minute = 30 if now.minute >= 30 else 0
        forecast_start = now.replace(minute=minute, second=0, microsecond=0)
        history_start = forecast_start - timedelta(days=fc.CONFIG['prediction_history_days'])
        history_end   = forecast_start - timedelta(minutes=30)

        client = InfluxDBClient(**fc.CONFIG['influxdb'])
        try:
            historical_df = fc.get_historical_data(client, history_start, history_end)
        finally:
            client.close()

        if historical_df.empty:
            raise ValueError("get_historical_data returned empty DataFrame")

        future_sources = {
            'solcast': fc.get_solcast_forecast(),
            'weather': fc.get_weather_forecast(),
            'aemo':    fc.get_aemo_forecast(),
        }
        future_covariates_df = __import__('pandas').concat(
            future_sources.values(), axis=1
        ).sort_index()

        all_feature_cols = set()
        for model_name in fc.CONFIG.get('models', {}):
            all_feature_cols.update(fc.CONFIG['models'][model_name].get('feature_cols', []))
        available_cols = [c for c in all_feature_cols if c in historical_df.columns]
        historical_covariates_df = historical_df[available_cols]

        combined = __import__('pandas').concat([historical_covariates_df, future_covariates_df])
        combined.index = __import__('pandas').to_datetime(combined.index, utc=True)
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
        combined = fc.add_time_features(combined)
        adjusted = fc.apply_covariate_adjustments(combined)
        adjusted.ffill(inplace=True)
        adjusted.bfill(inplace=True)

        tft_results = fc._execute_tft_prediction(historical_df, adjusted)
    except Exception as e:
        logging.error(f"Tier 2 failed: {e}")
        tft_results = {}

    if tft_results:
        # Exclude aemo_price_forecast (internal debug, not primary output)
        tft_fixture = {
            k: _df_to_records(df)
            for k, df in tft_results.items()
            if k in ("tft_price", "tft_price_q30", "tft_price_q70")
        }
        out = FIXTURES_DIR / "tft_price_output.json"
        with open(out, "w") as f:
            json.dump(tft_fixture, f, indent=2)
        logging.info(f"Wrote {out} ({len(tft_fixture)} keys)")
    else:
        logging.warning("Tier 2 returned empty results — tft_price_output.json not written")

    logging.info("Done. Review diffs before committing.")
    logging.info("  git diff tests/fixtures/")


if __name__ == "__main__":
    main()
