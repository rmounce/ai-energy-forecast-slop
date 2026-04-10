#!/usr/bin/env python3
"""
train/train_tft_price.py — Train a TFT price forecast model using AEMO data.

Uses Darts TFTModel with:
  - Target:           SA1 dispatch price ($/MWh, 30-min)
  - Past covariates:  actual demand/interchange/load/pv (no weather — in future only)
  - Future covariates: PREDISPATCH rrp/demand/interchange, PD7Day rrp,
                       SEVENDAYOUTLOOK demand/interchange, weather actuals
                       (used as proxy for future weather; small lookahead bias),
                       cyclic time features, public holiday flag, DST flag
  - Likelihood:       QuantileRegression([0.3, 0.5, 0.7]) — handles negative prices
  - Scaling:          StandardScaler via Darts Scaler, fit on training split only
  - Validation:       Last 30 days held out; EarlyStopping on val_loss

Output (paths from config.json["paths"]):
  - tft_price_model/              — Darts native save (directory)
  - tft_price_scaler_target.pkl
  - tft_price_scaler_past.pkl
  - tft_price_scaler_future.pkl

Units: all prices in $/MWh, all power/demand in MW.
Timestamps: UTC throughout (timezone-naive UTC for Darts compatibility).

PD7Day imputation: pre-2026-02-10 intervals have no PD7Day history.
  Those intervals are filled with predispatch_rrp (same market signal).
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import holidays
import joblib
import numpy as np
import pandas as pd
import pytz

# Darts imports — torch/lightning must be installed first:
#   uv pip install torch lightning --extra-index-url https://download.pytorch.org/whl/cpu
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression
from lightning.pytorch.callbacks import EarlyStopping

ADELAIDE_TZ = pytz.timezone("Australia/Adelaide")

# Pre-2026-02-10: no PD7Day data; impute with predispatch_rrp
PD7DAY_IMPUTATION_CUTOFF = pd.Timestamp("2026-02-10", tz="UTC")

# Training window: how many months of history to use
TRAINING_HISTORY_MONTHS = 13  # PREDISPATCH archive covers ~13 months

# TFT architecture
INPUT_CHUNK_LENGTH = 336   # 7-day encoder window  (7 × 48)
OUTPUT_CHUNK_LENGTH = 144  # 72-hour decoder window (3 × 48)

# Train/val split
VAL_DAYS = 30


# ---------------------------------------------------------------------------
# Config and InfluxDB helpers
# ---------------------------------------------------------------------------

def load_config(path: str = "config.json") -> dict:
    with open(path) as f:
        return json.load(f)


def influx_client(cfg: dict):
    from influxdb import InfluxDBClient
    ic = cfg["influxdb"]
    return InfluxDBClient(
        host=ic["host"],
        port=ic.get("port", 8086),
        username=ic["username"],
        password=ic["password"],
        database=ic["database"],
    )


def _time_range_str(start_dt: datetime, end_dt: datetime) -> tuple[str, str]:
    return (
        start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def query_to_series(client, query: str, col_name: str) -> pd.Series:
    """Execute a single-column GROUP BY time(30m) query; return UTC-naive Series."""
    result = client.query(query, database=client._database)
    rows = list(result.get_points())
    if not rows:
        return pd.Series(dtype=float, name=col_name)
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    df = df.set_index("time").sort_index()
    # Take the first numeric column (the SELECT target)
    numeric_cols = [c for c in df.columns if c not in ("time",)]
    return df[numeric_cols[0]].rename(col_name)


def query_tagged_to_df(client, query: str, tag_name: str) -> pd.DataFrame:
    """
    Execute a GROUP BY <tag_name> query; return flat DataFrame with the tag as a column.
    Columns: 'time' (UTC-naive datetime), tag_name (UTC-naive datetime), + field columns.
    """
    result = client.query(query, database=client._database)
    rows = []
    for key in result.keys():
        _measurement, tags = key
        if not tags:
            continue
        tag_val = tags.get(tag_name)
        for pt in result.get_points(tags=tags):
            row = dict(pt)
            row[tag_name] = tag_val
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    df[tag_name] = pd.to_datetime(df[tag_name], utc=True).dt.tz_localize(None)
    return df.sort_values(["time", tag_name]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Contemporaneous covariate construction
# ---------------------------------------------------------------------------

def build_contemporaneous(
    raw_df: pd.DataFrame,
    value_cols: list[str],
) -> pd.DataFrame:
    """
    For each interval_datetime (raw_df['time']), select the row with
    max(run_time) where run_time <= interval_datetime.

    This gives the "most recently published forecast" for each interval —
    the forecast that would have been available at the time of the interval.

    Args:
        raw_df:     DataFrame with columns 'time' (interval_dt), 'run_time', + value_cols
        value_cols: columns to keep in the output
    Returns:
        DataFrame indexed by interval_datetime with value_cols
    """
    df = raw_df[["time", "run_time"] + value_cols].copy()
    # Keep only contemporaneous rows: run published before or at the interval
    df = df[df["run_time"] <= df["time"]]
    if df.empty:
        return pd.DataFrame(columns=value_cols)
    # For each interval, keep the most recent run
    best_idx = df.groupby("time")["run_time"].idxmax()
    result = df.loc[best_idx].set_index("time")[value_cols].sort_index()
    return result


# ---------------------------------------------------------------------------
# Data loading functions
# ---------------------------------------------------------------------------

def load_actuals(
    client,
    start_dt: datetime,
    end_dt: datetime,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load SA1 dispatch price (target) and past covariates from InfluxDB.

    Target:
      price ($/MWh) — stored as $/MWh in aemo_dispatch_sa1_30m
      NOTE: forecast.py queries this as price/1000 for LightGBM ($/kWh).
            TFT uses $/MWh directly; Scaler handles the scale difference.

    Past covariates (historical actuals only):
      SA1/VIC1/NSW1: total_demand, net_interchange
      power_load (net of dump load), power_pv

    Returns:
      target_df: single-column DataFrame with 'price'
      past_df:   multi-column DataFrame with past covariate columns
    """
    start_str, end_str = _time_range_str(start_dt, end_dt)

    cols = {
        "price":              f'SELECT mean("price") FROM "rp_30m"."aemo_dispatch_sa1_30m" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\' GROUP BY time(30m)',
        "total_demand_sa1":   f'SELECT mean("total_demand") FROM "rp_30m"."aemo_dispatch_sa1_30m" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\' GROUP BY time(30m)',
        "net_interchange_sa1":f'SELECT mean("net_interchange") FROM "rp_30m"."aemo_dispatch_sa1_30m" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\' GROUP BY time(30m)',
        "total_demand_vic1":  f'SELECT mean("total_demand") FROM "rp_30m"."aemo_dispatch_vic1_30m" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\' GROUP BY time(30m)',
        "net_interchange_vic1":f'SELECT mean("net_interchange") FROM "rp_30m"."aemo_dispatch_vic1_30m" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\' GROUP BY time(30m)',
        "total_demand_nsw1":  f'SELECT mean("total_demand") FROM "rp_30m"."aemo_dispatch_nsw1_30m" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\' GROUP BY time(30m)',
        "net_interchange_nsw1":f'SELECT mean("net_interchange") FROM "rp_30m"."aemo_dispatch_nsw1_30m" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\' GROUP BY time(30m)',
        "temperature":        f'SELECT mean("mean_value") FROM "rp_30m"."temperature_adelaide" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\' GROUP BY time(30m)',
        "humidity":           f'SELECT mean("mean_value") FROM "rp_30m"."humidity_adelaide" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\' GROUP BY time(30m)',
        "wind_speed":         f'SELECT mean("mean_value") FROM "rp_30m"."wind_speed_adelaide" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\' GROUP BY time(30m)',
        "power_load":         f'SELECT mean("mean_value") FROM "rp_30m"."power_load_30m" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\' GROUP BY time(30m)',
        "power_dump_load":    f'SELECT mean("mean_value") FROM "rp_30m"."power_dump_load_30m" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\' GROUP BY time(30m)',
        "power_pv":           f'SELECT mean("mean_value") FROM "rp_30m"."power_pv_30m" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\' GROUP BY time(30m)',
    }

    series_list = []
    for name, q in cols.items():
        s = query_to_series(client, q, name)
        if len(s) > 0:
            series_list.append(s)

    combined = pd.concat(series_list, axis=1)

    # Net load = load - dump load (same adjustment as forecast.py)
    combined["power_load"] = (
        combined["power_load"].fillna(0) - combined.get("power_dump_load", pd.Series(0, index=combined.index)).fillna(0)
    )
    combined = combined.drop(columns=["power_dump_load"], errors="ignore")

    target_df = combined[["price"]].copy()

    # Past covariates: demand/interchange/load/pv (weather goes to future only)
    past_cols = [
        "total_demand_sa1", "net_interchange_sa1",
        "total_demand_vic1", "net_interchange_vic1",
        "total_demand_nsw1", "net_interchange_nsw1",
        "power_load", "power_pv",
    ]
    past_df = combined[[c for c in past_cols if c in combined.columns]].copy()

    return target_df, past_df


def load_predispatch_contemporaneous(
    client,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    """
    Load PREDISPATCH SA1 forecasts and return contemporaneous rrp/demand/interchange.

    'Contemporaneous' means: for each interval_datetime I, the forecast for I
    produced by the most recent PREDISPATCH run published at or before I.
    """
    start_str, end_str = _time_range_str(start_dt, end_dt)
    q = (
        f"SELECT rrp, total_demand, net_interchange "
        f"FROM \"rp_30m\".\"aemo_predispatch_forecast\" "
        f"WHERE region='SA1' AND time >= '{start_str}' AND time <= '{end_str}' "
        f"GROUP BY run_time"
    )
    raw = query_tagged_to_df(client, q, "run_time")
    if raw.empty:
        raise RuntimeError("No PREDISPATCH data found in InfluxDB")

    raw = raw.rename(columns={
        "rrp":           "predispatch_rrp",
        "total_demand":  "predispatch_demand",
        "net_interchange": "predispatch_interchange",
    })
    return build_contemporaneous(
        raw,
        ["predispatch_rrp", "predispatch_demand", "predispatch_interchange"],
    )


def load_pd7day_contemporaneous(
    client,
    start_dt: datetime,
    end_dt: datetime,
    predispatch_rrp: pd.Series,
) -> pd.Series:
    """
    Load PD7Day SA1 RRP forecasts (contemporaneous), imputing with predispatch_rrp
    before PD7DAY_IMPUTATION_CUTOFF (2026-02-10) where no PD7Day data exists.
    """
    start_str, end_str = _time_range_str(start_dt, end_dt)
    q = (
        f"SELECT rrp "
        f"FROM \"rp_30m\".\"aemo_pd7day_forecast\" "
        f"WHERE region='SA1' AND time >= '{start_str}' AND time <= '{end_str}' "
        f"GROUP BY run_time"
    )
    raw = query_tagged_to_df(client, q, "run_time")

    if not raw.empty:
        raw = raw.rename(columns={"rrp": "pd7day_rrp"})
        contemp = build_contemporaneous(raw, ["pd7day_rrp"])
        pd7day_series = contemp["pd7day_rrp"]
    else:
        pd7day_series = pd.Series(dtype=float, name="pd7day_rrp")

    # Reindex to match the predispatch_rrp index (the common 30-min grid)
    combined = pd7day_series.reindex(predispatch_rrp.index)

    # Impute pre-cutoff gaps with predispatch_rrp
    cutoff_naive = PD7DAY_IMPUTATION_CUTOFF.tz_localize(None)
    impute_mask = (combined.index < cutoff_naive) | combined.isna()
    combined.loc[impute_mask] = combined.loc[impute_mask].fillna(
        predispatch_rrp.loc[impute_mask]
    )

    return combined.rename("pd7day_rrp")


def load_sevendayoutlook_contemporaneous(
    client,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    """
    Load SEVENDAYOUTLOOK SA1 demand/interchange (contemporaneous).

    Note: SEVENDAYOUTLOOK horizon starts at midnight *tomorrow* relative to run time,
    so near-term intervals (<24h ahead) may have sparse coverage. Gaps are forward-
    filled in align_and_fill().
    """
    start_str, end_str = _time_range_str(start_dt, end_dt)
    q = (
        f"SELECT scheduled_demand, net_interchange "
        f"FROM \"rp_30m\".\"aemo_sevendayoutlook\" "
        f"WHERE region='SA1' AND time >= '{start_str}' AND time <= '{end_str}' "
        f"GROUP BY run_time"
    )
    raw = query_tagged_to_df(client, q, "run_time")
    if raw.empty:
        raise RuntimeError("No SEVENDAYOUTLOOK data found in InfluxDB")

    raw = raw.rename(columns={
        "scheduled_demand": "sdo_demand",
        "net_interchange":  "sdo_interchange",
    })
    return build_contemporaneous(raw, ["sdo_demand", "sdo_interchange"])


# ---------------------------------------------------------------------------
# Time feature construction
# ---------------------------------------------------------------------------

def build_time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Build cyclic time encodings, public holiday flag, and DST flag.
    Input: UTC-naive DatetimeIndex (values are UTC).
    Features are computed in Adelaide local time.
    """
    # Re-attach UTC so tz_convert works
    local = index.tz_localize("UTC").tz_convert(ADELAIDE_TZ)

    aus_holidays = holidays.Australia(
        state="SA",
        years=range(local[0].year, local[-1].year + 2),
    )

    hour = local.hour + local.minute / 60.0
    doy  = local.day_of_year.astype(float)
    dow  = local.day_of_week.astype(float)
    month = local.month.astype(float)

    df = pd.DataFrame(index=index)
    df["hour_sin"]  = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * dow / 7)
    df["doy_sin"]   = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"]   = np.cos(2 * np.pi * doy / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    # Public holiday flag (date in Adelaide local time)
    local_dates = local.normalize().tz_localize(None)
    df["is_public_holiday"] = np.array(
        [1.0 if d.date() in aus_holidays else 0.0 for d in local_dates]
    )

    # DST flag: Adelaide standard offset is UTC+9:30; DST = UTC+10:30
    dst_offsets = np.array([dt.dst().total_seconds() for dt in local])
    df["is_dst"] = (dst_offsets > 0).astype(float)

    return df


# ---------------------------------------------------------------------------
# Alignment and gap-filling
# ---------------------------------------------------------------------------

def align_and_fill(
    target_df: pd.DataFrame,
    past_df: pd.DataFrame,
    future_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reindex all DataFrames to the 30-min UTC grid defined by target_df.
    Forward/backward-fill gaps up to 2h (4 intervals).
    Warns if NaN values remain after filling.
    """
    idx = pd.date_range(
        start=target_df.index.min(),
        end=target_df.index.max(),
        freq="30min",
    )

    def fill(df: pd.DataFrame, label: str) -> pd.DataFrame:
        df = df.reindex(idx)
        n_before = df.isna().sum().sum()
        df = df.ffill(limit=4).bfill(limit=4)
        n_after = df.isna().sum().sum()
        if n_after > 0:
            cols_with_nan = df.columns[df.isna().any()].tolist()
            print(
                f"  Warning: {n_after} NaN values remain in {label} "
                f"(had {n_before} before fill). Columns: {cols_with_nan}. "
                f"Filling remaining with 0."
            )
            df = df.fillna(0.0)
        return df

    target_df = fill(target_df.reindex(idx), "target")
    past_df   = fill(past_df.reindex(idx), "past_covariates")
    future_df = fill(future_df.reindex(idx), "future_covariates")

    return target_df, past_df, future_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train TFT price forecast model using AEMO pre-dispatch data"
    )
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate data only; skip training and saving",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    client = influx_client(cfg)

    end_dt = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=TRAINING_HISTORY_MONTHS * 31)
    print(f"Training window: {start_dt.date()} → {end_dt.date()}")
    print(f"InfluxDB: {cfg['influxdb']['host']}:{cfg['influxdb'].get('port', 8086)}")

    # -------------------------------------------------------------------------
    # Load actuals
    # -------------------------------------------------------------------------
    print("\n[1/6] Loading dispatch actuals...")
    target_df, past_df = load_actuals(client, start_dt, end_dt)
    print(
        f"  price: {len(target_df)} rows, "
        f"range {target_df['price'].min():.1f}–{target_df['price'].max():.1f} $/MWh, "
        f"{target_df['price'].isna().sum()} NaN"
    )
    print(f"  past covariates: {list(past_df.columns)}")

    # -------------------------------------------------------------------------
    # Load future covariates
    # -------------------------------------------------------------------------
    print("\n[2/6] Loading PREDISPATCH contemporaneous (SA1)...")
    pd_df = load_predispatch_contemporaneous(client, start_dt, end_dt)
    print(
        f"  {len(pd_df)} intervals, "
        f"NaN: rrp={pd_df['predispatch_rrp'].isna().sum()}, "
        f"demand={pd_df['predispatch_demand'].isna().sum()}"
    )

    print("[3/6] Loading PD7Day contemporaneous (SA1, with imputation)...")
    pd7day_series = load_pd7day_contemporaneous(
        client, start_dt, end_dt, pd_df["predispatch_rrp"]
    )
    print(f"  {len(pd7day_series)} intervals, NaN after imputation: {pd7day_series.isna().sum()}")

    print("[4/6] Loading SEVENDAYOUTLOOK contemporaneous (SA1)...")
    sdo_df = load_sevendayoutlook_contemporaneous(client, start_dt, end_dt)
    print(
        f"  {len(sdo_df)} intervals, "
        f"NaN: demand={sdo_df['sdo_demand'].isna().sum()}, "
        f"interchange={sdo_df['sdo_interchange'].isna().sum()}"
    )

    # -------------------------------------------------------------------------
    # Assemble future covariates
    # -------------------------------------------------------------------------
    print("\n[5/6] Building time features and assembling future covariates...")

    # Weather actuals used as proxy for future weather (small lookahead bias)
    # Using actuals is standard practice for offline training; at inference time,
    # forecast.py already fetches BOM weather forecasts for 72h.
    start_str, end_str = _time_range_str(start_dt, end_dt)
    weather_df = pd.DataFrame(
        {
            "temperature": query_to_series(
                client,
                f'SELECT mean("mean_value") FROM "rp_30m"."temperature_adelaide" '
                f"WHERE time >= '{start_str}' AND time <= '{end_str}' GROUP BY time(30m)",
                "temperature",
            ),
            "humidity": query_to_series(
                client,
                f'SELECT mean("mean_value") FROM "rp_30m"."humidity_adelaide" '
                f"WHERE time >= '{start_str}' AND time <= '{end_str}' GROUP BY time(30m)",
                "humidity",
            ),
            "wind_speed": query_to_series(
                client,
                f'SELECT mean("mean_value") FROM "rp_30m"."wind_speed_adelaide" '
                f"WHERE time >= '{start_str}' AND time <= '{end_str}' GROUP BY time(30m)",
                "wind_speed",
            ),
        }
    )

    time_features = build_time_features(target_df.index)

    future_df = pd.concat(
        [pd_df, pd7day_series, sdo_df, weather_df, time_features],
        axis=1,
    )
    print(f"  Future covariate columns ({len(future_df.columns)}): {list(future_df.columns)}")

    # -------------------------------------------------------------------------
    # Align and fill gaps
    # -------------------------------------------------------------------------
    print("\n[6/6] Aligning to 30-min grid and filling short gaps (≤2h)...")
    target_df, past_df, future_df = align_and_fill(target_df, past_df, future_df)
    n_total = len(target_df)
    print(f"  Series length: {n_total} intervals ({n_total / 48:.0f} days)")

    if args.dry_run:
        print("\n--- DRY RUN complete: data validated, training skipped ---")
        print(f"  Target columns:         {list(target_df.columns)}")
        print(f"  Past covariate columns: {list(past_df.columns)}")
        print(f"  Future covariate cols:  {list(future_df.columns)}")
        print(f"  Target stats:  min={target_df['price'].min():.1f}  max={target_df['price'].max():.1f}  mean={target_df['price'].mean():.1f}")
        return

    # -------------------------------------------------------------------------
    # Convert to Darts TimeSeries
    # -------------------------------------------------------------------------
    print("\nConverting to Darts TimeSeries...")
    target_ts = TimeSeries.from_dataframe(target_df, freq="30min", fill_missing_dates=False)
    past_ts   = TimeSeries.from_dataframe(past_df,   freq="30min", fill_missing_dates=False)
    future_ts = TimeSeries.from_dataframe(future_df, freq="30min", fill_missing_dates=False)

    # -------------------------------------------------------------------------
    # Train/val split — last VAL_DAYS days held out
    # -------------------------------------------------------------------------
    val_cutoff = target_ts.end_time() - pd.Timedelta(days=VAL_DAYS)
    target_train, target_val = target_ts.split_before(val_cutoff)
    past_train,   past_val   = past_ts.split_before(val_cutoff)
    future_train, future_val = future_ts.split_before(val_cutoff)
    print(f"  Train: {target_train.start_time().date()} → {target_train.end_time().date()} ({len(target_train)} pts)")
    print(f"  Val:   {target_val.start_time().date()} → {target_val.end_time().date()} ({len(target_val)} pts)")

    # -------------------------------------------------------------------------
    # Scale — fit on training split only
    # -------------------------------------------------------------------------
    print("\nFitting scalers on training data...")
    scaler_target = Scaler()
    scaler_past   = Scaler()
    scaler_future = Scaler()

    target_train_sc = scaler_target.fit_transform(target_train)
    target_val_sc   = scaler_target.transform(target_val)
    past_train_sc   = scaler_past.fit_transform(past_train)
    past_val_sc     = scaler_past.transform(past_val)
    future_train_sc = scaler_future.fit_transform(future_train)
    future_val_sc   = scaler_future.transform(future_val)

    # Full scaled series (needed for prediction at inference time)
    target_full_sc = scaler_target.transform(target_ts)
    past_full_sc   = scaler_past.transform(past_ts)
    future_full_sc = scaler_future.transform(future_ts)

    # -------------------------------------------------------------------------
    # Build and train TFT
    # -------------------------------------------------------------------------
    print("\nBuilding TFT model (CPU training, expect 30–60 min)...")

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=True,
    )

    model = TFTModel(
        input_chunk_length=INPUT_CHUNK_LENGTH,
        output_chunk_length=OUTPUT_CHUNK_LENGTH,
        hidden_size=32,
        lstm_layers=1,
        num_attention_heads=4,
        full_attention=False,
        dropout=0.1,
        batch_size=64,
        n_epochs=100,
        add_relative_index=True,
        likelihood=QuantileRegression(quantiles=[0.3, 0.5, 0.7]),
        optimizer_kwargs={"lr": 1e-3},
        pl_trainer_kwargs={
            "callbacks": [early_stopping],
            "enable_progress_bar": True,
        },
    )

    print("Training...")
    model.fit(
        series=target_train_sc,
        past_covariates=past_train_sc,
        future_covariates=future_train_sc,
        val_series=target_val_sc,
        val_past_covariates=past_val_sc,
        val_future_covariates=future_val_sc,
        verbose=True,
    )

    # -------------------------------------------------------------------------
    # Save model and scalers
    # -------------------------------------------------------------------------
    paths = cfg.get("paths", {})
    model_path    = paths.get("tft_price_model",          "tft_price_model")
    scaler_t_path = paths.get("tft_price_scaler_target",  "tft_price_scaler_target.pkl")
    scaler_p_path = paths.get("tft_price_scaler_past",    "tft_price_scaler_past.pkl")
    scaler_f_path = paths.get("tft_price_scaler_future",  "tft_price_scaler_future.pkl")

    print(f"\nSaving model to '{model_path}'...")
    model.save(model_path)
    joblib.dump(scaler_target, scaler_t_path)
    joblib.dump(scaler_past,   scaler_p_path)
    joblib.dump(scaler_future, scaler_f_path)
    print("Saved.")

    # -------------------------------------------------------------------------
    # Evaluate on validation set
    # -------------------------------------------------------------------------
    print("\nEvaluating on validation set...")
    pred_sc = model.predict(
        n=len(target_val_sc),
        series=target_train_sc,
        past_covariates=past_full_sc,
        future_covariates=future_full_sc,
    )
    pred = scaler_target.inverse_transform(pred_sc)
    actual = target_val

    # p50 component (index 1 for quantiles [0.3, 0.5, 0.7])
    pred_p50 = pred.univariate_component(1).pd_series()
    actual_s = actual.univariate_component(0).pd_series()

    smape = (
        2 * np.abs(actual_s - pred_p50)
        / (np.abs(actual_s) + np.abs(pred_p50) + 1e-8)
    ).mean() * 100
    mae = np.abs(actual_s - pred_p50).mean()

    print(f"  Val sMAPE: {smape:.1f}%")
    print(f"  Val MAE:   {mae:.1f} $/MWh")
    print(
        "\nDone. Compare val sMAPE against LightGBM baseline "
        "before wiring TFT into the live pipeline."
    )


if __name__ == "__main__":
    main()
