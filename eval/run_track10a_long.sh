#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source .venv/bin/activate

START_TS="${START_TS:-2025-09-01T00:00:00Z}"
END_TS="${END_TS:-2025-10-13T00:00:00Z}"
SOURCES="${SOURCES:-amber_apf_lgbm,p5min_naive,model_a_hybrid}"
BASELINE_SOURCE="${BASELINE_SOURCE:-amber_apf_lgbm}"
WORKERS="${WORKERS:-2}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-rolling_mpc_eval_tracka_followup_6week}"
TFT_CHECKPOINT="${TFT_CHECKPOINT:-models/tft_price/checkpoint_run014_phase7_best.pt}"
TFT_SCALERS="${TFT_SCALERS:-models/tft_price/scalers_run014_phase7.pkl}"
MPL_DIR="${MPL_DIR:-/tmp/mpl_track10a_long}"
LOG_PATH="${LOG_PATH:-/tmp/${OUTPUT_PREFIX}.log}"
EXITCODE_PATH="${EXITCODE_PATH:-/tmp/${OUTPUT_PREFIX}.exitcode}"

mkdir -p "$MPL_DIR"

echo "Running Track 10A long eval"
echo "  start:    $START_TS"
echo "  end:      $END_TS"
echo "  sources:  $SOURCES"
echo "  baseline: $BASELINE_SOURCE"
echo "  workers:  $WORKERS"
echo "  output:   $OUTPUT_PREFIX"
echo "  log:      $LOG_PATH"
echo "  exitcode: $EXITCODE_PATH"

MPLCONFIGDIR="$MPL_DIR" PYTHONUNBUFFERED=1 \
python eval/rolling_mpc_eval.py \
  --start "$START_TS" \
  --end "$END_TS" \
  --sources "$SOURCES" \
  --baseline-source "$BASELINE_SOURCE" \
  --tft-checkpoint "$TFT_CHECKPOINT" \
  --tft-scalers "$TFT_SCALERS" \
  --workers "$WORKERS" \
  --output-prefix "$OUTPUT_PREFIX" \
  2>&1 | tee "$LOG_PATH"
status=${PIPESTATUS[0]}
printf '%s\n' "$status" > "$EXITCODE_PATH"
exit "$status"
