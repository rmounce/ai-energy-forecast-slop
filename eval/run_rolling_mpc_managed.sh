#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ "$#" -eq 0 ]; then
  echo "Usage: OUTPUT_PREFIX=<prefix> eval/run_rolling_mpc_managed.sh [rolling_mpc_eval args...]" >&2
  exit 2
fi

source .venv/bin/activate

OUTPUT_PREFIX="${OUTPUT_PREFIX:?OUTPUT_PREFIX is required}"
LOG_PATH="${LOG_PATH:-eval/results/${OUTPUT_PREFIX}.log}"
EXITCODE_PATH="${EXITCODE_PATH:-eval/results/${OUTPUT_PREFIX}.exitcode}"
MPL_DIR="${MPL_DIR:-/tmp/mpl_${OUTPUT_PREFIX}}"
NICE_LEVEL="${NICE_LEVEL:-19}"

mkdir -p "$(dirname "$LOG_PATH")" "$(dirname "$EXITCODE_PATH")" "$MPL_DIR"

echo "Running managed rolling MPC eval"
echo "  output:   $OUTPUT_PREFIX"
echo "  log:      $LOG_PATH"
echo "  exitcode: $EXITCODE_PATH"
echo "  mpldir:   $MPL_DIR"
echo "  nice:     $NICE_LEVEL"
echo "  args:     $*"

set +e
MPLCONFIGDIR="$MPL_DIR" PYTHONUNBUFFERED=1 \
nice -n "$NICE_LEVEL" \
./.venv/bin/python eval/rolling_mpc_eval.py --output-prefix "$OUTPUT_PREFIX" "$@" 2>&1 | tee "$LOG_PATH"
status=${PIPESTATUS[0]}
set -e

printf '%s\n' "$status" > "$EXITCODE_PATH"
exit "$status"
