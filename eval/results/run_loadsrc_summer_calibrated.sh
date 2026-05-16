#!/bin/bash
# Summer (Nov-Mar) LGBM strategic-bias calibration sensitivity.
#
# Baseline OFF outputs already exist:
#   eval/results/loadsrc_summer_2wk_{actual,lgbm_load_log}_*
#
# This script runs only the calibration-ON pair, keeping all other flags
# identical to run_loadsrc_summer.sh.
#
# Launch via:
#   tmux -S /tmp/loadsrc_summer_cal.tmux new-session -d -s loadsrc_summer_cal \
#     'bash /home/saltspork/src/ai-energy-forecast-slop/eval/results/run_loadsrc_summer_calibrated.sh'
set -uo pipefail

cd /home/saltspork/src/ai-energy-forecast-slop
LOG="eval/results/loadsrc_summer_cal_$(date -u +%Y%m%dT%H%M%SZ).log"
EXITCODE="${LOG%.log}.exitcode"

{
  status=0
  echo "=== Summer calibration ON: 2026-01-01 -> 2026-01-15 ==="
  echo "Compare against existing loadsrc_summer_2wk_{actual,lgbm_load_log}_* baselines."
  echo "Started: $(date -u --iso-8601=seconds)"

  for LOAD in actual lgbm_load_log; do
    echo ""
    echo "--- calibration ON, load source: $LOAD ---"
    PYTHONUNBUFFERED=1 nice -n 19 ./.venv/bin/python eval/rolling_mpc_eval.py \
      --start 2026-01-01T00:00:00Z --end 2026-01-15T00:00:00Z \
      --sources amber_apf_lgbm \
      --economic-mode netload_tariffed \
      --strategic-soc-handoff --strategic-target-mode exact \
      --load-forecast-source "$LOAD" \
      --lgbm-bias-calibration \
      --workers 1 \
      --output-prefix "loadsrc_summer_2wk_cal_${LOAD}" || status=$?
    if [ "$status" -ne 0 ]; then
      echo "FAILED for load source ${LOAD} with exit code ${status}"
      break
    fi
  done

  echo ""
  echo "=== FINISHED: $(date -u --iso-8601=seconds), status=${status} ==="
  exit "$status"
} 2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}
echo "$status" > "$EXITCODE"
exit "$status"
