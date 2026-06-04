#!/usr/bin/env bash
# OTHER MACHINE: converged (12.5k-step) retraining for partitions 1/32 and 1/8.
# After git pull: conda run -n slit bash experiments/generalization_memorization/runner_32_8.sh
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$REPO_ROOT/experiments/generalization_memorization/outputs/train_logs"
mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

ARGS="--steps 12500 --sample-every-steps 1250 --tag conv12500"

echo "[runner_32_8] start $(date)  python=$(which python)"
for pn in 32 8; do
  for po in 1 2; do
    log="$LOG_DIR/conv_${po}v${pn}.log"
    echo "[runner_32_8] ($(date +%H:%M:%S)) training ${po}v${pn} -> $log"
    python -u train.py --partno $po --partnum $pn $ARGS 2>&1 | tee "$log"; rc=${PIPESTATUS[0]}
    [ $rc -ne 0 ] && echo "[runner_32_8] !! ${po}v${pn} FAILED (exit $rc)" || echo "[runner_32_8] ok ${po}v${pn}"
  done
done
echo "[runner_32_8] ALL DONE $(date)"
