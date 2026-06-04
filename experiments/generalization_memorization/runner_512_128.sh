#!/usr/bin/env bash
# THIS MACHINE: converged (12.5k-step) retraining for partitions 1/512 and 1/128.
# Launch: conda run -n slit bash experiments/generalization_memorization/runner_512_128.sh
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$REPO_ROOT/experiments/generalization_memorization/outputs/train_logs"
mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

ARGS="--steps 12500 --sample-every-steps 1250 --tag conv12500"

echo "[runner_512_128] start $(date)  python=$(which python)"
for pn in 512 128; do
  for po in 1 2; do
    log="$LOG_DIR/conv_${po}v${pn}.log"
    echo "[runner_512_128] ($(date +%H:%M:%S)) training ${po}v${pn} -> $log"
    python -u train.py --partno $po --partnum $pn $ARGS 2>&1 | tee "$log"; rc=${PIPESTATUS[0]}
    [ $rc -ne 0 ] && echo "[runner_512_128] !! ${po}v${pn} FAILED (exit $rc)" || echo "[runner_512_128] ok ${po}v${pn}"
  done
done
echo "[runner_512_128] ALL DONE $(date)"
