#!/bin/bash
# Is fp16 worth switching to? Measured, not argued — and measured ALONE, because a wall-clock number
# taken while another lane holds the GPU measures the contention and not the precision.
#
# V100 is compute capability 7.0: fp16 tensor cores yes (125 against 15.7 TFLOPS), TF32 no — that is
# Ampere, so `allow_tf32` is not a lever on this card. Half precision is the only one.
#
# WHERE THE TIME ACTUALLY GOES, from the utilisation logged during the bakes: shot reading 0% GPU
# (~15 min, CPU), the MLP fits 9-22% (matrices of 512x512 are too small to saturate anything —
# that is a launch-latency problem, not an arithmetic one), the seq fit 65-98%, scoring 0% (~10 min,
# CPU). So ~40% of a run is not GPU work at all and precision cannot touch it. `seq` is the only
# phase where this can pay, and it is the only phase measured here.
#
# The FIT time is compared, not the wall time: reading dominates the wall and would swamp the
# signal. Both arms run a fixed 10000 steps with early stopping disabled, so they do the same work.
cd /workspace/fusion-equilibrium-challenge-starter
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 120; done; done
mkdir -p artefacts/bench
OUT=results/x_amp_bench.csv
[ -f "$OUT" ] || echo "arm,fit_s,best_val,steps,wall_s" > "$OUT"
for arm in fp32 amp; do
  grep -q "^${arm}," "$OUT" && { echo "have $arm"; continue; }
  LOG="logs/$(date -u +%Y%m%d-%H%M%S)-ampbench-${arm}.log"
  echo "=== $(date -u +%H:%M) $arm ==="
  T0=$SECONDS
  env $( [ "$arm" = amp ] && echo FUSION_AMP=1 ) \
    FUSION_ARTIFACT="artefacts/bench/baseline_amp_${arm}.joblib" \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    uv run python my_experiments/train.py --share 0.80/1.0 --val-share 0.15/1.0 --jobs 20 \
    --salt 1 --only seq \
    --set models.seq.max_steps=10000 models.seq.patience_steps=1000000 \
          models.seq.eval_every_steps=1000 > "$LOG" 2>&1 \
    || { echo "  FAILED, see $LOG"; continue; }
  FIT=$(grep -oE "seq: fitted in [0-9.]+ s" "$LOG" | tail -1 | grep -oE "[0-9.]+")
  BEST=$(grep -oE "best [0-9.]+ @" "$LOG" | tail -1 | grep -oE "[0-9.]+")
  echo "${arm},${FIT:-NA},${BEST:-NA},10000,$((SECONDS-T0))" >> "$OUT"
  echo "  -> ${arm} fit ${FIT}s best_val ${BEST}"
done
echo "--- verdict ---"; column -s, -t "$OUT"
echo "AMP BENCH COMPLETE"
