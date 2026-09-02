#!/bin/bash
# p100 stopped, p70 taken to 12 pairs. User's call, 2026-09-02.
#
# The numbers behind it, at 5 pairs: p70 mean +0.000003, t +0.05, 3/5 positive — a clean zero
# against a ceiling that promised +0.000413, eight times the standard error. p100 mean -0.000095,
# t -1.29, 1/5 positive — not refuted, but the direction is consistent and the mechanism is: more
# coordinates means more noise carried in them.
#
# NOTHING IN FLIGHT IS WASTED. base_s5 is running and is a row this plan needs; p70_s5 follows it
# and is also needed. So the lane is left alone until BOTH have landed, and cut before it reaches
# p100_s5 — the only run in its queue that this plan does not want.
cd /workspace/fusion-equilibrium-challenge-starter
OLD_PGID=$1
until grep -q "^z_npca_p70_s5," results/z_npca.csv 2>/dev/null; do sleep 120; done
echo "=== $(date -u +%H:%M) salt 5 pair complete; retiring the old lane before it reaches p100_s5 ==="
kill -- -"$OLD_PGID" 2>/dev/null
sleep 30

echo "===== $(date -u +%F\ %H:%M) n_pca 70 to twelve pairs ====="
# Salts 0-11, both arms, same CSV: lane.sh skips every row already present, so this resumes rather
# than repeats and the five pairs already measured are not re-run.
NAME=z_npca OUTFILE=z_npca.csv SALTS="0 1 2 3 4 5 6 7 8 9 10 11" ARMS="base p70" \
ARM_SET_base="" \
ARM_SET_p70="--set features.n_pca=70" \
MODELS="ridge mlp mlp1 mlp2 mlp3 seq" SCORE=ensemble JOBS=20 tools/lane.sh

echo "===== $(date -u +%F\ %H:%M) does conv help the ENSEMBLE ====="
NAME=z_convseq OUTFILE=z_convseq.csv SALTS="0 1 2" ARMS="base convseq" \
ARM_SET_base="" \
ARM_SET_convseq="--set models.seq.seq_cell=conv" \
MODELS="ridge mlp mlp1 mlp2 mlp3 seq" SCORE=ensemble JOBS=20 tools/lane.sh

echo "===== $(date -u +%F\ %H:%M) ladder rung 2, skip only ====="
NAME=y_seq OUTFILE=y_seq_rung2.csv SALTS="1 2 3" ARMS="base skip" \
ARM_SET_base="" \
ARM_SET_skip="--set models.seq.seq_skip=true" \
MODELS=seq SCORE=seq JOBS=20 tools/lane.sh

echo "===== $(date -u +%F\ %H:%M) PLAN 4 COMPLETE ====="
