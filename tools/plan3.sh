#!/bin/bash
# The queue after the user's two calls of 2026-09-01, and the stop rule is PRE-REGISTERED: it is
# written here while `z_sbasis` has TWO of its twelve rows, before salts 1 and 2 exist. That is the
# only condition under which a futility rule is worth anything — `1f-stop-rule` was written the same
# way for the same reason.
#
# RULE. After salts 0, 1 and 2 (six rows), if the mean paired dS of `shaped` against `base` is
# NEGATIVE, the lane is stopped and its remaining ~9 hours go to the next direction. If it is
# positive by any amount, the lane runs its full six salts.
#
# Usage: tools/plan2.sh <pid of the running z_sbasis lane.sh>
cd /workspace/fusion-equilibrium-challenge-starter
LANE=$1
while kill -0 "$LANE" 2>/dev/null; do
  sleep 120
  n=$(awk -F, 'NR>1' results/z_sbasis.csv | wc -l)
  [ "$n" -lt 6 ] && continue
  verdict=$(python3 - <<'PY'
import csv
r = {x["name"]: float(x["S"]) for x in csv.DictReader(open("results/z_sbasis.csv"))}
d = [r[f"z_sbasis_shaped_s{s}"] - r[f"z_sbasis_base_s{s}"]
     for s in range(3) if f"z_sbasis_shaped_s{s}" in r and f"z_sbasis_base_s{s}" in r]
if len(d) < 3:
    print("wait")
else:
    m = sum(d) / len(d)
    print(f"stop {m:+.6f}" if m < 0 else f"continue {m:+.6f}")
PY
)
  case "$verdict" in
    stop*) echo "=== $(date -u +%H:%M) PRE-REGISTERED STOP: mean dS over salts 0-2 is ${verdict#stop } ==="
           echo "  killing the z_sbasis lane; its remaining salts go to n_pca"
           kill -- -"$(ps -o pgid= -p "$LANE" | tr -d ' ')" 2>/dev/null
           sleep 30; break ;;
    continue*) echo "=== $(date -u +%H:%M) rule not triggered: mean dS ${verdict#continue } over 3 salts, lane continues ==="
           while kill -0 "$LANE" 2>/dev/null; do sleep 300; done; break ;;
  esac
done

echo "===== $(date -u +%F\ %H:%M) n_pca 70 and 100 in production ====="
# The gate that closed this direction dissolved on 2026-08-31: components 38-49 used to arrive at
# 1.0-1.5 sd of their own spread and now arrive at 0.672 with R2 0.540, with none reaching its own
# sd. The ceilings are +0.000413 at 70 and +0.000624 at 100, four and six times shaped-basis. The
# ceiling cannot refuse this direction — more components reconstruct strictly better — so it prices
# the prize and the gate is what could have said no.
NAME=z_npca OUTFILE=z_npca.csv SALTS="0 1 2 3 4 5" ARMS="base p70 p100" \
ARM_SET_base="" \
ARM_SET_p70="--set features.n_pca=70" \
ARM_SET_p100="--set features.n_pca=100" \
MODELS="ridge mlp mlp1 mlp2 mlp3 seq" SCORE=ensemble JOBS=20 tools/lane.sh

echo "===== $(date -u +%F\ %H:%M) does conv help the ENSEMBLE, which is the question ====="
# ADAPTED 2026-09-01 from the error-correlation matrix, which inverted the ladder's own ranking.
#
# Two arms of the same architecture differing only in INITIALISATION err at correlation 0.605. So
# 0.605 is what a reseed buys for free, and it is the bar an architecture has to clear. Eight of the
# nine arms do not: lstm 0.630, attn 0.637, L2 0.634, skip 0.624 all err MORE like the control than
# a reseed does. As ensemble members they are worth less than models.seq.seed = 1.
#
# conv is the exception and not narrowly: 0.546 against the control, the lowest row in the matrix,
# and 0.491 against s7same, the minimum of the whole table. It is also the cheapest arm at 21
# minutes against 42-76.
#
# So conv is NOT promoted to ladder rung 2. We already know its single-model composite is level with
# the control, and three more salts would confirm that without answering anything. The question conv
# raises is about the AVERAGE, and it is asked with the average.
NAME=z_convseq OUTFILE=z_convseq.csv SALTS="0 1 2" ARMS="base convseq" \
ARM_SET_base="" \
ARM_SET_convseq="--set models.seq.seq_cell=conv" \
MODELS="ridge mlp mlp1 mlp2 mlp3 seq" SCORE=ensemble JOBS=20 tools/lane.sh

echo "===== $(date -u +%F\ %H:%M) ladder rung 2, skip only ====="
# lstm is dropped from rung 2 for the reason above: it was promoted by composite, before the
# criterion that decides membership had been measured. skip stays — best composite of the ten and
# best Consistency of the ten — and more salts genuinely narrow its error bar.
NAME=y_seq OUTFILE=y_seq_rung2.csv SALTS="1 2 3" ARMS="base skip" \
ARM_SET_base="" \
ARM_SET_skip="--set models.seq.seq_skip=true" \
MODELS=seq SCORE=seq JOBS=20 tools/lane.sh

echo "===== $(date -u +%F\ %H:%M) PLAN 3 COMPLETE ====="
