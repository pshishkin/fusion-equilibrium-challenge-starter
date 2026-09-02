#!/bin/bash
# Salts 2 and 3, but only if salt 1 came out whole. User's call, 2026-09-02.
#
# THE GATE IS THE POINT. Twelve hours behind a bake that half-failed would produce a collection with
# holes in it at three salts instead of one, and the holes would not be in the same places — each
# salt would be missing whatever happened to break there, and no blend could be compared across
# them. So this checks that salt 1 produced every member before it spends anything.
cd /workspace/fusion-equilibrium-challenge-starter
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 120; done; done

WANT="conv lstm skip attn parseval cat"
missing=""
for t in $WANT; do
  grep -q "^c_${t}_s1," results/c_bake_s1.csv 2>/dev/null || missing="$missing $t"
done
[ -f artefacts/lanes/baseline_c_mapnet_s1.joblib ] || missing="$missing mapnet"
if [ -n "$missing" ]; then
  echo "=== $(date -u +%F\ %H:%M) SALT 1 INCOMPLETE, missing:$missing ==="
  echo "  not spending twelve hours widening a collection that has holes in it. Fix salt 1 first;"
  echo "  bake.sh resumes from its CSV, so re-running it only retrains what is absent."
  exit 1
fi
echo "=== $(date -u +%F\ %H:%M) salt 1 complete, all 7 members present ==="

for s in 2 3; do
  echo "===== $(date -u +%F\ %H:%M) baking salt $s ====="
  SALT=$s tools/bake.sh
done
echo "===== $(date -u +%F\ %H:%M) SALTS 2 AND 3 COMPLETE ====="
