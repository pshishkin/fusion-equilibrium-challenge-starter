#!/bin/bash
# The blend search on ONE composition, across every salt that has it.
#
# The first table mixed pools: salts 1-2 carried z_sbasis_shaped and salt 3 did not, so "top 75%"
# was 10 of 14 on one and 10 of 13 on the other. That is a difference in COMPOSITION reported as a
# difference between salts, and the comparison across salts is the entire validation design. The
# earlier table is kept at results/archive/ for what it measured, not deleted.
#
# Waits for the shaped basis to exist on every requested salt before starting, because starting
# without it would reproduce exactly the fault this exists to remove.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
SALTS="${SALTS:-1 2 3 4 5}"
while :; do
  missing=""
  for s in $SALTS; do
    [ -f "artefacts/lanes/baseline_z_sbasis_shaped_s${s}.joblib" ] || missing="$missing $s"
  done
  [ -z "$missing" ] && break
  echo "$(date -u +%H:%M) waiting for the shaped basis on salts:$missing"
  sleep 600
done
echo "=== $(date -u +%F\ %H:%M) every salt has all 14 members; searching ==="
for s in $SALTS; do
  n=$(ls artefacts/lanes/ | sed 's/^baseline_//; s/\.joblib$//' | grep -E "_s${s}$" \
      | grep -cE "^(z_npca|z_sbasis_shaped|c_)")
  echo "===== salt ${s}: ${n} artifacts ====="
  FUSION_ARTIFACT="artefacts/lanes/baseline_z_npca_base_s${s}.joblib" OMP_NUM_THREADS=2 \
    uv run python tools/blend_search.py --salt "$s" --jobs "${JOBS:-12}" \
    --rounds "${ROUNDS:-0}" --out results/x_blend_uniform.csv || echo "  salt ${s} FAILED"
done
echo "UNIFORM SEARCH COMPLETE"
