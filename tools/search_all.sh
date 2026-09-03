#!/bin/bash
# The coarse pass and the greedy, on every salt that has a collection. Sequential: each run holds
# ~20 GB of cached member predictions and two at once would not fit beside a bake.
cd /workspace/fusion-equilibrium-challenge-starter
for s in ${SALTS:-1 2 3}; do
  echo "===== $(date -u +%F\ %H:%M) blend search, salt ${s} ====="
  FUSION_ARTIFACT="artefacts/lanes/baseline_z_npca_base_s${s}.joblib" OMP_NUM_THREADS=2 \
    uv run python tools/blend_search.py --salt "$s" --jobs "${JOBS:-16}" --rounds "${ROUNDS:-6}" \
    || echo "  salt ${s} FAILED"
done
echo "SEARCH ALL COMPLETE"
