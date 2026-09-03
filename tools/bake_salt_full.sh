#!/bin/bash
# A whole salt's collection from nothing: the basis configs first, then the members, then the wide
# MLPs. Salts 8-12 have no artifact at all, so the base zoo has to be trained before anything can
# reference it — mapnet borrows its scaler and coil plan from the base artifact OF ITS OWN SALT and
# refuses a mismatch, which is the check that makes this order compulsory rather than tidy.
#
# Everything resumes: a config already on disk is skipped, so re-running costs nothing for what
# exists. Roughly 11 hours per salt from scratch.
cd /workspace/fusion-equilibrium-challenge-starter
one () {   # name | --set args
  local row=$1 sets=$2
  local art="artefacts/lanes/baseline_${row}.joblib"
  [ -f "$art" ] && { echo "have $row"; return; }
  local log="logs/$(date -u +%Y%m%d-%H%M%S)-${row}.log"
  echo "=== $(date -u +%H:%M) $row"
  FUSION_ARTIFACT="$art" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    uv run python my_experiments/train_eval.py 0.80/1.0 0.15/1.0 0.05 \
    --jobs 20 --salt "$SALT" --only ridge mlp mlp1 mlp2 mlp3 seq \
    --score-models ensemble $sets > "$log" 2>&1 \
    && echo "  -> $row done" || echo "  $row FAILED, see $log"
}
for SALT in ${SALTS:-4 5 6 7 8 9 10 11 12}; do
  export SALT
  echo "########## $(date -u +%F\ %H:%M) SALT ${SALT} ##########"
  one "z_npca_base_s${SALT}"  ""
  one "z_npca_p70_s${SALT}"   "--set features.n_pca=70"
  one "z_npca_p100_s${SALT}"  "--set features.n_pca=100"
  [ -f "artefacts/lanes/baseline_z_npca_base_s${SALT}.joblib" ] || {
    echo "  no base artifact for salt ${SALT}; the rest of this salt cannot reference it, skipping"
    continue; }
  SALT=$SALT tools/bake.sh
  SALTS=$SALT tools/bake_wide.sh
done
echo "FULL BAKE COMPLETE"
