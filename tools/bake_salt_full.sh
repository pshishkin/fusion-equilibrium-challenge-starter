#!/bin/bash
# A whole salt's collection from nothing: the basis configs first, then the members, then the wide
# MLPs. Salts 8-12 have no artifact at all, so the base zoo has to be trained before anything can
# reference it — mapnet borrows its scaler and coil plan from the base artifact OF ITS OWN SALT and
# refuses a mismatch, which is the check that makes this order compulsory rather than tidy.
#
# Everything resumes: a config already on disk is skipped, so re-running costs nothing for what
# exists. Roughly 11 hours per salt from scratch.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
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
  # The shaped basis, envelope x16 and sobolev 0.1 — the ceiling sweep's winner. It is REFUTED as a
  # config change (-0.000272 over three pairs) and kept as a MEMBER, which are different questions:
  # a member only has to err somewhere the others do not.
  #
  # It is here because a collection whose composition differs by salt cannot be compared across
  # salts. Salts 1-2 had it and 3 onward did not, so "top 75%" meant 10 of 14 on one salt and 10 of
  # 13 on another — the same fraction over a different pool, which measures composition as much as
  # anything else.
  one "z_sbasis_shaped_s${SALT}" "--set features.basis_envelope=16 features.basis_sobolev=0.1"
  [ -f "artefacts/lanes/baseline_z_npca_base_s${SALT}.joblib" ] || {
    echo "  no base artifact for salt ${SALT}; the rest of this salt cannot reference it, skipping"
    continue; }
  SALT=$SALT tools/bake.sh
  SALTS=$SALT tools/bake_wide.sh
done
echo "FULL BAKE COMPLETE"
