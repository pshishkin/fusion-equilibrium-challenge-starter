#!/bin/bash
# The submission across three salts, on ONE composition.
#
# Waits for z_sbasis_shaped_s3, because the alternative is a blend whose composition differs by
# salt — the exact fault the uniform search was rebuilt to remove. The other nine configs exist on
# all three salts already; shaped is the last one.
#
# WEIGHTS ARE PER MODEL. Each basis artifact holds five models and each c_* artifact holds one, so
# the spec carries *5 and *1 and every model ends up weighing the same. Without that the four
# single models of a salt would outvote its five-model ensembles.
#
# WHAT THIS IS AND IS NOT. Three salts means three different 80% training draws — the one diversity
# axis this project has never varied. It cannot be scored locally: the members' training sets
# between them cover the whole corpus, so no clean tail exists for a cross-salt blend, and
# evaluate.py would refuse the run and be right to. The single-salt version of this same
# composition measured 0.993261 against production's 0.992831; the three-salt version is a
# reasonable bet on top of that and NOT a measured improvement.
cd /workspace/fusion-equilibrium-challenge-starter
# WAITS FOR THE PIDS IT NEEDS, not only for the artifacts. A submission build runs 28 worker
# processes; started beside another one, plus a bake's twenty, the pool cannot be created and the
# failure is `SemLock._rebuild -> FileNotFoundError` — the same signature a pid-budget exhaustion
# gives, from a different cause. Any pid passed as an argument is waited out first.
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 120; done; done
CONFIGS="z_npca_base:5 z_npca_p70:5 z_npca_p100:5 z_sbasis_shaped:5 \
         c_wide:1 c_widep70:1 c_widep100:1 c_lstm:1 c_attn:1 c_conv:1"
SALTS="${SALTS:-1 2 3}"
while :; do
  missing=""
  for s in $SALTS; do for c in $CONFIGS; do
    [ -f "artefacts/lanes/baseline_${c%%:*}_s${s}.joblib" ] || missing="$missing ${c%%:*}_s${s}"
  done; done
  [ -z "$missing" ] && break
  echo "$(date -u +%H:%M) waiting for:$missing"
  sleep 600
done
SPEC=""
for s in $SALTS; do for c in $CONFIGS; do
  SPEC="${SPEC}+${c%%:*}_s${s}*${c##*:}"
done; done
SPEC=${SPEC#+}
echo "=== $(date -u +%F\ %H:%M) building, $(echo "$SPEC" | tr '+' '\n' | wc -l) artifacts ==="
echo "$SPEC" | tr '+' '\n' | sed 's/^/    /'
echo "$SPEC" > /tmp/submit_3salts_spec.txt
# Its own output directory. `submission/` is the default and is shared, which is how two builds
# come to overwrite each other's arrays.
OUT="submission_3salts"
mkdir -p "$OUT"
FUSION_ARTIFACT=artefacts/lanes/baseline_z_npca_base_s1.joblib \
  make predict_and_submit_to_hf SUBMIT_DIR="$OUT" MODEL="salts:$SPEC"
echo "SUBMIT 3 SALTS COMPLETE"
