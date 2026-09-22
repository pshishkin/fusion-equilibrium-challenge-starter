#!/bin/bash
# What the collection is FOR: does blending it beat the production ensemble?
#
# Assembled WITHIN a salt and repeated across salts, which is the design the user set out: every
# member of a salt shares the split, so the salt's 352-shot tail is untouched by all of them and no
# holdout is involved. The comparison is paired within salt; the salt-to-salt spread is the error
# bar.
#
# `salts:a+b+c` averages the DECODED predictions of whole artifacts. z_npca_base is itself a
# five-member ensemble, so "base" here is production exactly, and each c_* artifact contributes the
# one or two models it holds. Ladder rungs, so the shape is visible and not just the endpoint.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
OUT=results/x_blend_eval.csv
[ -f "$OUT" ] || echo "salt,rung,n_artifacts,S,R2_psi,R2_qb,D_LCFS,Cons,members" > "$OUT"
for SALT in ${SALTS:-1 2 3}; do
  B="z_npca_base_s${SALT}"
  MIX="c_conv_s${SALT}+c_lstm_s${SALT}+c_skip_s${SALT}+c_attn_s${SALT}"
  BAS="z_npca_p70_s${SALT}+z_npca_p100_s${SALT}"
  [ -f "artefacts/lanes/baseline_z_sbasis_shaped_s${SALT}.joblib" ] && BAS="${BAS}+z_sbasis_shaped_s${SALT}"
  WID="c_wide_s${SALT}+c_widep70_s${SALT}+c_widep100_s${SALT}"
  ODD="c_parseval_s${SALT}+c_cat_s${SALT}+c_mapnet_s${SALT}"
  for spec in "base:${B}" "mixers:${B}+${MIX}" "bases:${B}+${BAS}+${WID}" \
              "all:${B}+${MIX}+${BAS}+${WID}+${ODD}"; do
    rung=${spec%%:*}; members=${spec#*:}
    grep -q "^${SALT},${rung}," "$OUT" && { echo "have s${SALT} ${rung}"; continue; }
    n=$(echo "$members" | tr '+' '\n' | wc -l)
    LOG="logs/$(date -u +%Y%m%d-%H%M%S)-blend-s${SALT}-${rung}.log"
    echo "=== $(date -u +%H:%M) salt ${SALT} ${rung} (${n} artifacts)"
    FUSION_ARTIFACT="artefacts/lanes/baseline_${B}.joblib" OMP_NUM_THREADS=2 \
      uv run python my_experiments/evaluate.py --share 0.05 --jobs "${JOBS:-8}" \
      --models "salts:${members}" > "$LOG" 2>&1 || { echo "  FAILED, see $LOG"; continue; }
    R=$(awk '
      /COMPOSITE S = /        {s=$4}
      /^R2_psi /              {p=$2}
      /^R2_\{q95,betaN\} /    {q=$2}
      /^1 - D_LCFS /          {d=$4}
      /^Consistency /         {c=$2}
      END {if (s!="") print s","p","q","d","c}
    ' <(sed -E 's/^\[[^]]*\] *//' "$LOG"))
    [ -z "$R" ] && { echo "  NO row in $LOG"; continue; }
    echo "${SALT},${rung},${n},${R},${members}" >> "$OUT"
    echo "  -> s${SALT} ${rung}: ${R%%,*}"
  done
done
echo "BLEND EVAL COMPLETE"
