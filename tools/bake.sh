#!/bin/bash
# Bake the members of the salt-1 collection that are not on disk yet.
#
# One row here is ONE MODEL, not a lane: `--only` names exactly what each run fits, so nothing is
# retrained that already exists. A run that only changes `seq_cell` fits only `seq` — its MLP would
# be byte-for-byte the MLP of the base run, same salt, same seed, same features, same basis, same
# objective. Runs that change the BASIS or the OBJECTIVE fit both, because there both models are
# genuinely different.
#
# Already on disk at salt 1, and therefore absent from this list: mlp and seq-gru
# (z_npca_base_s1), mlp/seq at 70 components (z_npca_p70_s1), at 100 (z_npca_p100_s1), and on the
# shaped basis (z_sbasis_shaped_s1). Eight members for nothing.
cd /workspace/fusion-equilibrium-challenge-starter
SALT=${SALT:-1}
OUT=results/c_bake_s${SALT}.csv
[ -f "$OUT" ] || echo "name,S,R2_psi,R2_qb,D_LCFS,Cons,wall_s" > "$OUT"
run () {   # tag | --only models | --set args
  tag=$1; only=$2; sets=$3
  row="c_${tag}_s${SALT}"
  grep -q "^${row}," "$OUT" && { echo "have $row"; return; }
  ART="artefacts/lanes/baseline_${row}.joblib"
  LOG="logs/$(date -u +%Y%m%d-%H%M%S)-${row}.log"
  echo "=== $(date -u +%H:%M) $row  [only: $only]"
  T0=$SECONDS
  FUSION_ARTIFACT="$ART" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    uv run python my_experiments/train_eval.py 0.80/1.0 0.15/1.0 0.05 \
    --jobs 20 --salt "$SALT" --only $only \
    --score-models "$(echo $only | tr ' ' '+')" $sets > "$LOG" 2>&1 \
    || { echo "  FAILED, see $LOG"; return; }
  R=$(awk '
    /COMPOSITE S = /        {s=$4}
    /^R2_psi /              {p=$2}
    /^R2_\{q95,betaN\} /    {q=$2}
    /^1 - D_LCFS /          {d=$4}
    /^Consistency /         {c=$2}
    END {if (s!="") print s","p","q","d","c}
  ' <(sed -E 's/^\[[^]]*\] *//' "$LOG"))
  [ -z "$R" ] && { echo "  NO row in $LOG"; return; }
  echo "${row},${R},$((SECONDS-T0))" >> "$OUT"
  echo "  -> ${row},${R}"
}

run conv     "seq"      "--set models.seq.seq_cell=conv"
run lstm     "seq"      "--set models.seq.seq_cell=lstm"
run skip     "seq"      "--set models.seq.seq_skip=true"
run attn     "seq"      "--set models.seq.seq_cell=attn models.seq.seq_layers=3"
run parseval "mlp seq"  "--set loss.metric=parseval"
run cat      "catboost" ""

# mapnet has no PCA basis at all, so it is not a member of the zoo and not fittable by train_eval.
# Its reference must be a salt-1 artifact: the scaler and coil plan it borrows are fitted, and the
# script now refuses a mismatch rather than standardising against the wrong training window.
MROW="c_mapnet_s${SALT}"
if [ ! -f "artefacts/lanes/baseline_${MROW}.joblib" ]; then
  echo "=== $(date -u +%H:%M) ${MROW}"
  uv run python my_experiments/mapnet.py --salt "$SALT" --shots 0 --steps 60000 \
    --latent 50 --mask-weight 16 --sobolev 1.0 --seed 0 \
    --reference "artefacts/lanes/baseline_z_npca_base_s${SALT}.joblib" \
    --save-artifact "artefacts/lanes/baseline_${MROW}.joblib" \
    --out "/tmp/mapnet_${MROW}.npz" > "logs/$(date -u +%Y%m%d-%H%M%S)-${MROW}.log" 2>&1 \
    && echo "  -> ${MROW} artifact written" || echo "  mapnet FAILED"
fi
echo "BAKE COMPLETE"
