#!/bin/bash
# The chain that keeps the box working while nobody is watching.
#
# On 2026-08-31 a lane finished in the night, nothing was queued behind it, and the GPU sat idle for
# about five hours. Every step here is resumable from its own CSV, so a restart loses nothing and a
# step that is already done is skipped rather than repeated.
#
# Usage: tools/overnight.sh [wait-pids...]
cd /workspace/fusion-equilibrium-challenge-starter
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 120; done; done

echo "===== $(date -u +%F\ %H:%M) STEP 1: rung 1b — the replicate, and the three mixers ====="
# `s7same` is `seq_hidden2: 256`, which at dropout 0.0 is the SAME architecture as `L2` with a
# different initialisation. `fit()` calls torch.manual_seed(self.seed), so re-running a config
# reproduces it byte for byte and cannot measure init noise; changing the module construction order
# is the only way to draw a second sample without touching the architecture or the objective. It is
# therefore the rung-1 table's ONLY error bar, and every other row is uninterpretable without it.
NAME=y_seq OUTFILE=y_seq_rung1.csv SALTS="0" ARMS="s7same conv attn skip" \
ARM_SET_s7same="--set models.seq.seq_hidden2=256" \
ARM_SET_conv="--set models.seq.seq_cell=conv" \
ARM_SET_attn="--set models.seq.seq_cell=attn models.seq.seq_layers=3" \
ARM_SET_skip="--set models.seq.seq_skip=true" \
MODELS=seq SCORE=seq JOBS=20 tools/lane.sh

echo "===== $(date -u +%F\ %H:%M) STEP 2: shaped-basis in production ====="
# Whichever (envelope, sobolev) the ceiling sweep ranked first, read from the table rather than
# pinned here — the lambda bracket lands after this script is written and before this step runs.
# Restricted to n_pca == 50 rows so the basis choice is not confounded with a component count.
read -r ENVW SOBW <<< "$(python3 - <<'PY'
import csv
rows = [r for r in csv.DictReader(open("results/x_basis_ceiling.csv")) if r["n_pca"] == "50"]
best = max(rows, key=lambda r: float(r["S"]))
print(best["envelope"], best["sobolev"])
PY
)"
if [ "$ENVW" = "1.0" ] && [ "$SOBW" = "0.0" ]; then
  echo "  the plain basis ranked first at the ceiling — nothing to test in production, skipping"
else
  echo "  best ceiling: envelope x${ENVW}, sobolev ${SOBW}"
  NAME=z_sbasis OUTFILE=z_sbasis.csv SALTS="0 1 2 3 4 5" ARMS="base shaped" \
  ARM_SET_base="" \
  ARM_SET_shaped="--set features.basis_envelope=${ENVW} features.basis_sobolev=${SOBW}" \
  MODELS="ridge mlp mlp1 mlp2 mlp3 seq" SCORE=ensemble JOBS=20 tools/lane.sh
fi

echo "===== $(date -u +%F\ %H:%M) STEP 3: ladder rung 2 — the top two arms at three salts ====="
# The ladder's own rule: top half of rung 1 promoted, control carried along because the comparison
# is paired within salt. Chosen from the CSV rather than named here, so this step cannot promote
# something the measurement did not.
TOP=$(python3 - <<'PY'
import csv
rows = {r["name"]: float(r["S"]) for r in csv.DictReader(open("results/y_seq_rung1.csv"))}
arms = [(s, n.removeprefix("y_seq_").removesuffix("_s0"))
        for n, s in rows.items() if n.endswith("_s0") and not n.startswith("y_seq_base")]
print(" ".join(a for _, a in sorted(arms, reverse=True)[:2]))
PY
)
echo "  promoting: $TOP"
declare -A SETS=( [lstm]="--set models.seq.seq_cell=lstm"
                  [h512]="--set models.seq.seq_head_hidden=512"
                  [h256]="--set models.seq.seq_head_hidden=256"
                  [L2]="--set models.seq.seq_layers=2"
                  [h2n128]="--set models.seq.seq_hidden2=128"
                  [s7same]="--set models.seq.seq_hidden2=256"
                  [conv]="--set models.seq.seq_cell=conv"
                  [attn]="--set models.seq.seq_cell=attn models.seq.seq_layers=3"
                  [skip]="--set models.seq.seq_skip=true" )
export ARM_SET_base=""
for a in $TOP; do
  # An arm with no entry here would export an EMPTY --set and run as a second copy of the control,
  # scoring near it and looking like a refutation of itself. Refuse instead.
  [ -n "${SETS[$a]+x}" ] || { echo "  no --set recorded for arm '$a'; refusing to run it as a"; \
                              echo "  silent duplicate of the control. Add it to SETS."; exit 1; }
  export "ARM_SET_${a}=${SETS[$a]}"
done
NAME=y_seq OUTFILE=y_seq_rung2.csv SALTS="1 2 3" ARMS="base $TOP" \
MODELS=seq SCORE=seq JOBS=20 tools/lane.sh

echo "===== $(date -u +%F\ %H:%M) OVERNIGHT CHAIN COMPLETE ====="
