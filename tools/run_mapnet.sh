#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
SP=/tmp/claude-0/-workspace-fusion-equilibrium-challenge-starter/1d754179-abd1-49e1-84a8-0bc77c512eb4/scratchpad
set -e
echo "=== $(date -u +%H:%M) $TAG training ==="
uv run python "$SP/mapnet.py" --shots "${SHOTS:-3000}" --steps "${STEPS:-60000}" \
  --seed "${SEED:-0}" --latent "${LATENT:-50}" --mask-weight "${MASKW:-16}" --sobolev "${SOB:-1.0}" \
  --upsample "${UPS:-convt}" ${COORDS:+--coords} --out "/tmp/mapnet_${TAG}.npz"
cp /tmp/mapnet_files.txt "/tmp/mapnet_files_${TAG}.txt"
echo "=== $(date -u +%H:%M) $TAG scoring ==="
uv run python local_score.py --source local --pred "/tmp/mapnet_${TAG}.npz" \
  --jobs 6 --files $(tr '\n' ' ' < "/tmp/mapnet_files_${TAG}.txt")
echo "=== $(date -u +%H:%M) $TAG DONE ==="
