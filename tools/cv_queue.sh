#!/usr/bin/env bash
# Serial job queue for K-fold runs (one fit at a time: a full-corpus fit peaks near 64 GB).
# Each line of the queue file: <campaign_dir> <fold> <K> <params> <only models,comma> [--set ...]
# Lines are consumed in order; a line whose fold artifact and oof.npz exist is skipped.
set -uo pipefail
queue=$1
cd /home/pavel/fusion-equilibrium-challenge-starter
while true; do
  line=$(grep -v '^#' "$queue" | grep -v '^\s*$' | head -1) || true
  [ -z "$line" ] && { echo "queue empty $(date)"; exit 0; }
  read -r camp f K params only rest <<<"$line"
  d=$camp/fold$f; mkdir -p "$d"
  echo "START $line $(date)"
  if [ ! -f "$d/baseline.joblib" ]; then
    while pgrep -f -- "--kfold $f/$K" >/dev/null && pgrep -f -- "$params" >/dev/null && [ "$camp" = runs/cv5 ]; do sleep 30; done
  fi
  if [ ! -f "$d/baseline.joblib" ]; then
    set -f
    FUSION_ARTIFACT=$d/baseline.joblib uv run python my_experiments/train.py --params "$params" \
      --only ${only//,/ } --kfold "$f/$K" --jobs 24 $rest > "$d/train.log" 2>&1
    set +f
  fi
  if [ -f "$d/baseline.joblib" ] && [ ! -f "$d/oof.npz" ]; then
    uv run python tools/cv_oof.py dump "$d/baseline.joblib" "$d/oof.npz" --jobs 24 > "$d/dump.log" 2>&1
  fi
  if [ -f "$d/oof.npz" ]; then echo "DONE $line $(date)"; else echo "FAILED $line $(date)"; fi
  # drop the consumed line
  python3 - "$queue" "$line" <<'PY'
import sys
q, line = sys.argv[1], sys.argv[2]
lines = open(q).read().splitlines()
lines.remove(line)
open(q, "w").write("\n".join(lines) + ("\n" if lines else ""))
PY
done
