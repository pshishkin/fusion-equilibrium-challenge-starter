#!/usr/bin/env bash
# Score queued blends serially: each line "<campaign> <fold> <blend> <out json>".
cd /home/pavel/fusion-equilibrium-challenge-starter
q=$1
while true; do
  line=$(grep -v '^\s*$' "$q" | head -1) || true
  [ -z "$line" ] && { sleep 60; continue; }
  read -r camp f blend out <<<"$line"
  echo "START $line $(date)"
  flock /tmp/fusion_analysis.lock uv run python tools/cv_oof.py score "$camp" --blend "$blend" --folds "$f" --jobs 16 --json "$out" $([ "$blend" = stack ] && echo --oof-name oof_stack.npz) 2>&1 | grep -E "^blend|^  S|Error" || echo "FAILED $line"
  python3 - "$q" "$line" <<'PY'
import sys
q, line = sys.argv[1], sys.argv[2]
L = open(q).read().splitlines(); L.remove(line)
open(q, "w").write("\n".join(L) + ("\n" if L else ""))
PY
done
