#!/bin/bash
# The queue after salt 1: benchmark ALONE first, then widen the collection.
#
# Order matters and it is the user's: the AMP measurement runs before salts 2 and 3, on an idle box.
# Run it beside a lane and it measures GPU contention instead of precision, which is the one thing
# it exists to avoid.
cd /workspace/fusion-equilibrium-challenge-starter
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 120; done; done
echo "===== $(date -u +%F\ %H:%M) salt 1 done; AMP benchmark, alone ====="
tools/amp_bench.sh
echo "===== $(date -u +%F\ %H:%M) salts 2 and 3, gated ====="
tools/bake_more.sh
echo "===== $(date -u +%F\ %H:%M) width x components, salts 1-3 ====="
SALTS="1 2 3" tools/bake_wide.sh
echo "===== $(date -u +%F\ %H:%M) QUEUE COMPLETE ====="
