#!/usr/bin/env bash
# queue_restore.sh <hold-file>: put a held training queue back (anything queued meanwhile goes after it).
# Run as ExecStopPost of every queue-holding unit: a SIGKILL from the OOM killer skips the script's
# own `trap restore EXIT`, and on 2026-10-06 that left the queue held and the GPU idle.
cd /home/pavel/fusion-equilibrium-challenge-starter
h=$1
[ -f "$h" ] || exit 0
cat runs/queue.txt >> "$h" 2>/dev/null
mv "$h" runs/queue.txt
