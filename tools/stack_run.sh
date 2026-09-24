#!/usr/bin/env bash
# Stack gate on one fold, then queue its native score. Holds the analysis lock while fitting.
cd /home/pavel/fusion-equilibrium-challenge-starter
fold=$1; shift
flock /tmp/fusion_analysis.lock uv run python tools/stack_gate.py runs/cv5/fold$fold "$@" 2>&1 | grep -vE 'shot/s|s/shot'
echo "runs/cv5 $fold stack runs/cv5/scores/f${fold}_stack.json" >> runs/score_queue.txt
echo ALLDONE
