#!/usr/bin/env bash
# fusion_run.sh <unit-name> <MemoryMax> <MemorySwapMax> <command...>
# Every heavy job as a systemd user unit with memory caps and an OnFailure hook that tells the
# agent thread at once (fusion-notify@.service -> tools/fusion_notify.sh).
set -euo pipefail
unit=$1 mem=$2 swap=$3; shift 3
exec systemd-run --user --unit="$unit" -p OnFailure="fusion-notify@${unit%.service}.service" \
  -E PATH=/home/pavel/.local/bin:/usr/local/bin:/usr/bin:/bin -E HOME=/home/pavel -E PYTHONUNBUFFERED=1 \
  -E BB_CLI="${BB_CLI:-}" -p MemoryMax="$mem" -p MemorySwapMax="$swap" \
  --working-directory=/home/pavel/fusion-equilibrium-challenge-starter "$@"
