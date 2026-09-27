#!/usr/bin/env bash
# OnFailure hook of every fusion-* unit (via fusion-notify@.service): tell the agent thread which
# unit failed, how, and the tail of its journal, the moment it happens.
set -uo pipefail
unit=$1
thread=${WATCHDOG_THREAD:-thr_kcjvptiji9}
bb=${BB_CLI:-}
[ -x "$bb" ] || bb=$(ls -t /home/pavel/.npm/_npx/*/node_modules/bb-app/host-daemon/dist/bb 2>/dev/null | head -1)
result=$(systemctl --user show "$unit" -p Result --value)
tail=$(journalctl --user -u "$unit" -n 12 --no-pager -o cat | cut -c1-200)
oom=$(journalctl -k --since "-10min" --no-pager | grep -E "Killed process" | tail -2)
printf 'Unit %s FAILED at %s (result: %s).\n%s\nLast journal lines:\n%s\nDiagnose, fix and restart it (systemd-run via tools/fusion_run.sh), then continue the autonomous work.\n' \
  "$unit" "$(date '+%F %H:%M:%S')" "$result" "${oom:+Kernel: $oom}" "$tail" \
  | "$bb" thread tell "$thread" --mode auto --message-file -
