#!/usr/bin/env bash
# Hourly watchdog for the fusion job units, run by a BB script automation.
#
# Silent when everything is fine. On a problem it sends one message to the agent thread
# ($WATCHDOG_THREAD) describing it, and exits 0 all the same: three failed runs would pause the
# automation, exactly when it is needed. The same problem is re-sent at most every 3 hours.
set -uo pipefail
cd /home/pavel/fusion-equilibrium-challenge-starter
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
THREAD=${WATCHDOG_THREAD:-thr_kcjvptiji9}
STATE=runs/.watchdog_last
problems=()
# Read by tools/boot_recover.sh after a reboot: which one-off units were lost.
systemctl --user list-units 'fusion-*' --state=active --no-legend --plain | awk '{print $1}' > runs/.active_units

failed=$(systemctl --user list-units 'fusion-*' --failed --no-legend --plain | awk '{print $1}')
[ -n "$failed" ] && problems+=("failed units: $(echo $failed)")

systemctl --user is-active -q ${WATCHDOG_TRAIN_UNIT:-fusion-train} || problems+=("fusion-train (training queue) is not running")

oom=$(journalctl -k --since "-16min" --no-pager 2>/dev/null | grep -i "killed process" | tail -3)
[ -n "$oom" ] && problems+=("kernel OOM kills in the last 15 min: $oom")

queued=$(grep -v '^#' runs/queue.txt 2>/dev/null | grep -cv '^\s*$')
last=$(grep -a -E '^(START|DONE|queue empty)' runs/queue.log | tail -1)
# An empty training queue is fine while something else (a CatBoost stage fit) holds the GPU.
gpu_apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
[ "$queued" -eq 0 ] && [ "$gpu_apps" -eq 0 ] && [[ "$last" == "queue empty"* || "$last" == DONE* ]] \
  && problems+=("training queue is empty, the GPU has no work: $last")

# GPU idle: three samples over 20 s, all under 5%, while the last job started over 15 min ago.
util=0
for _ in 1 2 3; do
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
  [ "${u:-0}" -gt "$util" ] && util=$u
  sleep 10
done
start_age=$(( $(date +%s) - $(stat -c %Y runs/queue.log) ))
# A job's CPU phases (reading shots, the OOF dump) leave the GPU idle for minutes: not a stall
# while a fit or dump process younger than 90 minutes is alive.
busy=$(ps -eo etimes=,args= | awk '/my_experiments\/train.py|cv_oof.py dump/ && !/awk/ && $1 < 5400' | head -1)
if [ "$util" -lt 5 ] && [ "$queued" -gt 0 ] && [ "$start_age" -gt 900 ] && [ -z "$busy" ]; then
  problems+=("GPU idle (max ${util}% over 20 s) with $queued queued jobs; queue.log untouched for $((start_age / 60)) min")
fi
# A job running far longer than any fit takes (the longest is about an hour).
if [[ "$last" == START* ]] && [ "$start_age" -gt 10800 ]; then
  problems+=("current job started $((start_age / 60)) min ago and has not finished: ${last:0:120}")
fi

[ ${#problems[@]} -eq 0 ] && { rm -f "$STATE"; exit 0; }

msg="Watchdog $(date '+%F %H:%M'): $(printf '%s; ' "${problems[@]}")"
key=$(printf '%s\n' "${problems[@]}" | sed 's/[0-9]\+ min/N min/g' | sha1sum | cut -c1-12)
if [ -f "$STATE" ] && [ "$(cut -d' ' -f1 "$STATE")" = "$key" ] \
   && [ $(( $(date +%s) - $(cut -d' ' -f2 "$STATE") )) -lt 10800 ]; then
  exit 0
fi
[ -n "${WATCHDOG_DRY:-}" ] && { echo "DRY: $msg"; exit 0; }
echo "$key $(date +%s)" > "$STATE"
printf '%s\nDiagnose, fix and restart what is broken (systemd-run --user units with MemoryMax), keep the GPU queue filled, then continue the autonomous work.\n' "$msg" \
  | bb thread tell "$THREAD" --mode auto --message-file -
echo "$msg"
