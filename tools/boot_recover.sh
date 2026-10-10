#!/usr/bin/env bash
# Run at user-manager start (fusion-boot-recover.service, WantedBy=default.target; linger is on).
# Transient fusion-* units die with a reboot: on 2026-10-08 22:22 the box rebooted and the GPU sat
# idle for 45 h until the watchdog noticed. This brings back the training queue and its log mirror
# (a fold killed mid-fit has no baseline.joblib, so cv_queue.sh simply refits it) and tells the
# agent thread which one-off units (evaluations, waiters) were lost and need relaunching by hand.
set -uo pipefail
cd /home/pavel/fusion-equilibrium-challenge-starter
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
sleep 60
since=$(date '+%Y-%m-%d %H:%M:%S')
if ! systemctl --user is-active -q fusion-train; then
  tools/fusion_run.sh fusion-train 100G 40G -p ManagedOOMPreference=avoid -p Restart=on-failure \
    -p RestartSec=120 /bin/bash tools/cv_queue.sh runs/queue.txt
fi
if ! systemctl --user is-active -q fusion-log-mirror; then
  systemd-run --user --unit=fusion-log-mirror --working-directory="$PWD" /bin/bash -c \
    "journalctl --user -u fusion-train -f -o cat --since '$since' >> runs/queue.log"
fi
# A queue hold left by a killed full*/gbm script is merged back (ExecStopPost never ran).
for h in runs/queue.hold_*; do [ -f "$h" ] && tools/queue_restore.sh "$h"; done
bb=${BB_CLI:-}
[ -x "$bb" ] || bb=$(ls -t /home/pavel/.npm/_npx/*/node_modules/bb-app/host-daemon/dist/bb 2>/dev/null | head -1)
# The hourly watchdog snapshots the active fusion-* units (a hard reset logs no stops).
lost=$(tr '\n' ' ' < runs/.active_units 2>/dev/null)
printf 'Host rebooted (%s). fusion-train and fusion-log-mirror relaunched. Units active at the last hourly watchdog snapshot: %s\nRelaunch the lost evaluations/waiters and continue the autonomous work.\n' \
  "$since" "${lost:-none}" | "$bb" thread tell thr_kcjvptiji9 --mode auto --message-file - || true
