#!/usr/bin/env bash
# heavy.sh <MemoryMax> <command...>: run a memory-hungry command in its own cgroup (transient
# systemd user scope), in the foreground, with its exit code. MemoryHigh = 85% of MemoryMax
# (reclaim/throttle first), MemorySwapMax = 2G (an OOM kill instead of hours of swap thrash).
# The limit covers the command's whole process tree. Refuses to run — never runs unprotected —
# if the user bus is unreachable (e.g. inside a sandbox) or the limit does not fit in MemAvailable
# minus 8G of headroom for BB and everything else. See AGENTS.md, "Memory-hungry commands".
set -uo pipefail
if [ $# -lt 2 ]; then echo "usage: tools/heavy.sh <MemoryMax, e.g. 16G> <command...>" >&2; exit 2; fi
max=$1; shift
case $max in
  *G) gb=${max%G} ;;
  *M) gb=$(( ${max%M} / 1024 )) ;;
  *) echo "heavy.sh: MemoryMax must end in G or M, got $max" >&2; exit 2 ;;
esac
high_mb=$(( $(numfmt --from=iec "$max") * 85 / 100 / 1048576 ))
avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
if [ "$gb" -gt $(( avail - 8 )) ]; then
  echo "heavy.sh: MemoryMax $max does not fit: MemAvailable ${avail}G minus 8G headroom. Not started." >&2
  exit 75
fi
if ! systemd-run --user --scope -q -p MemoryMax=64M -- true 2>/dev/null; then
  echo "heavy.sh: no user systemd bus here (sandbox? DBUS_SESSION_BUS_ADDRESS/XDG_RUNTIME_DIR unset?)." >&2
  echo "heavy.sh: cannot isolate the command, so it was NOT started. Report this instead of running it bare." >&2
  exit 75
fi
exec systemd-run --user --scope -q --unit="heavy-$(date +%s)-$$" \
  -p MemoryHigh="${high_mb}M" -p MemoryMax="$max" -p MemorySwapMax=2G -- "$@"
