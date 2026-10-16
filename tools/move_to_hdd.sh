#!/usr/bin/env bash
# move_to_hdd.sh <src-dir>...: move bulky tokamak data from the SSD to /mnt/hdd/tokamak and leave a
# symlink at the old path, so every script and path keeps working (user's request 2026-10-16: free
# the SSD temporarily). Per directory: rsync, verify (same file count and byte total), only then
# delete the source and link. Stops on the first mismatch, leaving the source untouched.
set -euo pipefail
DEST=/mnt/hdd/tokamak
mkdir -p "$DEST"
for src in "$@"; do
  src=$(realpath -s "$src")
  [ -L "$src" ] && { echo "$src is already a link -> $(readlink "$src")"; continue; }
  name=$(echo "$src" | sed 's|^/home/pavel/||; s|/|__|g')
  dst="$DEST/$name"
  echo "$(date +%T) copy $src -> $dst"
  rsync -aH --info=progress2 "$src/" "$dst/"
  n1=$(find "$src" | wc -l); n2=$(find "$dst" | wc -l)
  b1=$(du -sb "$src" | cut -f1); b2=$(du -sb "$dst" | cut -f1)
  if [ "$n1" != "$n2" ] || [ "$b1" != "$b2" ]; then
    echo "MISMATCH $src: files $n1 vs $n2, bytes $b1 vs $b2 — source kept" >&2
    exit 1
  fi
  rm -rf -- "$src"
  ln -s "$dst" "$src"
  echo "$(date +%T) done $src -> $dst ($((b1 / 1073741824)) GB, $n1 entries)"
done
