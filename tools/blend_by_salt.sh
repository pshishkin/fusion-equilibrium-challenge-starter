#!/bin/bash
# Build ONE salt's blend, cache it, and never build it again.
#
# The identity: every salt carries the same composition with the same per-model weights, so each
# salt's blend is a mean over the same total weight, and the mean of equal-sized means is the
# overall mean EXACTLY. Verified numerically on 2026-09-04: averaging two per-salt blends against
# the four-artifact blend differs by 6.8e-08 relative, which is float32's own resolution.
#
# So a submission over N salts costs one salt's predictions, not N, and adding a salt later costs
# that salt alone. Only the test configs are built — nothing here touches training data.
#
# The cache stores DECODED MAPS, which is what makes it safe where caching coefficients was not: a
# map has no invariant left to violate, and nothing downstream can reinterpret it against the wrong
# basis. The key includes the composition, so a different set of members is a different entry.
#
# Usage: SALTS="1 2 3" tools/blend_by_salt.sh          # fills .blend_cache/s<N>/
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
PCT="${PCT:-75}"
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 120; done; done
for SALT in ${SALTS:-1 2 3}; do
  # The composition and the cache identity come from one place, so the directory a blend is written
  # to always describes the blend that is in it: salt, percentile, and a hash over the members' FILE
  # CONTENT. Retraining one member changes the hash and the next build misses rather than reusing a
  # cache that quietly holds a different model.
  KEY=$(uv run python tools/blend_key.py --salt "$SALT" --pct "$PCT" 2>/dev/null) || {
    echo "salt ${SALT}: composition incomplete, skipping"; continue; }
  SPEC=$(printf '%s' "$KEY" | cut -f1)
  DIR=".blend_cache/$(printf '%s' "$KEY" | cut -f2)"
  if [ -f "$DIR/diii_d_public_test.npz" ] && [ -f "$DIR/mast_public_test.npz" ]; then
    uv run python -m tools.submission_format "$DIR/diii_d_public_test.npz" \
      "$DIR/mast_public_test.npz" || exit 1
    echo "have salt ${SALT} at $DIR"; continue
  fi
  mkdir -p "$DIR"
  echo "=== $(date -u +%F\ %H:%M) building salt ${SALT}, top ${PCT}% -> ${DIR} ==="
  # No --repo: this only builds the .npz. Pushing happens once, on the averaged result.
  FUSION_ARTIFACT="artefacts/lanes/baseline_z_npca_base_s${SALT}.joblib" \
    uv run python submission_skeleton.py --max-shots 0 --source local \
    --configs diii_d_public_test mast_public_test --jobs "${JOBS:-20}" \
    --out "$DIR" --model "salts:$SPEC" \
    && echo "  -> salt ${SALT} cached" || echo "  salt ${SALT} FAILED"
done
echo "BLEND BY SALT COMPLETE"
