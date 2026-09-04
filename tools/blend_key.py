#!/usr/bin/env python3
"""The cache identity of one salt's blend: its spec, and a key naming what went into it.

    uv run python tools/blend_key.py --salt 1 --pct 75      # prints "<spec>\\t<dirname>"

The directory is `s<salt>_top<pct>_<hash>`, and each part answers a question the others cannot:

  * **salt** — which 80% of the shots these members trained on.
  * **pct** — which slice of the collection by solo score, so a top-50 blend cannot be mistaken for
    a top-75 one when both exist.
  * **hash** — sha256 over each member's FILE CONTENT, not its name or mtime. Artifacts do get
    overwritten in this project; that is the 2026-08-22 incident behind lane.sh's per-arm naming.
    Without this, retraining one member leaves a cached blend that is stale, valid-looking, and
    silently a different model than its name claims.

Content rather than mtime because mtime is the weaker claim and the read is cheap: ten artifacts is
about 140 MB and under a second, against the hours the cache saves.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

LANES = Path("artefacts/lanes")
# The composition, as the blend search chose it: four bases of five models each, three wide MLPs,
# three shot mixers. The weight is how many models the artifact holds, so every model weighs 1.
TOP75 = [("z_npca_base", 5), ("z_npca_p70", 5), ("z_npca_p100", 5), ("z_sbasis_shaped", 5),
         ("c_wide", 1), ("c_widep70", 1), ("c_widep100", 1),
         ("c_lstm", 1), ("c_attn", 1), ("c_conv", 1)]
COMPOSITIONS = {75: TOP75}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--salt", type=int, required=True)
    ap.add_argument("--pct", type=int, default=75, choices=sorted(COMPOSITIONS))
    a = ap.parse_args()

    members, spec, h = COMPOSITIONS[a.pct], [], hashlib.sha256()
    for name, weight in members:
        p = LANES / f"baseline_{name}_s{a.salt}.joblib"
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            return 1
        spec.append(f"{name}_s{a.salt}*{weight}")
        d = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda fh=fh: fh.read(1 << 22), b""):
                d.update(chunk)
        h.update(f"{name}:{weight}:".encode())
        h.update(d.digest())
    print(f"{'+'.join(spec)}\ts{a.salt}_top{a.pct}_{h.hexdigest()[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
