#!/usr/bin/env python3
"""
Run a pre-declared list of configurations through the quality screen, one at a time, and collect
them into a CSV.

One at a time is not a limitation to work around: every run writes the same
`my_experiments/baseline.joblib`, so two at once would score each other's model.

Results are appended to the CSV as each run finishes, and a run whose name is already in it is
SKIPPED — so an interrupted sweep resumes instead of restarting, and a killed run costs one
configuration rather than the sweep. To re-run something, delete its row.

Every run also lands in `logs/<UTC timestamp>-<name>.log` with `logs/latest.log` pointing at the
newest, so `tail -F logs/latest.log` follows whatever is training right now — capital F, since
the symlink is repointed per run and `-f` would stay on the finished file — see AGENTS.md,
"Every run is watchable while it runs".

    uv run python my_experiments/sweep.py grid.json results.csv
    make sweep GRID=grids/a.json OUT=results/a.csv

The grid is a JSON list of `{"name": ..., "sets": [...], "salt": N, "shares": [...]}`; `sets` are
`--set` overrides, `shares` replaces the three default shares, and the name is both the CSV key and
the log filename, so it has to be unique and filesystem-safe.

**When `shares` changes the row count, `patience` and `epochs` have to move with it.** Both are
counted in EPOCHS and an epoch is `rows / batch_size` steps, so ten times the rows is ten times the
leash unless it is rescaled — the sweep would then measure the leash and call it data. Match the
step budget instead: `patience = 18400 / (rows // batch_size)`.
"""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG_DIR = REPO / "logs"
SHARES = ["0.60/0.1", "0.15/0.1", "0.01"]
ENV_CAPS = {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}

# `[  5:32.9   +0.0s 10.38G]                mlp    0.9873    0.9995 ...`
ROW = re.compile(r"^\[[^\]]*\]\s+(\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")
FIT = re.compile(r"(\w+): fitted in ([\d.]+) s(.*)")
BEST = re.compile(r"best epoch (\d+) of (\d+) run")
FIELDS = ["name", "S", "R2_psi", "R2_qb", "D_LCFS", "Cons", "ridge_S",
          "fit_s", "best_epoch", "ran_epochs", "wall_s", "sets"]


def parse(text: str) -> dict[str, object]:
    """Pull the scored table and the fit report out of one run's log."""
    out: dict[str, object] = {}
    for line in text.splitlines():
        m = ROW.match(line)
        if m:
            name, *vals = m.groups()
            if name == "mlp":
                out.update(zip(("S", "R2_psi", "R2_qb", "D_LCFS", "Cons"), vals, strict=True))
            elif name == "ridge":
                out["ridge_S"] = vals[0]
        m = FIT.search(line)
        if m and m.group(1) == "mlp":
            out["fit_s"] = m.group(2)
            b = BEST.search(m.group(3))
            if b:
                out["best_epoch"], out["ran_epochs"] = b.group(1), b.group(2)
    return out


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    grid_path, csv_path = Path(sys.argv[1]), Path(sys.argv[2])
    grid = json.loads(grid_path.read_text())
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if csv_path.exists():
        with csv_path.open() as fh:
            done = {r["name"] for r in csv.DictReader(fh)}
    else:
        with csv_path.open("w", newline="") as fh:
            csv.DictWriter(fh, FIELDS).writeheader()

    env = {**os.environ, **ENV_CAPS}

    for i, cfg in enumerate(grid, 1):
        name = cfg["name"]
        if name in done:
            print(f"[{i}/{len(grid)}] {name}: already in {csv_path.name}, skipping", flush=True)
            continue
        sets = cfg.get("sets", [])
        # A grid entry may carry its own shares. Growing the data is a configuration like any
        # other, and it is the one axis params.yaml cannot express — the shares are arguments.
        cmd = ["uv", "run", "python", "my_experiments/train_eval.py", *cfg.get("shares", SHARES),
               "--only", "ridge", "mlp", "--jobs", "24",
               *(["--salt", str(cfg["salt"])] if "salt" in cfg else []),
               *(["--set", *sets] if sets else [])]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        log = LOG_DIR / f"{stamp}-{name}.log"
        # Both named and symlinked BEFORE the run starts, and the child writes straight into the
        # file descriptor rather than into a buffer this process drains at the end — so
        # `tail -f logs/latest.log` follows the fit while it happens, which is the whole point of
        # keeping logs at all. Capturing to memory and writing at exit would produce the same file
        # and none of the visibility.
        latest = LOG_DIR / "latest.log"
        latest.unlink(missing_ok=True)
        latest.symlink_to(log.name)
        print(f"[{i}/{len(grid)}] {name}: {' '.join(sets) or 'baseline'}\n"
              f"    --> {log.relative_to(REPO)}  (tail -F logs/latest.log)", flush=True)

        t0 = time.perf_counter()
        with log.open("w") as fh:
            proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=fh,
                                  stderr=subprocess.STDOUT, text=True)
        wall = time.perf_counter() - t0

        if proc.returncode != 0:
            # Recorded, not swallowed: a configuration that CANNOT run is a result about that
            # configuration, and leaving a hole in the CSV would read as one that was never tried.
            print(f"    FAILED rc={proc.returncode}, see {log}", flush=True)
            row = {"name": name, "S": "FAILED", "wall_s": f"{wall:.0f}", "sets": " ".join(sets)}
        else:
            row = {"name": name, "wall_s": f"{wall:.0f}", "sets": " ".join(sets),
                   **parse(log.read_text())}
            print(f"    S={row.get('S', '?')}  fit={row.get('fit_s', '?')}s  "
                  f"best={row.get('best_epoch', '?')}/{row.get('ran_epochs', '?')}  "
                  f"wall={wall / 60:.1f}m", flush=True)
        with csv_path.open("a", newline="") as fh:
            csv.DictWriter(fh, FIELDS).writerow({k: row.get(k, "") for k in FIELDS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
