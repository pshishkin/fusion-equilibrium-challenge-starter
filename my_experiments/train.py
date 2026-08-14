#!/usr/bin/env python3
"""
Entry point 1 of 2 — fit the model zoo and save it.

    uv run python my_experiments/train.py --share 0.01     # 1% of the shots

Which models are fitted, and with which hyper-parameters, is params.yaml — not a flag here, so
there is one place to change a setting and one place to read what a run used. This script only
decides which SHOTS take part.

Shots come from the HEAD of the list ordered by sha1 of the filename, and the validation window
sits right behind them; evaluate.py takes the tail of that same list, so all three windows stay
disjoint while the shares sum to under 1. The order is deterministic, so the split reproduces on
any machine.

The validation shots are part of fitting, not of measuring: CatBoost and the MLP stop on them and
keep their best iteration. That is why they come out of the head end — only the tail is untouched
by the fit, and only an untouched fold measures generalization.

Both shares also take a frame fraction: `--share 0.05/0.1` reads 5% of the shots and keeps every
tenth frame of each. Neighbouring frames of one shot are near-duplicates, so at a fixed row budget
more shots and fewer frames each is the better buy.

Writes my_experiments/baseline.joblib, which evaluate.py and submission_skeleton.py pick up.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import DEFAULT_LOCAL_DATA_DIR, HF_TRAIN_CONFIG
from my_experiments.baseline_model import train
from my_experiments.models import DEFAULT_PARAMS_PATH
from my_experiments.progress import install_timestamps


def main() -> int:
    install_timestamps()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--share", default="0.01",
                    help="shots to train on, from the head of the list: 'shots' or "
                         "'shots/frames', e.g. 0.05/0.1 for 5%% of the shots and every tenth "
                         "frame of each (default 0.01)")
    ap.add_argument("--val-share", default="0.002",
                    help="shots to stop on, taken right behind the training window, same "
                         "'shots' or 'shots/frames' form (default 0.002). CatBoost and the MLP "
                         "pick their best iteration by it")
    ap.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH,
                    help=f"model zoo and its hyper-parameters (default {DEFAULT_PARAMS_PATH.name})")
    ap.add_argument("--jobs", type=int, default=0,
                    help="worker processes for reading shots (default 0 = cores - 2, 1 = serial). "
                         "Fitting itself is threaded by CatBoost and torch, not by this")
    ap.add_argument("--local-data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR,
                    help="root of the downloaded dataset (the folder containing 'data/')")
    ap.add_argument("--config", default=HF_TRAIN_CONFIG)
    ap.add_argument("--salt", type=int, default=None,
                    help="override split.salt for this run. The salt reshuffles which shots "
                         "train, validate and score, and it is the replicate that matters — "
                         "sweeping it from the command line beats editing params.yaml between "
                         "runs, which is how two configurations get compared by accident")
    ap.add_argument("--only", nargs="+", default=None, metavar="MODEL",
                    help="fit only these models from params.yaml and build the ensemble from "
                         "them; everything else is switched off, `ridge` included. One net "
                         "instead of four is a quarter of the fit, which is what a screen is for")
    ap.add_argument("--set", nargs="+", default=None, metavar="KEY=VALUE", dest="sets",
                    help="override one params.yaml key for this run, dotted: "
                         "--set models.mlp.batch_norm=true models.mlp.learning_rate=0.003 . The "
                         "key must already exist, and the value is parsed as YAML, so `true`, "
                         "`1e-3` and `[512, 512]` mean what they mean in the file")
    args = ap.parse_args()

    train(args.share, args.val_share, args.local_data_dir, args.config, args.params,
          args.jobs, args.salt, args.only, args.sets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
