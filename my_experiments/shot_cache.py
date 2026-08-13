#!/usr/bin/env python3
"""
A decoded-shot cache, so the same parquet is never parsed twice.

Reading the training shots costs 62 s and a 20 GiB peak at production, against 1.5 GiB of data
actually kept: each shot carries 480256 raw magnetics samples per signal that are interpolated
down to ~222 EFIT frames and then thrown away. That work is identical on every run.

This stores the RESULT of it — features, flux and the two scalar targets, every frame — and hands
it back on the next run. Two files per shot, because the flux is the only large one and it is the
one worth memory-mapping:

    <cache>/<config>/<stem>.psi.npy      float32 (T, 65, 65)
    <cache>/<config>/<stem>.meta.npz     features, scalars, and the key below

**Bit-exact, or it is worthless.** Every number this fork has measured came out of arrays built by
`_read_training_shot`; a cache that returns something merely close would make old and new runs
incomparable while looking like an optimisation. So the stored dtypes are exactly the produced
ones — the flux is float32 because `_as_psirz_stack` produces float32, not because float32 is
smaller — and thinning happens on the way out, on the same indices, in the same order.

**Invalidation is by fingerprint, not by memory.** The key is the source parquet's size and
modification time together with a hash of the source text of every function that shapes the
arrays. Edit any of them and the key stops matching, so the cache rebuilds itself; there is no
constant anyone has to remember to bump. `CACHE_VERSION` is there for the case the hash cannot
see — a change in a library's behaviour rather than in our code.

To drop it by hand: `make clean-cache`.
"""
from __future__ import annotations

import hashlib
import inspect
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from my_experiments.models import FloatArray

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(os.environ.get("FUSION_SHOT_CACHE", REPO_ROOT / ".shot_cache"))

# Bumped by hand only for a change the source hash cannot see — a numpy or pandas upgrade that
# alters a decoded value. Our own edits invalidate the cache on their own.
CACHE_VERSION = 1


def fingerprint(*producers: Callable[..., Any], extra: object = "") -> str:
    """A key over the code that builds the cached arrays, plus anything else worth keying on.

    `inspect.getsource` reads the file, so this covers comments and formatting too. That makes it
    slightly over-eager — reformatting a docstring rebuilds the cache — which is the right side to
    err on: a stale cache is a wrong number, a rebuilt one is 80 seconds.
    """
    h = hashlib.sha1(f"v{CACHE_VERSION}|{extra}".encode())
    for fn in producers:
        h.update(inspect.getsource(fn).encode())
    return h.hexdigest()[:16]


def _paths(path: Path) -> tuple[Path, Path]:
    # Keyed by the config directory as well as the stem: two configs may name a shot alike, and a
    # collision would serve one machine's frames for another's.
    folder = CACHE_DIR / path.parent.name
    return folder / f"{path.stem}.psi.npy", folder / f"{path.stem}.meta.npz"


def _source_key(path: Path, code: str) -> str:
    stat = path.stat()
    return f"{code}|{stat.st_size}|{stat.st_mtime_ns}"


def load(path: Path, code: str, keep: Callable[[int], np.ndarray],
         ) -> tuple[FloatArray, FloatArray, FloatArray] | None:
    """Cached (features, psi, scalars) for `path`, thinned by `keep(n_frames)`, or None on a miss.

    The flux is memory-mapped and indexed, so a run that keeps a tenth of the frames reads about a
    tenth of the file rather than all of it.
    """
    psi_path, meta_path = _paths(path)
    if not (psi_path.exists() and meta_path.exists()):
        return None
    with np.load(meta_path) as meta:
        if str(meta["key"]) != _source_key(path, code):
            return None
        feats, scal = meta["feats"], meta["scal"]
    psi = np.load(psi_path, mmap_mode="r")
    if len(psi) != len(feats):
        raise ValueError(f"{psi_path.name}: {len(psi)} flux frames against {len(feats)} feature "
                         f"rows — the cache entry is inconsistent with itself")
    rows = keep(len(psi))
    return feats[rows], np.asarray(psi[rows]), scal[rows]


def store(path: Path, code: str, feats: FloatArray, psi: FloatArray, scal: FloatArray) -> None:
    """Write one shot's full-frame arrays. Written to a temporary name and renamed, so a run that
    dies halfway leaves no entry that would later be read as complete."""
    psi_path, meta_path = _paths(path)
    psi_path.parent.mkdir(parents=True, exist_ok=True)
    # Unique per process: several pool workers write different shots into the same directory.
    tag = f".{os.getpid()}.tmp"
    tmp_psi, tmp_meta = Path(f"{psi_path}{tag}"), Path(f"{meta_path}{tag}")
    # Through an open handle rather than by name: np.save and np.savez APPEND their own extension
    # to a path that does not already carry it, so a temporary name ending in .tmp would be written
    # to <name>.tmp.npy and the rename below would look for a file that was never created.
    with open(tmp_psi, "wb") as fh:
        np.save(fh, psi)
    with open(tmp_meta, "wb") as fh:
        np.savez(fh, feats=feats, scal=scal, key=_source_key(path, code))
    os.replace(tmp_psi, psi_path)
    os.replace(tmp_meta, meta_path)
