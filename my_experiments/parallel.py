#!/usr/bin/env python3
"""
One process pool for the whole run, shared by everything that is per-shot work.

Two places need it — reading shots (parquet decode plus interpolation, in `baseline_model`) and
the per-shot half of scoring (LCFS extraction and the seven functionals, in `local_score`) — and
they must not each start their own: spawn pays a fresh interpreter per worker, and paying that
twice, or once per phase, gives back most of what parallelism won.

Three rules this module exists to enforce:

* **Order is preserved.** `Executor.map` yields in submission order, so callers can zip results
  back against their input list, and float sums accumulate exactly as they did serially. A
  parallel run is bit-identical to a serial one — that is the whole point, since the score must
  not depend on how many cores the machine has.
* **Spawn, not fork.** By the time scoring runs, CatBoost's and torch's thread pools are alive in
  this process, and forking on top of a live OpenMP pool is a documented way to hang.
* **`jobs=1` really is serial**, no pool, no pickling — so anything that misbehaves under
  parallelism can be bisected with one flag.

One consequence of spawn worth knowing: the child re-imports the parent's `__main__`, so a script
piped through stdin (`uv run python - <<EOF`) dies with `BrokenProcessPool`. Every entry point here
is a file, so this only ever bites ad-hoc one-liners; write them to a file and they work.
"""
from __future__ import annotations

import atexit
import multiprocessing as mp
import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

_POOL: ProcessPoolExecutor | None = None
_POOL_JOBS = 0


def resolve_jobs(jobs: int, n_tasks: int) -> int:
    """Worker processes to use: 0 means auto, never more than there are tasks to chew on."""
    if jobs < 0:
        raise SystemExit(f"--jobs must be 0 (auto) or positive, got {jobs}")
    if jobs == 0:
        jobs = max(1, (os.cpu_count() or 1) - 2)
    return max(1, min(jobs, n_tasks))


def pool(jobs: int) -> ProcessPoolExecutor:
    """The process pool, created once. A later call asking for more workers grows it; asking for
    fewer reuses what is there rather than tearing down interpreters we already paid for."""
    global _POOL, _POOL_JOBS
    if _POOL is not None and jobs > _POOL_JOBS:
        _POOL.shutdown()
        _POOL = None
    if _POOL is None:
        _POOL = ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("spawn"))
        _POOL_JOBS = jobs
        atexit.register(_POOL.shutdown)
    return _POOL


def pimap(fn: Callable[[Any], Any], tasks: Sequence[Any], jobs: int) -> Iterator[Any]:
    """`fn` over `tasks` on `jobs` processes, yielding IN ORDER.

    Wrap it in tqdm to show progress."""
    if jobs == 1:
        # A generator, not a list: the caller's progress bar has to advance as work happens, and
        # a list comprehension would do all of it before tqdm saw the first item.
        return (fn(t) for t in tasks)
    out: Iterable[Any] = pool(jobs).map(fn, tasks)
    return iter(out)


def pmap(fn: Callable[[Any], Any], tasks: Sequence[Any], jobs: int) -> list[Any]:
    """`pimap`, collected."""
    return list(pimap(fn, tasks, jobs))
