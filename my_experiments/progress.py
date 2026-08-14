#!/usr/bin/env python3
"""
Timestamps and memory on every printed line, so a run profiles itself.

    from my_experiments.progress import install_timestamps
    install_timestamps()

Every line then carries where it is in the run, how long the step that produced it took, and how
much memory the run holds at that moment:

    [  0:00.0  +0.0s   0.3G] Training on 3168 shots (45.0% of 7041, 10% of their frames), ...
    [  4:12.7 +11.3s  14.8G]   PCA: 50 components explain 100.0% of the psi variance

The gap since the previous line is the useful number: the code already prints once per stage, so
the gap IS that stage's cost. No stopwatch calls scattered through the pipeline, no decision about
what counts as a stage, and nothing to keep in sync when a stage moves.

Memory is the RESIDENT SET OF THE WHOLE PROCESS TREE, not of this process. Reading shots runs on a
pool of ~18 spawned workers, and a parent that looks small while the machine swaps is exactly the
picture that cost us an afternoon once already.

The clock starts at the first `install_timestamps()` of the process and keeps running across later
calls, so `train_eval.py` — which runs training and scoring in one interpreter — profiles both
halves on one time base.

Only stdout is wrapped. tqdm writes its bars to stderr, which stays untouched: a progress bar
redraws one line with `\\r` many times a second, and stamping each redraw would bury the log.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO

_START: float | None = None
_PROC = Path("/proc")
_PAGE = os.sysconf("SC_PAGE_SIZE")


def tree_rss() -> int:
    """Resident bytes held by this process and every descendant of it.

    Straight from /proc rather than from psutil, which is not a dependency of this repo. Costs a
    few hundred small reads, paid once per printed line — that is a handful of times per run.
    """
    children: dict[int, list[int]] = {}
    rss: dict[int, int] = {}
    for entry in _PROC.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            # Field 2 of statm is the resident set, in pages. `stat` field 4 is the parent pid,
            # read after the command name, which may itself contain spaces and parentheses.
            resident = int((entry / "statm").read_text().split()[1])
            stat = (entry / "stat").read_text()
            ppid = int(stat[stat.rindex(")") + 1:].split()[1])
        except (OSError, ValueError, IndexError):
            # The process exited between listing /proc and reading it. Not a failure to report:
            # a dead process holds no memory, which is the right answer to the question asked.
            continue
        rss[pid] = resident * _PAGE
        children.setdefault(ppid, []).append(pid)

    total, stack = 0, [os.getpid()]
    while stack:
        pid = stack.pop()
        total += rss.get(pid, 0)
        stack.extend(children.get(pid, ()))
    return total


class _Timestamped:
    """A text stream that prefixes every line with elapsed time and time since the last line.

    Wrapping the stream rather than replacing `print` is what makes this total: a line printed
    from a library, from a traceback, or from code written after this module was forgotten still
    gets stamped.
    """

    def __init__(self, stream: TextIO, start: float) -> None:
        self._stream = stream
        self._start = start
        self._last = start
        # Nothing has been written yet, so the next character starts a line and wants a stamp.
        self._at_line_start = True

    def _stamp(self) -> str:
        now = time.perf_counter()
        elapsed = now - self._start
        gap = now - self._last
        self._last = now
        return (f"[{int(elapsed // 60):3d}:{elapsed % 60:04.1f} {gap:+6.1f}s "
                f"{tree_rss() / 2 ** 30:5.2f}G] ")

    def write(self, text: str) -> int:
        if not text:
            return 0
        out = []
        for part in text.splitlines(keepends=True):
            if self._at_line_start:
                out.append(self._stamp())
            out.append(part)
            self._at_line_start = part.endswith("\n")
        self._stream.write("".join(out))
        # The caller is told how much of ITS text was written, not how much came out with the
        # prefixes added: a wrapper that reports more characters than it was given breaks the
        # contract every other writer relies on.
        return len(text)

    def __getattr__(self, name: str) -> Any:
        """Everything else — flush, fileno, isatty, encoding — is the underlying stream's."""
        return getattr(self._stream, name)

    def __enter__(self) -> _Timestamped:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self._stream.flush()


def install_timestamps() -> None:
    """Stamp every line of stdout from here on. Idempotent, and the clock survives repeat calls."""
    global _START
    if isinstance(sys.stdout, _Timestamped):
        return
    if _START is None:
        _START = time.perf_counter()
    sys.stdout = _Timestamped(sys.stdout, _START)


# How often a progress bar redraws when its output is a FILE rather than a terminal. A bar on a
# terminal overwrites one line with `\r` and costs nothing to watch; the same bar teed into
# logs/ writes every redraw as another 200 characters, and a production run buries its own
# messages under a megabyte of them.
#
# On a terminal nothing changes — the defaults are what make a bar useful to a human.
SHOT_EVERY = 1000       # reading and scoring loops, which run to thousands of shots


def bar_kwargs(every: int = 0, *, off_in_log: bool = False) -> dict[str, object]:
    """tqdm settings for a bar that may be writing into a log file.

    `mininterval` has to go to zero alongside `miniters`: tqdm redraws when EITHER threshold is
    met, so leaving the default 0.1 s would keep the per-second spam whatever `miniters` says.

    `off_in_log` turns the bar off entirely instead of thinning it, which is right wherever a loop
    already prints its own periodic line. The MLP's does, through `bar.write`, and that line
    carries the train loss, the validation loss and the best epoch with a timestamp in front —
    strictly more than the bar was showing. `tqdm.write` keeps working on a disabled bar and goes
    to stdout, so those lines still pick up `install_timestamps`' prefix.
    """
    import sys

    if sys.stderr.isatty():
        return {}
    if off_in_log:
        return {"disable": True}
    return {"miniters": every, "mininterval": 0.0}
