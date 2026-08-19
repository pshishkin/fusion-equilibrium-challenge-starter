"""Challenge 2 — MAST, zero-shot.

Deliberately shares no code with `my_experiments/`, which is Challenge 1's module. The two
machines have nothing in common but the scorer: MAST has no training data at all, one of the 21
DIII-D magnetics signals, a flux grid that starts at R = 0.06 m, and an aspect ratio of 1.3
against 2.7. A shared feature pipeline would be a shared set of assumptions, and every one of
them is false here.

What IS shared is `fusion_scoring/` — the vendored scorer — because both challenges are graded by
it, and the top-level `machines.py`, which is the only file that knows both modules exist.
"""
