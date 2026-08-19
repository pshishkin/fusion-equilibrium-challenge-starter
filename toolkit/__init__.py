"""Shared infrastructure: things both challenges need and neither owns.

`my_experiments/` is DIII-D's and `mast/` is MAST's, and AGENTS.md forbids either from importing
the other. A process pool and a progress bar belong to neither — but they lived under
`my_experiments/`, which meant the two top-level files that serve BOTH machines, `local_score.py`
and `submission_skeleton.py`, reached into Challenge 1's package to get them. This package is where
that kind of thing goes instead: no knowledge of a machine, a model, or the metric.
"""
