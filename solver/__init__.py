"""A free-boundary Grad-Shafranov solve, and nothing that knows which challenge is using it.

`my_experiments/` is Challenge 1's and `mast/` is Challenge 2's, and AGENTS.md forbids either from
importing the other. This package is neither: it is the physics — a Green's function for the coil
field, a Picard iteration on a sparse `Delta*`, a boundary search, and the functionals read off the
result. Every machine-dependent number it needs comes in through `machine.Machine`.

It exists as its own package because of what it is FOR. The three MAST demo shots have refuted
three decisions fitted on them, so the way to choose anything for MAST is to test it where there
are labels — 7041 DIII-D shots — and the solver has to run on both machines for that to be
possible. `mast/` then holds only MAST's calibration and its predictor.
"""
