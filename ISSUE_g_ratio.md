# Challenge 2's `G_ratio` pays for a worse DIII-D model — propose scoring `S_MAST` directly

## The problem

Challenge 2 ranks by

```
G_ratio = S_MAST / S_DIII-D          among entries with R²ψ > 0.6 on DIII-D
```

The denominator is under the entrant's control, and lowering it raises the rank. Nothing about
transfer has to improve — the same MAST predictions score better when the DIII-D predictions are
made worse.

The eligibility guard does not close this, because it constrains only one of the four terms:

```
S = 0.55·R²ψ  +  0.15·R²{q95,βN}  +  0.10·(1 − D_LCFS)  +  0.20·Consistency
```

`R²ψ > 0.6` bounds 55% of the composite and leaves the other 45% free. An entry that keeps
`R²ψ` just above the threshold and lets the three remaining terms fall to zero scores

```
S_DIII-D ≈ 0.55 × 0.6 = 0.33
```

against roughly 0.99 for an honest entry — so **`G_ratio` is multiplied by about 3.0 for free**.
Worse, `R²` and `1 − D_LCFS` are not bounded below, so a determined entry can push the denominator
lower still.

The three sacrificial terms are also the easy ones to break selectively. `D_LCFS` and
`Consistency` read the *geometry* of the flux map rather than its values, so a small, deliberate
distortion — one that barely moves `R²ψ` — collapses them. The starter kit's own README makes
exactly this point in the opposite direction: a change that "leaves R²ψ at 0.99997 while destroying
~35% of the MAST Consistency term". The same lever works on purpose.

## Why the guard cannot be patched into working

Raising the threshold, or applying one per term, does not remove the incentive — it only moves the
optimum. Whatever the floor is, the best Challenge 2 strategy is to sit exactly on it. The
incentive is created by having the entrant's own score in the denominator, and only removing it
from there removes the incentive.

## Why "one submission enters both" does not close it either

The README notes that a single submission produces both leaderboard columns, so degrading DIII-D
costs the entrant their Challenge 1 standing. That is only true for a competitor who makes exactly
one submission. If the leaderboard takes each column's best independently, one account submitting
twice already separates the two — an honest entry for Challenge 1 and a crippled-denominator entry
for Challenge 2. If it does not, a second account or team does the same thing.

Either way the metric rewards it, and detecting deliberate degradation after the fact is
considerably harder than not asking for it.

## Proposal, in order of preference

**1. Rank Challenge 2 by `S_MAST` alone**, keeping `R²ψ > 0.6` on DIII-D as the eligibility gate.

The gate already answers the question the denominator was there to answer — *did this entry
actually learn DIII-D, or is it a MAST-only fit?* — and it answers it as a threshold, which cannot
be gamed for rank, only passed or failed. What the ratio adds beyond that is an incentive to fail
better.

It also reads more honestly on a leaderboard: `G_ratio = 0.85` does not say whether MAST transfer
is good or DIII-D is bad, and those are very different results.

**2. If a ratio is wanted, put a fixed constant in the denominator.** `S_MAST / S_ref`, where
`S_ref` is a published reference — the organisers' baseline on DIII-D, fixed for the season.
That preserves "what fraction of same-machine quality survives the transfer" as the thing being
measured, while making the denominator something no entrant can move.

**3. If the entrant's own denominator must stay**, clamp it from below at a published value, e.g.
`G_ratio = S_MAST / max(S_DIII-D, 0.9)`. This is the weakest of the three: it caps the exploit at
a known factor instead of removing it, and it needs a number chosen by hand.

## What this is not

Not a claim that anyone has done this. It is a claim about what the metric currently pays for, and
the fix costs one line in the scorer.

## Suggested wording for the rules, if option 1 is taken

> **Challenge 2 — cross-machine.** Highest `S_MAST`, among entries scoring `R²ψ > 0.6` on DIII-D.
> The DIII-D threshold is an eligibility gate, not part of the rank: it confirms the entry is a
> working equilibrium model rather than a MAST-only fit.
