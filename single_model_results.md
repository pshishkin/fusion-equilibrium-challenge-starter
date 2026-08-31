# Single-model results — DIII-D / Challenge 1

What the single-model screens returned. Hypotheses live in
[single_model_hypotheses.md](single_model_hypotheses.md).

## How to read a row here

**A screen result is a direction with an error bar, not a verdict.** Everything on this page was
measured on ONE model, where the paired sd of dS is 0.000283-0.000831 depending on the model against
the ensemble's 0.000155. Nothing here is established on its own, and nothing here has been shown to
survive into the ensemble.

**Kept means: positive point estimate, and eligible for a stack.** Roughly half of anything truly
worthless earns that at 6 salts, so the length of the KEPT list is not evidence of anything. The
stack's confirmation is where the evidence comes from.

**Dropped means: negative point estimate.** Equally weak in the other direction, and equally
reversible — a dropped hypothesis with a good mechanism is worth re-screening at more salts rather
than treated as refuted.

## Ladder rung 1 — one salt, six arms, `seq` alone — 2026-08-31

Ranking, not filtering (`backlog.md`, `0-ladder`). Salt 0, `--only seq`, paired against the same
control within the salt. **The table has no error bar yet**: `fit()` calls
`torch.manual_seed(self.seed)`, so re-running a configuration reproduces it byte for byte and cannot
sample initialisation noise. `S7same` supplies the only replicate available and is running.

| arm | S | dS | R2_psi | R2_qb | 1-D_LCFS | Cons | best step | wall |
|---|---|---|---|---|---|---|---|---|
| **`lstm`** | **0.990612** | **+0.000710** | 0.999678 | 0.991563 | 0.986395 | 0.967079 | 35000 | 42m |
| `base` (GRU) | 0.989902 | — | 0.999651 | 0.989730 | 0.985986 | 0.965179 | 50000 | 52m |
| `h2n128` | 0.989566 | −0.000336 | 0.999622 | 0.988270 | 0.985938 | 0.964698 | 54000 | 76m |
| `h512` | 0.989238 | −0.000664 | 0.999639 | 0.989346 | 0.985477 | 0.962433 | 35000 | 42m |
| `L2` | 0.988922 | −0.000980 | 0.999556 | 0.985665 | 0.985577 | 0.963796 | 28500 | 50m |
| `h256` | 0.988919 | −0.000983 | 0.999624 | 0.989801 | 0.985385 | 0.960587 | 41000 | 45m |

**All four capacity arms landed BELOW the control, and only the architecture change beat it.** Under
a null of no effect, four of four falling on the same side is a 1-in-16 arrangement. That is not a
result on one salt, but it is the pattern S8/S9/S10 were written in anticipation of — capacity is
not what this model is short of.

**The stopping points say why it cannot be.** Every arm ends at 14-27% of the 200000-step budget:
the fit stops because validation stops improving, not because it runs out of steps. A model in that
regime has no use for more parameters. `L2` is the largest recurrent arm, stops EARLIEST at 28500
and has the worst R2_qb of the six, which is what overfitting faster looks like — with n = 1 that is
a consistent story rather than evidence.

**Where the LSTM's gain came from, and it is the interpretable part.** `seq` trails the MLPs almost
entirely in R2_qb — 0.9897 against their 0.9939-0.9951 on this same salt — and that is where the
LSTM moved most: **+0.0018 of its +0.0007 composite comes from the q95/betaN block**. The gain
landed in the term that was actually short, rather than spreading evenly the way a lucky
initialisation would. Read against the four MLP seeds on this salt, whose spread is **0.000295**,
+0.000710 is two to three such spreads — but `seq` is the noisiest member on file (paired sd
0.000831 against `mlp`'s 0.000283), so another model's error bar cannot be borrowed. Hence `S7same`.

## Kept — awaiting a stack

| hypothesis | model | salts | mean dS | sd | ensemble dS | notes |
|---|---|---|---|---|---|---|
| `lstm` (S4) | seq | 1 (salt 0) | +0.000710 | pending `S7same` | not yet | gain is concentrated in R2_qb, the term `seq` was short in; promoted to rung 2 |

## Dropped

| hypothesis | model | salts | mean dS | sd | notes |
|---|---|---|---|---|---|
| `seq-head-2` @512 (S6) | seq | 1 (salt 0) | −0.000664 | — | capacity, not bias |
| `seq-head-2` @256 (S6) | seq | 1 (salt 0) | −0.000983 | — | worst of the six on Consistency |
| `seq-depth` L2 (S7) | seq | 1 (salt 0) | −0.000980 | — | largest recurrent arm, stops earliest, worst R2_qb |
| `seq-depth` h2n128 (S7) | seq | 1 (salt 0) | −0.000336 | — | narrowest second stage; least bad of the four |

**Dropped here means a negative point estimate on ONE salt with no error bar.** Roughly half of
anything truly worthless earns a positive estimate at six salts, so the reverse is equally weak: none
of these four is refuted, and any of them is worth re-screening if a mechanism argues for it.

## Stacks assembled

_None yet._ When one is: record the members, the SUM of their screened estimates, the stack's
measured ensemble effect, and the ratio. **Coming in far under the sum is the transfer rule eating
the overlap, and is the most informative thing a stack can report.**
