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


## The mega-ensemble roster — kept members, 2026-09-02

**The user's decision on 2026-09-02: `p70` goes into the mega-ensemble.** It is not a config change —
that was measured and it is a zero, 7 pairs at −0.000006, t −0.14 — it is a MEMBER, and the two
questions are different. A config change has to beat the control in the composite; a member only has
to err somewhere the others do not.

**What earns it a place, measured rather than argued.** `p70` improves D_LCFS on **seven salts out
of seven**, mean +0.000252, t **+5.91** — the most reliable per-term effect anything in this file has
produced. It buys that with a Consistency change that is not established (t −1.19, 2/7) and nets to
zero in the composite. So the boundary gain is real and the composite cannot hold it: at weight 0.10
it is +0.000025, against a 12-pair resolution of 0.000088. **The prize is a third of what the
instrument can see** — which is exactly the situation where a member is the right vehicle and a
config arm is the wrong one.

And it errs in a different space by construction: 70 coordinates, not 50, so its target vector is
not the control's with noise on top.

## The error-correlation matrix — 2026-09-01, and it inverts the ranking above

Rung 1 read every arm by its own composite. That is the wrong number for this model and the file
already said so: `seq` is in the ensemble because it DISAGREES, not because it is accurate. The
matrix below is the criterion itself, computed from artifacts already on disk
(`my_experiments/diagnose_diversity.py`, 352 tail shots of salt 0, psi block, identical basis in
every arm).

**The bar is not zero, it is what a reseed already buys.** `s7same` is the control's own
architecture with a different initialisation, and it errs at correlation **0.605** with the control.
Anything above that is worth less as a member than `models.seq.seed = 1`.

| arm | corr. with control | vs the 0.605 reseed bar | single-model dS |
|---|---|---|---|
| **`conv`** | **0.546** | **below — the only one** | −0.000038 |
| `h256` | 0.608 | level | −0.000983 |
| `h512` | 0.611 | level | −0.000664 |
| `h2n128` | 0.621 | above | −0.000336 |
| `skip` | 0.624 | above | +0.000827 |
| `lstm` | 0.630 | above | +0.000710 |
| `L2` | 0.634 | above | −0.000980 |
| `attn` | 0.637 | **most correlated of all** | +0.000591 |

**Eight of nine arms err MORE like the control than the control's own reseed does.** The two that
took the best composites, `lstm` and `attn`, are among the three most correlated. By composite the
order was `skip` > `lstm` > `attn` > `base` > `conv`; by the criterion that decides membership it is
close to reversed.

`conv` is the exception and not narrowly: the lowest row in the matrix (0.491–0.593), with its
minimum against `s7same` at 0.491 — the lowest entry in the whole table. It is also the cheapest arm
at 21 minutes against 42–76, and it had the best R2_qb of the ten, which is the block `seq` is
actually short in.

**What this changed in the queue.** `lstm` was dropped from ladder rung 2, where the ladder's own
composite ranking had put it. `conv` was NOT promoted there either: its single-model composite is
already known to be level with the control, and three more salts would confirm that while answering
nothing. The question `conv` raises is about the average, so it is asked with the average — a full
production lane with `seq_cell: conv` against the current zoo.

**The caveat that keeps this honest.** One salt, one correlation measure, on the psi block. The
correlations sit in a narrow band (0.49–0.69) and no error bar has been put on the differences
between them; `conv`'s 0.546 against the 0.605 bar is a gap of the same order as the band's own
width. It reorders a queue, which is a cheap decision to get wrong, and it does not establish
anything.

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
