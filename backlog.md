# Backlog — DIII-D / Challenge 1

**One sorted list. Every entry has a slug, not a number.** Slugs are stable and mnemonic
(`shot-tail-weight`, `huber`, `ema`); the old `A22 / B9 / Z3` numbering is dead and survives only in
the closed table at the bottom, so old write-ups stay findable. `ideas.md` keeps the long reasoning
for each entry; this file is what gets read to decide what runs next.

**Sorted by expected value over cost, not by interest.** An entry moves up only when a measurement
moves it, and the reason is written in its row.

## The arithmetic every rank is read against

- **What is left**: R²ψ is 0.9998 of a 1.0000 ceiling. Everything reachable now lives in the seven
  derived functionals, worth **0.0039 of S** in total.
- **Per-scalar budget**: `dS/dR²_j = W_CONS/N_CONS = 0.20/7 = 0.0286`.
- **Screen resolution**: 16 paired salts resolve about **0.00065**. One net, one salt, is sd 0.0013 —
  so a single-salt win under +0.0028 over ten arms is selection, not signal.
- **The screen over-states variance-type changes.** `ema` cleared the gate at +0.00101 with 16 signs
  of 16 and measured **zero** in production, because the 5-member ensemble had already removed the
  trajectory noise EMA works through. Anything whose mechanism is "less variance" must be confirmed
  in production before it ships.
- **Base rate**: four of twenty iterations paid; every physics construction proposed so far has been
  refuted. Run the free gate before the run.

## Running now

| slug | what | state |
|---|---|---|
| `huber-prod` | production confirmation of the one accepted change | **DONE: 6 pairs, +0.00000, 3/6. ZERO, like `ema`.** Extending to 12 pairs |
| `seq-cosine` | `lr_schedule: cosine`, `lr_t_max_steps: 80000` for the sequence member | queued behind `huber-prod`; base arm computing on salts 2–7 |
| `tail-noise` | does the 1%→5% tail actually cut paired noise | **DONE: sample is 73% of paired variance; the tail HALVED the sd for +44% wall. Worth it.** |

## Open, ranked

**PRIORITY, set by the user 2026-08-25: the three `aux` variations come first.** They are the only
entries aimed at where the TEST SET says the loss is — `li` at R2 0.960156 and `kappa` at 0.965434,
together 6.6x the gap to first place. Everything else on this list is a general-purpose knob acting
on all seven scalars alike, and nine of those measured zero this week.

    1   aux                       extra output dimensions; gradient may bypass the coefficients
    1b  aux-surrogate-frozen      g on top of the PCA outputs, pre-fitted and fixed
    1c  aux-surrogate-trained     the same, g learning alongside the model

Run the free gradient-fidelity gate before writing any training code for 1b/1c: it can close both.


### 0-speed. SPEED IS A FIRST-CLASS GOAL, not a convenience
**User, 2026-08-30.** Everything below is rate-limited by how many arms a day the box can answer,
and the effects are small enough that breadth beats depth. Treat a change that makes runs cheaper as
worth as much as a change that makes the score higher, because it multiplies every hypothesis after
it.

**Done 2026-08-30: scoring only what is read.** A run was setup 3.0 min (7%), fitting 19.2 min
(45%), **scoring 20.5 min (48%)** — and scoring ran six members at ~2.5 min each while the lane read
one row. `evaluate.py` already had `--models`; `train_eval.py` did not forward it. It does now, and
the lane defaults to the ensemble alone. Run 43 min -> ~30, **1.4x, from not doing work rather than
from measuring less.** `SCORE="mlp ensemble"` when a per-model table is actually wanted.

**Still on the table:**
- **Halve the scoring set** (352 tail shots -> 176). Both arms score the SAME shots, so the
  set's sampling noise largely cancels in the paired difference — but that is an argument, not a
  measurement, and it needs checking before it is believed. Worth ~15%.
- **Futility stopping.** Drop an arm whose running mean is negative after 3 salts: kills 33% of
  genuinely positive arms, saves ~38% of the average arm's cost. Only acceptable in the
  accumulate-and-stack regime, where a lost hypothesis can be re-screened rather than lost.
- **Fewer MLPs is a BAD trade** and is recorded so it is not re-proposed: the ensemble is quieter
  than its members precisely because it averages four of them.

### 0-ladder. `rank-then-stack` — one salt to RANK, not to filter
**User's design, 2026-08-30, and the distinction is what makes it work.** Filtering on one salt is
hopeless: the one-salt paired sd is 0.000195 against effects of 0.00005, so at any threshold the
pass rates for a true null and a true effect are within a few points of each other — a threshold of
0.0002 discards 78% of what works to remove 85% of what does not. **Ranking asks less.** To order
candidates you do not need to distinguish them individually.

**How well one salt ranks, by how much the hypotheses truly differ:**

| spread of true effects | corr(1 salt, truth) |
|---|---|
| 0.00005 | 0.25 |
| 0.00010 | 0.46 |
| 0.00020 | 0.72 |

**The ladder: K screened on 1 salt, top half to 3, top half to 6, top two to 12.** Expected TRUE
effect of the survivors, against picking at random:

| spread | K=8 | K=16 | hours (K=16) |
|---|---|---|---|
| 0.00005 | +0.000022 | +0.000039 | 44 |
| 0.00010 | +0.000071 | **+0.000118** | 44 |
| 0.00020 | +0.000196 | +0.000280 | 44 |

At a plausible spread of 0.0001, sixteen candidates yield survivors worth **+0.000118 truly** —
twice the effect this month has been chasing — against zero for choosing at random.

**The ladder's power is entirely in BREADTH.** K=8 gives +0.000071, K=16 gives +0.000118. Eight is
not enough. This is what makes `0b. recheck` fuel rather than housekeeping: a whole class of
"indistinguishable" verdicts was returned by instruments resolving 0.0013 and 0.00065 against
effects of 0.0002, and every one of them is a free candidate for rung one.

**The trap, and its fix.** The final 12-salt number is SELECTED — those two candidates got there by
looking good — so it is inflated even when the true effect is positive. The fix is the other half of
the protocol: **survivors go into a STACK, and the stack is what gets confirmed**, on salts the
ladder never saw. The ladder produces enriched candidates; the stack produces an honest number.
Neither works alone.

### 0a. `separate-reanalyze` — screen on single models, ACCUMULATE, confirm the stack, NEXT
**User's strategy, refined 2026-08-30.** Screen each change on the single model it touches, keep
everything with a positive point estimate, combine all the keepers, and test the STACK on the
ensemble — asking there for a positive result, not necessarily a significant one.

**This repairs the objection the first version had.** The single-model measurement is not being
asked to prove anything; it only filters direction. The proving happens once, on the stack, where
the effects add and the ensemble is the quieter instrument. That inverts the earlier problem
instead of arguing with it.

**Why it works, in the measured numbers.** Ensemble paired sd is 0.000155, single-model ~0.00045.
If K accepted changes are each truly worth +0.00005 — the size these arms have been returning —
the stack is K x that, and the ensemble resolves it easily:

| K | 6 salts | 12 salts |
|---|---|---|
| 3 | t 2.4, P(p<0.05) 0.77 | t 3.4, P(p<0.05) 0.96 |
| 6 | t 4.7, P(p<0.05) 0.999 | t 6.7, ~certain |
| 10 | t 7.9 | t 11.2 |

**Three changes is already enough to make a real stack visible, and six makes it unmissable.** That
is the whole argument for accumulating rather than testing one at a time: individually these effects
are unmeasurable on this machine, and summed they are trivial to measure.

**The one hole, stated plainly.** A positive point estimate on the stack is passed by a coin flip
**50% of the time when every accepted change is truly worth nothing** — and that probability does
not improve with K, because accumulating zeros gives zero. The screen does not fix this either: at
6 salts on one model, a change worth truly nothing shows a positive point estimate 50% of the time,
and one worth +0.00005 shows it 61% of the time. **The filter enriches only slightly.**

So the stack's confirmation has to carry the weight, and "positive" alone does not:

- **Require significance on the stack, not just a positive sign.** The table above says that is
  affordable — K=3 at 12 salts already has 96% power against a real effect. Asking for p < 0.05
  there costs nothing when the effect is real and refuses the coin flip when it is not.
- **Check that the parts ADD.** Compare the stack's measured effect against the sum of its members'
  screened estimates. Coming in far under is the transfer rule eating the overlap, and that is worth
  knowing rather than averaging away.
- **Keep the screens' estimates on record**, so the stack can be decomposed if it disappoints.

**Retained from the first version, since they are unaffected by the refinement:** the ensemble is
2-5x quieter than any member (sd 0.000155 against 0.000283-0.000831, `seq` the noisiest), so the
stack must be confirmed there and not on a member; and the members sit at 0.9910-0.9912 with the
ensemble at 0.9935, so **0.0024 of the score is member disagreement alone** — thirty times the
effects being chased — and a stack that improves every member in the same way can spend it.

**User's strategy, 2026-08-30.** Stop reading every change through the ensemble. Measure the change
on the single model it touches, and **accept when the individual model is statistically better AND
the ensemble is not statistically worse.** Same for `seq`.

**The premise does not survive contact with the data, and that is the first thing this entry has to
say.** The idea is that the ensemble dilutes an individual gain. It does — but it dilutes the NOISE
harder. Measured on 12 salts of the `pca-head` arm and 12 of `cons-reweight`:

| model | dS sd, head arm | dS sd, cons-reweight arm |
|---|---|---|
| mlp | 0.000507 | 0.000283 |
| mlp1 | 0.000426 | 0.000473 |
| mlp2 | 0.000424 | 0.000375 |
| mlp3 | 0.000711 | 0.000674 |
| seq | — | 0.000831 |
| **ensemble** | **0.000209** | **0.000155** |

**The ensemble is 2-5x quieter than any member, `seq` included.** So the t-statistics go the wrong
way for the proposal: on `hstat` the ensemble reads t 2.26 while the best single model reads 1.04;
on `pca-head` the ensemble reads 1.33 against mlp2's 1.80 and everything else below 0.8. Averaging
four or five independent fits removes per-fit randomness faster than it removes a signal that is
common to all of them.

**When the proposal WOULD be right, stated so the rule can be applied where it helps.** The
individual model is the finer instrument only when the ensemble absorbs the SIGNAL more than the
noise — that is, when the change's work is already done by another member. That is the transfer
rule, and it is a claim about a specific change, not about measurement in general. A change that
touches only `seq` and duplicates what four MLPs already supply is the case to use this on; a change
that helps every member alike is not.

**The acceptance rule needs one repair before it can be used, and it is the important part.**
"Statistically better individually AND not statistically worse on the ensemble" puts the burden of
proof on one side only: the first half needs evidence, the second half is satisfied by ABSENCE of
evidence. A noisy ensemble test therefore makes the gate EASIER, which is backwards. The fix is
non-inferiority rather than a null result: require the ensemble's 95% CI to exclude harm larger than
a stated margin. At the ensemble's measured sd of 0.000155, 12 salts give SE 0.000045, so a margin
of 0.0001 is checkable and 0.00002 is not — the margin has to be declared with the arm, not chosen
after seeing the interval.

**Two more hazards worth writing down before the first arm runs:**
- **Accumulation.** Several changes each individually harmless can sum to real ensemble damage; the
  rule as stated never re-tests the stack. Re-measure the ensemble against the pre-stack baseline
  every few accepted changes.
- **Diversity.** The ensemble profits from members disagreeing — seeds correlate 0.56, feature sets
  0.75. Making every member better in the SAME way removes disagreement, and can lower the ensemble
  while raising each member. This is not hypothetical: the members sit at 0.9910-0.9912 and the
  ensemble at 0.9935, so 0.0024 of the score is disagreement alone, thirty times the effects being
  chased.

### 0. `resolution` — the instrument is the bottleneck, so it is the first thing to fix
**Everything left to test is small. That makes resolution not a detail of the method but the
binding constraint on the whole search**, and it outranks every hypothesis below because it decides
which of them can be answered at all.

**Where we actually are.** A production pair has a paired sd of 0.00025 (70 shots, `ema`) to 0.00042
(352 shots, `huber`). At 6 pairs that is SE 0.00017, so an effect must reach **0.00034** to clear
t = 2. At 12 pairs SE falls to 0.00012 and the bar drops to **0.00024**. The screen at 16 salts
resolves 0.00065 — coarser still.

**Why that bar is the problem.** The one change ever accepted, `huber`, screened at +0.00088 — barely
above it. Everything refuted this month landed between −0.0006 and +0.0005, which is *inside* the
interval. **A real +0.0003 is currently indistinguishable from nothing**, and there is no reason to
think the remaining ideas are bigger than the ones already found; the honest budget is 0.0039 of S
spread over seven functionals. So some arms already written off may be real, and there is no way to
tell without a finer instrument.

**Two levers, and they are not interchangeable.**

| lever | shrinks | cost |
|---|---|---|
| more SHOTS per run (1% → 5%, done) | only the SAMPLE component | **+44% wall per run** (1854 s → 2666 s) |
| more PAIRS per comparison (6 → 12) | **everything**, as 1/sqrt(n) | linear: ~1.5 h a pair, so ~18 h for 12 |

More pairs is the reliable lever because it shrinks both components. More shots is the cheap lever
**only if the sample really dominates** — the claim behind the 1%→5% change was that 86% of paired
variance is the shot sample, and that has not been confirmed: on live data the 5%-tail comparison
shows a LARGER paired sd than the 1%-tail one, though the comparison is confounded and underpowered.

**`tail-noise` ANSWERED IT, 2026-08-21.** Two artifacts differing in one setting, same salt, same
split, scored on 5 disjoint blocks of the 352-shot tail — so the spread across blocks is the sample
component alone. Sample sd is 0.00079 at 71 shots and **0.00035 at 352**; the observed total is
0.00041, which splits as **sample 73%, training 0.000215**. At 70 shots the total would have been
0.00082, so the 5% tail HALVED the paired sd for 44% more wall clock. The change paid.

**And shots are now spent.** The paired sd cannot go below the training component, 0.000215, however
many shots are scored — and with the production shares only 5% of shots are free at all. Pairs are
the only lever left, which settles the decision below rather than leaving it pending.

**And a second problem, now the larger one: THE SCREEN IS 0 FOR 2.** `ema` cleared the gate at
+0.00101 with 16 signs of 16 and measured zero in production. `huber` cleared it at +0.00088 with
13 of 16 and measured **exactly zero** — 6 pairs, SE 0.00017, 3/6 signs, with the Consistency gain
that was 82% of it going from +0.00359 raw to −0.00004. Two for two. The 16-salt screen fits ONE
net; the submission is a five-member ensemble which has already taken the kind of gain a single fit
shows, and both accepted changes turned out to route through terms the ensemble covers. **A screen
that has never once predicted production is not a filter, it is a cost.** Either gate it behind a
production-configuration comparison or replace it with one — about 4x a screen run, which is the
price of measuring the quantity we act on rather than a correlate of it.

**THE STANDING PROTOCOL, settled 2026-08-21: 5% tail, 24 paired salts, `--jobs 20`.**

| protocol | SE | t=2 bar | hours |
|---|---|---|---|
| 5% tail, 6 pairs | 0.000168 | 0.000335 | 9 |
| 5% tail, 12 pairs | 0.000119 | 0.000237 | 18 |
| **5% tail, 24 pairs** | **0.000084** | **0.000168** | **36** |
| 20% tail, 12 pairs | 0.000080 | 0.000160 | 38 |

**A 20% tail was priced and declined.** It buys the same resolution as 24 pairs for the same 36-38
hours, but freeing 20% of shots means cutting training from 0.80 to ~0.70 — so it would tune a model
that is not the one shipped. Pairs carry no such cost. And shots are nearly spent anyway: the paired
sd cannot fall below the training component 0.000215, so even scoring ALL 7041 shots reaches only
0.000229 against 20%'s 0.000277.

`--jobs` raised 12 -> 20: the cgroup quota is 30.7 CPUs and only ~12 were in use, because fitting is
GPU-bound and the lanes are rarely in their CPU phase together. Scoring was already parallel over
shots (`local_score._map` runs one shot per task), so this is the only speed-up available there.

**One caveat that outranks the protocol.** I keep calling 0.000215 "the training component", but it
may not be noise at all — it may be genuine HETEROGENEITY, the change really helping more on some
splits than others. If so it should not be removed: averaging over pairs estimates the mean effect,
and the spread across salts is itself the evidence on how portable a change is. Worth separating.


### 0b. `recheck` — arms that were never measured, only found indistinguishable
**Added 2026-08-22, when first place turned out to be 0.9944 against our 0.9938.** The target is
+0.0006, made of three or four changes worth +0.0002 each. That is precisely the band the OLD
instrument could not see, so a whole class of verdicts needs re-reading:

| resolution | what it could resolve |
|---|---|
| single net, single salt (grids A-D) | 0.0013 |
| 16-salt screen, the standing gate | 0.00065 |
| production paired, 12 pairs | **0.00024** |
| production paired, a small uniform change | **0.00006** |

**A refutation is only a measurement when the instrument was finer than the effect.** These were not:

| arm | point estimate | SE | signs | what the verdict actually meant |
|---|---|---|---|---|
| ~~`jac`~~ — **REFUTED in production 2026-08-23**: −0.00000, SE 0.00009, 3/6 signs. The screen's +0.00044 was below the gate AND below zero; six pairs rule out the +0.0002 it was re-opened to find | | | | |
| `bnd` — `loss.boundary = true` | **+0.00040** | 0.00029 | 8/14 | same, and never completed to 16 |
| `aux` — kappa and li as auxiliary outputs | never measured | — | 1 salt | one salt is not a measurement |
| `cos600`, `deep3` (2026-08-14) | +0.0013, +0.0012 | ~0.0009 | 2 salts | "opposite signs on two salts" is what NOISE looks like at SE 0.0009 |

**The filter that decides which are worth a production run, and it is not the point estimate.**
Two changes have cleared the screen and measured EXACTLY zero in production — `ema` and `huber` —
because the four-seed ensemble already does what they do. One has transferred: `seq-cosine`, at
+0.00011 with t = 2.5, because `seq` is a single member with no ensemble of its own to dilute it.
So the question for each candidate is *does the ensemble already do this?*

- **`jac` and `bnd` PASS the filter.** Both change what every member optimises — the metric the loss
  is estimated under, and a term (D_LCFS) that the Jacobian loss has structurally never contained
  because a distance is zero at the truth and a central difference sees no derivative there.
  Averaging seeds cannot substitute for either. **Run both in production, 12 pairs.**
- **`aux` PASSES, weakly.** An auxiliary task also regularises, which the ensemble partly does, but
  it teaches the two most expensive scored functionals directly and no ensemble supplies that.
- **`hd4`, `hd1`, `d9995`, `d99995` FAIL the filter and are NOT re-proposed.** They are variants of
  `huber` and `ema`, whose mechanisms measured zero in production. Tuning the parameter of a dead
  mechanism cannot revive it — `d9995`'s +0.00109 at 14/14 signs is the screen measuring the ema
  effect it always measured, and that effect is worth zero where it counts.

### 1. ~~`shot-tail-weight`~~ — **REFUTED 2026-08-22: +0.00002, SE 0.00003, 4/8 signs — zero, against a measured bound of +0.00106.** The knob was verified to land (the artifact's `params_yaml` carries the strength), and the reason is A22's own number: 55% of the tail residual is SHARED across seeds, i.e. bias, and re-weighting cannot supply information the inputs lack. With `shot-tail-features` also refuted, the tail family is closed on both separable mechanisms; only label noise remains, which nothing here can address.

<details><summary>the original entry</summary>

### `shot-tail-weight` — up-weight the last tenth of a shot
**ΔS +0.0005…+0.0015, bound 0.00106 MEASURED, confidence 0.45. Four lines. The best entry here.**
The final decile carries **24.1% of the geometry cost over 10.2% of the frames** — density 2.36×,
where every other decile is 0.51–1.03×. It is the model's fault, not EFIT's: the truth's second
difference there is 0.85–0.89× mid-shot while its first difference is 1.3–2.0×, so the tail moves
faster without moving less smoothly. `Model.fit` already receives a per-frame shot id and
`models.py:516` says it "accepts and ignores" it, so position within the shot is computable inside
`fit` with no pipeline change. `my_experiments/frame_weight.py` holds the designed weight, inert.
**The weight MUST be mean-normalised** or the arm measures a learning-rate change in a frame-weight
costume — the confound that made `cos600` look like a result. *Gate: refuted if the last decile
stays within 0.0003 of where it is.*

</details>

### 2. ~~`one-vs-four`~~ — **REFUTED: −0.00106, SE 0.00012, 0/6 signs.** One net with ema+huber is decisively WORSE than four seeds. The ensemble gives something neither mechanism does: independent initialisation error, which weight averaging inside one trajectory cannot touch.

</details>

### 1b. ~~`aux-surrogate`~~ — **BOTH ARMS CLOSED 2026-08-25 by the free gradient gate.** kappa correlation 0.610 against a required 0.90, and WORSE than the linearisation's 0.732 — while the surrogate's training loss was 0.00001, i.e. it fits the values almost exactly and still gets the direction wrong. That is precisely what L3's "gate on gradient fidelity, not value R2" was written to catch. Post-hoc only: on `li` the surrogate is better (0.780 vs 0.598), the second time kappa and li have split.

<details><summary>the original entry</summary>

### `aux-surrogate` — the aux loss read THROUGH the PCA coefficients. User's design, 2026-08-25
**A separate arm from `aux`, deliberately, because it fixes a specific hole in it and the contrast
between the two is the measurement.**

**The hole.** In plain `aux` the auxiliary scalars are extra dimensions of the SAME output layer, each
with its own row of weights. The gradient pressures the shared trunk, but nothing forces it through
the PCA coefficients: the net can satisfy kappa and li using trunk features that never enter the
coefficient rows. At inference only the coefficients are read, so that route improves nothing.
**And this failure mode is measured, not hypothetical** — D1 fitted a head to the seven functionals
and it knew kappa **+0.0119** better than the map did, then died at the transport step, "and the
transport would realise about half of that".

**The design.** Put a small net `g` (two layers) ON TOP of the 50 predicted coefficients, predicting
kappa and li, and add its error as a small extra loss. Now the ONLY path from the aux gradient to a
lower loss runs through the coefficients themselves. That is the mechanism I wrongly attributed to
plain `aux` when the user asked how it works.

**L3's gate applies and it is the sharp part — FREE, before any training.** The surrogate is consumed
at PREDICTED coefficients, which sit off the manifold of true equilibria, so fitting it on true
coefficients and checking value R2 measures the wrong thing:

> "Fit `g` on coefficients PERTURBED around the truth at the model's own error scale… compare
> predicted-vs-actual change of each functional along the real error direction `dc = c_pred - c_true`,
> `g`'s autograd Jacobian against the fixed M's. **Proceed only if kappa's correlation exceeds 0.90**
> (against the linearisation's 0.77) and both triangularities exceed 0.75 (against 0.45/0.46)."

**Gate on GRADIENT fidelity, not on value accuracy.** A surrogate that predicts kappa well but whose
derivative points the wrong way teaches the model to move the wrong direction, and value R2 cannot
see that.

**TWO ARMS, frozen and jointly trained — the user asked for both, and both have a real argument.**

`aux-surrogate-frozen`: `g` is pre-fitted on coefficients perturbed around the truth, then held
fixed. It is then a fixed differentiable approximation of the true functional restricted to the PCA
subspace, and its Jacobian says which coefficient directions matter FOR KAPPA specifically — exactly
what the Jacobian target metric cannot express, since that weights directions by flux geometry
rather than by their effect on one scalar.

`aux-surrogate-trained`: `g` learns alongside the model. **The argument for it is one I missed first
time and it is not weak.** The frozen surrogate is calibrated at the model's error scale AT FITTING
TIME, and that scale SHRINKS as training proceeds — so by the end the frozen `g` is tuned for a worse
model than the one it is teaching, and its gradient is least apt exactly where the predictions have
moved to. A trained `g` follows them.

**The risk on the trained side, stated so it can be checked rather than argued:** `g` and the model
can co-adapt. `g` learns to read kappa off whatever the coefficients happen to be, compensating for
the model's errors, and the auxiliary loss falls while the map does not improve at all. **Mitigation
if it is run: train `g` on BOTH true coefficients (with true kappa) and predicted ones, so it cannot
drift away from the true functional while tracking where the model lives.**

**PREDICTION, recorded before either runs: frozen wins, or they tie.** The co-adaptation failure is
the same shape as D1's transport failure, which is measured; the shrinking-error-scale argument is
an inference. **What would refute me:** `aux-surrogate-trained` beating frozen on the production
gate. **The diagnostic that separates the two explanations** costs nothing extra — read `g`'s own
auxiliary loss beside the map-derived `R2_cons`. If the trained arm's aux loss falls much further
than frozen's while its `R2_cons` does not follow, that is co-adaptation, named in advance.

**Honest counterweight, from L3's own entry.** Centre +0.0002, because `calibrate_scalars` already
corrects the linearisation's known per-scalar errors and measured exactly flat, and this fork's
recorded conclusion is that what matters is the relative weighting of directions in aggregate rather
than per-frame fidelity. Unpriced cost: a nonlinear loss term is MLP/GRU-only, where the current
loss is a linear map on targets shared by ridge too — so the ensemble would mix two objectives.

**Same production gate as `aux`:** 12 pairs, t >= 2 and >= 9/12 signs, AND the map-derived
`R2_cons` for kappa and li must move — not the surrogate's own error, which is not what is scored.

</details>

### 1c. `cons-reweight` — **CANONISED 2026-08-28 at 12 pairs, on a WEAKER reading than the one it was accepted on.**
Final: **+0.000041, SE 0.000045, t = 0.91, one-sided p = 0.191, 6/12 wins, Wilcoxon p = 0.622**,
95% CI [-0.000058, +0.000139]. The four pairs added after the decision moved it the WRONG WAY — the
estimate fell from +0.000052 to +0.000041 and p rose from 0.138 to 0.191. **This is the regression
to the mean that accepting at t ~ 1 invites, and it is recorded rather than smoothed over.** 6/12
wins is a coin flip and the median per-salt dS is near zero; the positive mean rides on s9 (+0.00032)
and s7 (+0.00027) against s11 (-0.00018).

Canonised anyway, per the user's 2026-08-27 decision, on reasoning that the weaker number does not
overturn: the point estimate is still positive and is still the best available guess, the change
costs nothing to carry, and no selection happened among splits so the estimate is not inflated.
**What DID change is the confidence, and the honest summary is now "a positive estimate the data
cannot distinguish from zero", not "a small real gain".** Reversing it is one line in params.yaml.

<details><summary>the reading at 8 pairs, on which the decision was taken</summary>

**+0.000052, SE 0.000044, t = 1.18, one-sided p = 0.138.**
The user's call, and the reasoning is decision-theoretic rather than statistical: the estimate is
positive, the change costs nothing, and — the part that matters — **no selection happened among
splits.** One split was tested, `kappa=3,li=3`, named in advance from the test-set headroom, not
picked as the best of ten tried. So the estimate is not inflated by the winner's curse and taking it
is positive expected value even at p = 0.138. 95% CI [-0.000052, +0.000157]; ~18 pairs would be
needed to clear one-sided p < 0.05 at this effect and spread.

**The mechanism WORKS, and the earlier "reallocation creates nothing" was the wrong summary.** At 6
pairs the decomposition was kappa +0.00145, li +0.00038, the other five -0.00043 combined: the two
targeted scalars gained much more than the five paid. What kills it is the scorer's own arithmetic,
not the model's budget — Consistency is the MEAN of seven, so the sum divides by 7, and then by the
0.20 weight. `+0.0014 / 7 * 0.20 = +0.00004`, which is what was measured. A gain of 0.00145 on one
functional arrives at the composite as 0.00004. **Attenuation by 35x, not absence of transfer.**

**Canonisation is DEFERRED until lane `r` finishes its 12 pairs, and the reason is a trap worth
keeping written down.** `cons_split: equal` in params.yaml is what the CONTROL arm reads — `base`
overrides nothing and inherits the default. Editing params.yaml mid-lane would silently turn the
remaining base runs into a comparison of `x3` against `x3`, and the CSV would look entirely normal.
`pca-head` was relaunched with the split pinned EXPLICITLY in both arms so it is immune to the edit
whenever it lands. **Done 2026-08-28**: params.yaml now carries the split; `pca-head` started at
01:36 with it pinned, so it measures the head ON TOP of the accepted baseline.

</details>

<details><summary>the reading at 6 pairs</summary>

**NOT refuted: +0.00003, SE 0.00004, t = 0.8, 3/6 — a POSITIVE estimate smaller than its own error.** Stopped at 6 pairs as a judgement about cost, mislabelled as a measurement, and resumed to 12 on 2026-08-27. The mechanism worked and q95/betaN were untouched (−0.00001 against `aux`'s −0.00047), but kappa +0.00145 and li +0.00038 were paid for by −0.00043 across the other five. Third mechanism, same answer: the seven share one budget and reallocation creates no value. What remains untested is the 12x headroom ratio to q95, i.e. `cons-reweight-all-9`.

</details>

<details><summary>the original entry</summary>

### `cons-reweight` — the metric splits its Consistency weight SEVEN WAYS EQUALLY, 2026-08-26
**This came out of the user asking why, if the LINEAR predictor beat the surrogate on gradient
fidelity (kappa 0.732 vs 0.610), the linear one was not trained instead.** The answer is that it
already is — and the answer exposed the gap.

**The algebra, which is why no linear surrogate should ever be built.** If kappa is linear in the
coefficients, `kappa ~ w.c`, then penalising its error is `(w.dc)^2 = dc^T (w w^T) dc` — a rank-one
quadratic form on the coefficient error. And the metric the loss already carries is

    dc^T M dc  =  sum_j (W_CONS / N_CONS) * (df_j)^2 / var(f_j)

which for linearised `f_j` is `dc^T [sum_j weight_j w_j w_j^T] dc`. **A linear aux term for kappa is
literally one of the seven summands already in M.** Adding it changes nothing but kappa's RELATIVE
weight among the seven.

**And that weight has never been touched.** `target_metric.py:288` reads
`m += (W_CONS / N_CONS) * cal[j]**2 / var[j] * block` — the same `W_CONS/N_CONS` for every one of
the seven, with no per-scalar knob anywhere. The recorded argument "the loss already up-weights the
geometry by 1434x" is about geometry DIRECTIONS against flux, and says nothing about the split
inside the seven.

**Why the split is worth moving, from the test set.** li 0.9602 and kappa 0.9654 hold 0.002126 of
unclaimed S between them — 6.6x the gap to first place — while tri_top is at 0.9909 and tri_bot at
0.9892, where almost nothing is left to take. The loss spends equal effort on all seven.

**The honest objection, which must be stated because it is not obviously answerable.** The scorer
DOES weight the seven equally, so re-weighting makes the loss deliberately disagree with the score.
The defence is that a loss should allocate by MARGINAL RETURN rather than by the score's weights: a
unit of R2 is worth the same everywhere, but a unit of gradient buys more R2 where the scalar is
further from saturation. That is an argument, not a measurement — and it is exactly the kind of
argument this project has been wrong about before.

**Cost: a per-scalar weight vector in `jacobian_form` plus one config key. No architecture change,
no extra targets, no cache.** Cheaper than `aux`, which needs 13 core-hours of derived targets per
split, and it attacks the same two scalars.

**PRE-REGISTERED 2026-08-26, before any of it runs.**

The knob is `loss.cons_split`, a comma list of `name=weight` normalised to sum to 1; `equal` is the
scorer's own 1/7 each and is what everything before today was measured on. Only the SPLIT moves —
the Consistency block always carries exactly W_CONS against the flux term, so this cannot be
confused with re-weighting geometry against flux, which is already done and by 1434x.

    arm      cons_split                                          kappa+li share
    equal    (control)                                           2/7 = 0.286
    x2       kappa=2,li=2,R_axis=1,Z_axis=1,tri_top=1,...=1       4/9 = 0.444
    x3       kappa=3,li=3,R_axis=1,Z_axis=1,tri_top=1,...=1       6/11 = 0.545

**Only `x3` runs first.** One value, chosen in advance, because a sweep before the mechanism is
established is the lambda mistake — eleven points, a clean curve, +0.000075 in sample and −0.00015
out. `x2` runs only if `x3` shows something, as the check that the response is monotone rather than
a coincidence at one setting.

**Gate:** production pairs, 12 salts, t >= 2 AND >= 9/12 signs on `S`.

**And the mechanism must move its named scalars**, read from the per-scalar block rather than from
the composite — C10's lesson. Refuted if kappa and li gain less than +0.005 of R2 each. **A second
refutation clause this arm needs and `aux` does not:** the five it takes weight FROM must not lose
more than the two gain. The scorer weights all seven equally, so a trade that moves R2 from tri_top
to kappa one-for-one is worth exactly zero, and only a genuine difference in marginal return pays.
That is the whole hypothesis, stated as something that can fail.

</details>

### 1d/1e. ~~`aux-surrogate-frozen` / `-trained`~~ — **CLOSED AGAIN 2026-08-26, on a better argument.** With 3000 frames the overfitting vanished (0.781 fitted vs 0.787 held out) and the LINEAR predictor rose to **0.905** for kappa against the surrogate's 0.787. Over the model's own error range the coefficient-to-kappa map is close to linear, so a nonlinear surrogate adds nothing there — and the linear version is already the Jacobian metric. On li the surrogate DOES win (0.907 vs 0.874), the fourth split between these two scalars.

<details><summary>the restoration entry</summary>

### RESTORED 2026-08-26 at the user's
request, to run AFTER `cons-reweight`. **Stated plainly: the pre-registered gate said no** — kappa
gradient correlation 0.610 against a required 0.90, and worse than the linearisation's 0.732 — and
these run anyway, out of curiosity, with that on the record rather than quietly forgotten.

**What could make the gate wrong, honestly listed:**
- The surrogate was small and cheaply trained: 2 hidden layers of 128, 4000 steps, ONE perturbation
  draw per frame, fitted on 592 frames from 6 shots. None of that was tuned.
- The gate is itself a proxy. It measures gradient fidelity along `dc = c_pred - c_true` at ONE
  artifact's error scale; what actually matters is whether the arm pays in production.
- 1183 frames from 6 shots is a small sample for a correlation.

**And the fix the gate points straight at, which is why re-running is not just repeating.** The
failure was specific: `g` learned the VALUES almost exactly (training loss 0.00001) and the
DERIVATIVES badly. That is exactly what **Sobolev training** repairs — fit `g` to match both `f` and
`df/dc`, not `f` alone. The derivative targets already exist: `jacobian_form` computes `df_j/dc_k`
by finite differences and its output is what the current metric is built from. So the loss becomes

    ||g(c) - f(c)||^2  +  lambda * ||grad g(c) - df/dc||^2

**Run the gate again after Sobolev training, before spending a production lane.** If the kappa
correlation clears 0.90 the arms proceed on their merits; if it does not, we will at least know
whether the surrogate idea fails because surrogates cannot do this or because that one was trained
for the wrong objective — which the first gate could not distinguish.

**Order:** `cons-reweight` first (cheaper, same target, no new machinery), then Sobolev + re-gate,
then the two arms if it clears.

### 1k. `coef-shrink` — REFUTED 2026-08-28, and the ensemble is why
Per-component shrinkage: replace each predicted coefficient by `lam_k * chat + mu_k` with
`lam_k = cov(c, chat)/var(chat)`. Motivated by a measurement — components 38-49 arrive with error
at or above their own sd, and component 49 at 1.49 sd, so predicting them is worse than predicting
the mean. Inference only, nothing retrained.

**It does not pay, on either term.**

| | R2_psi gain | Consistency gain | composite |
|---|---|---|---|
| single mlp | +0.000016 | — | — |
| ENSEMBLE | **+0.000005** | **-0.00037** | **-0.000074** |

**The ensemble had already done it.** A single model needs `lam < 0.5` on five components; the
four-model average needs it on ONE. Averaging four fits is itself a shrinkage of exactly the noise
this was built to remove — the transfer rule again, and the fourth mechanism it has eaten.

The Consistency side is a genuine surprise and worth keeping: kappa **+0.00069** and li **+0.00089**
both improve, which is the two WORST scalars and the ones with the headroom — but Z_axis loses
**-0.00459** and drags the mean under. So the tail coefficients that look like noise for the flux
are carrying real information about the vertical position, and zeroing them costs more than the
geometry gains. Shrinking only the geometry-relevant directions is a live variant; a flat
per-component rule is not.

### 1q-resid. The PCA truncation residual IS 65% predictable — and only a sixth of that transfers
**Built 2026-08-30 as the first step of `1q`, and it answers the architectural question the entry
was really asking.** A small transposed-convolution decoder, 50 coefficients -> 65x65, trained on
88k frames in three minutes.

**1. The residual is nonlinearly predictable, which is exactly what a linear basis cannot do.**
`psi - PCA_decode(c)` is orthogonal to the retained subspace BY CONSTRUCTION, so no linear decoder
can touch it. Held-out **R2 = 0.657** from the same 50 coefficients. Flux maps lie on a smooth
nonlinear manifold and PCA is a linear approximation to it; two thirds of what truncation discards
is a FUNCTION of what it keeps.

**2. On the boundary, with TRUE coefficients, that is worth +0.00018 of the composite.**

| map | 1 - D_LCFS |
|---|---|
| PCA truncated | 0.994624 |
| + recovered residual | **0.996426** |

A third of the truncation loss, recovered by a decoder that took minutes.

**3. On PREDICTED coefficients it collapses to a sixth of that.**

| map | 1 - D_LCFS |
|---|---|
| model, PCA decode | 0.989325 |
| model + residual decoder | 0.989620 |

**+0.00003 in the composite** — below the paired resolution of 0.000045 at 12 salts, so it could not
be confirmed even if it were wanted. The reason is structural: a correction computed from `c_pred`
is the residual belonging to the map that HAS those coefficients, not to the true one.

**4. So the corrector was retrained WHERE IT IS CONSUMED — and that fails decisively.** Target
`psi_true - PCA_decode(c_pred)`, the model's FULL error, fitted on validation shots the trunk never
saw and tested on a disjoint set. Held-out **R2 = -0.31, falling to -0.48** with more training —
worse than predicting zero. Applied, it makes the boundary WORSE: 0.987660 -> 0.982792, and
`R2_psi` 0.999652 -> 0.999499.

**That negative is the most useful result of the four, and it generalises.** The model's own error
carries no structure recoverable from the model's own output — **if it did, the model would already
have removed it.** Truncation was predictable precisely because it is a property of the BASIS rather
than of the model's mistakes. Any post-hoc corrector reading only the model's output is bounded by
this.

**What survives for `1q` proper.** The 0.65 figure says a nonlinear decoder has real work to do; the
+0.00003 says it cannot be bolted on afterwards. **The decoder has to be trained END TO END with the
trunk**, so the coefficients it receives are the ones it was fitted for — which is the expensive
version this entry always described, and now it has a measured reason rather than an aesthetic one.

### 1s. `field-and-envelope` — two metric terms, CODE READY 2026-08-31, not yet run
**The user's observation, and it is worth more than the experiment that produced it:** the two
things helping the encoder-decoder can be carried back to the PCA path — and there they are EXACT
rather than approximate, and free rather than per-step.

**Why exact.** With a fixed basis, any per-pixel weighting of the flux error is one (n, n) matrix:
`dpsi^T W dpsi = dc^T (B^T W B) dc`. So a weighting the encoder-decoder must apply to 4225 pixels
on every training step becomes a 50x50 matrix computed once and folded into the metric the models
already minimise. **Training cost is zero.**

**Term one, `field_weight` — and half of it was already written.** `poloidal_field_form` has been in
`target_metric.py` all along, computing `dc^T G dc = integral |grad dpsi|^2 / R dV`, the poloidal-
field error, with the physically correct 1/R weight that a naive pixel-difference Sobolev term does
not have. It was used ONLY to replace li's block under `li_form: field`; it was never a general term
of the metric. It is now, behind `field_weight`.

**The motivation is measured, not argued.** Every scored functional reads a LEVEL SET — its position
is where the flux crosses a value, its shape is the gradient there — and the Parseval term is blind
to slope. On 2026-08-31 the encoder-decoder reached `R2_psi` **0.9996** with Consistency at
**0.9306**: a map right on average and too smooth for a contour to be read out of it. That is the
failure mode this term addresses, and the PCA path has the same blindness for the same reason.

**Term two, `env_weight`/`env_inside`.** The scorer's envelope covers **57.8%** of the grid, so a
plain Parseval loss spends **42% of itself outside the plasma**, where no scored functional reads
anything. `envelope_form` is that same loss with the inside up-weighted.

**Both are normalised by their own trace against the identity's**, so a weight of 1.0 means "as much
as the flux term" rather than a number whose meaning depends on the grid.

**What the smoke test proves**, because "the code exists" and "the code is right" have diverged here
before: `envelope_form` reproduces the pixel-weighted sum to a relative error of **1.1e-16** — it is
the same quantity, not a proxy; weight 0 reproduces the previous matrix bit for bit; weight 1 adds
exactly the identity's own trace; and a degenerate form raises rather than dividing by zero.

**Not yet run — the user's call to finish the encoder-decoder first.** When it runs: `field_weight`
in {0.3, 1.0} and `env_weight` in {0.3, 1.0} against the canonised baseline, and read `R2_psi`
beside Consistency, since both terms deliberately spend flux accuracy to buy geometry.

### 1q. `upconv-decoder` — an embedding and a convolutional decoder, AS AN ENSEMBLE MEMBER
**The user's actual hypothesis, and the framing matters: they asked for it in the ensemble, not as a
replacement for PCA.** So it is judged on two axes at once — its own composite AND its error
correlation with the existing members, where seeds reach 0.602 and `n_pca=30` reached 0.625. A
different function class is the only untried candidate that could land below either.

#### Step one, run 2026-08-31: joint training helps, but not on a trunk anyone would ship
Two arms, identical trunks, data and steps; one trained in coefficient space as production does, one
trained END TO END through a transposed-convolution decoder with the loss in MAP space. The decoder's
last layer is zero-initialised, so at step 0 arm B **is** arm A.

| steps | arm | 1 - D_LCFS | R2_psi |
|---|---|---|---|
| 12000 | A coefficient loss | 0.974776 | 0.994094 |
| 12000 | B joint, map loss | 0.973153 | **0.997733** |
| 60000 | A coefficient loss | 0.977514 | 0.994636 |
| 60000 | B joint, map loss | **0.978738** | **0.998034** |

**Joint training improves the map substantially (+0.0034 of R2_psi) and, once the trunk is trained
long enough, the boundary too (+0.0012).** At 12000 steps the boundary went the other way — that
was a training-length artifact, and it is exactly the kind of number that would have been reported
as a finding if the longer run had not been done.

**Neither number transfers, and the script now says so itself.** Both trunks sit 0.0054 and 0.0020
from a perfect map where production sits at **0.0002** — ten to twenty-seven times further out. A
decoder helping there is mostly rescuing an UNDERTRAINED trunk, which is not the question. The
check is wired into the output rather than left to memory: any arm landing below R2_psi 0.999 now
prints how far it is from production's residual, because reading 0.994 as if it were 0.9998 is the
mistake this table exists to prevent.

**Why the offline test cannot close the question.** The trunk here saw 500 shots against production's
5633 and is one net against four ensembled. Getting to 0.9998 IS the production pipeline, so the
remaining step is integration, not another script: the decoder has to be carried in the artifact and
applied after `pca.inverse_transform` in `predict_row`, and `fit()` needs the PCA basis to compute a
map-space loss at all.

#### The original framing, kept: replacing PCA outright
**User's clarification, 2026-08-30: not a different PCA, but a learned embedding decoded by
transposed convolutions.** features -> latent -> upconv stack -> 65x65 map. The decoder becomes part
of the model instead of being 50 fixed images.

**The argument that PCA variants do not have, and it is a real one.** PCA is a GLOBAL basis: every
component has support over the whole grid, and a map is a weighted sum of 50 whole-grid pictures.
The LCFS is a LOCAL structure — a single contour whose position is what `D_LCFS` scores. A global
basis represents a local feature only through cancellation among many modes, which is exactly the
regime where truncation hurts most. A convolutional decoder has locality and spatial smoothness
built in, which is the right prior for a flux field, and PCA has no spatial prior at all — it treats
the 4225 pixels as an unordered vector and would give identical results if they were shuffled.

**Where the payoff would have to come from, by the numbers.** Of what is still lost:

| block | headroom x weight | share |
|---|---|---|
| Consistency | 0.004749 | 70.9% |
| **D_LCFS** | **0.001128** | **16.8%** |
| q95/betaN | 0.000707 | 10.6% |
| R2_psi | 0.000114 | 1.7% |

**Consistency is capped at 0.00055 by `1l` and that cap applies here too** — the seven derived from a
predicted map are a function of the features, so no architecture beats the best predictor of them
from those features, which the direct net estimates at 0.9778. **But `D_LCFS` was never measured
against a PCA truncation ceiling**, and it is the one term whose geometry is local and whose share is
16.8%. That is the gap this entry lives in.

**DIAGNOSTIC RUN 2026-08-30, and it OPENS the entry.** True maps projected onto the PCA components
and reconstructed, boundary extracted and scored exactly as `local_score.py` does it — N_POINTS 512,
N_ITER 22, `axis_sign = AXIS_SIGN[machine]`, `rgeo` the MEAN over frames:

| n_pca | 1 - D_LCFS with EXACT coefficients |
|---|---|
| 30 | 0.990593 |
| 40 | 0.993163 |
| **50** | **0.994777** |
| production | 0.988700 |

**Truncation costs 46% of the entire D_LCFS loss:**

| | value | in the composite |
|---|---|---|
| basis truncation, 1 -> 0.994777 | 0.005223 | **0.000522** |
| model error beyond it, 0.994777 -> 0.9887 | 0.006077 | 0.000608 |
| total D_LCFS loss | 0.011300 | 0.001130 |

**This is the opposite of what the seven showed**, where truncation cost 0.0027 against a model
shortfall of 0.021 — eight times smaller and safely ignorable. On the boundary the two are
comparable, and **0.000522 is as large as the entire measured ceiling of the Consistency direction
and larger than anything this month has produced.**

**And raising `n_pca` does not collect it** — already measured: 75 gives -0.0005 and 100 gives
0.0000 in production, because components 38-49 arrive with error at or above their own sd, i.e. as
noise. More global modes cannot help when the model cannot predict the ones it already has. **So the
target is not more dimensions but dimensions in which the boundary is PREDICTABLE**, which is
precisely what a spatial decoder offers and a global linear basis cannot.

**Caveat on the number, stated rather than buried:** measured on 1500 frames from 60 validation
shots against production's 352 tail shots, so the two are not the same sample. The truncation
figure (0.994777 against a perfect 1.0) is internally consistent and does not depend on the
production comparison; only the 46% split does.

**A first attempt at this diagnostic returned 0.9634 — BELOW production, which is impossible for a
perfect reconstruction, and that impossibility is what exposed four mismatches with the scorer:**
`n_points` 100 instead of 512, `n_iter` 8 instead of 22, no `axis_sign`, and a per-frame `rgeo`
where the scorer uses one mean. Worth keeping: the sanity check that caught it was not statistical
but physical — a reconstruction cannot score worse than a prediction of itself.

**Second reason to build it even if the first fails: it is the most decorrelated member available.**
Under `1p` the question for a new member is not its own score but its error correlation with the
existing ones — seeds reach 0.56, feature sets 0.75. A convolutional decoder is a different function
class entirely, not a different fit of the same one, so it is the strongest candidate on that axis
in the whole file. `mlp+seq` = 0.9943 against 0.9940 for `mlp` plus three seeds already says one new
architecture beats three new seeds.

**Costs, priced honestly, because this is the most expensive entry here:**
- **Parseval breaks and takes the Jacobian metric with it.** `TargetScaler` divides the psi block by
  one number only because the basis is orthonormal; `metric_form`'s identity term IS the psi loss
  because of that. A nonlinear decoder voids both, and the metric would need rederiving rather than
  reusing.
- **4225 outputs instead of 52.** Every fit, every artifact, every scoring pass.
- **The decoder ships** and is part of inference.
- **Nothing on record transfers**: `cons_split`, `cons_gain`, `jacobian_delta`, the per-component
  error profile, `coef-shrink`, all measured in the PCA basis.

**So the order is not negotiable:** the `D_LCFS` truncation diagnostic first, then — only if it
shows a real loss, or if the diversity argument is taken on its own — the decoder.

### 1p. MY MISREADING, not a user hypothesis — refuted 2026-08-31, ~4.5 h of GPU spent on it
**The user asked for one thing and I filed two.** Their words were "другой базис в ансамбле" and
then, explicitly, *"я не про pca базис, а про эмбеддинги и дальнейший upconvolution"* — the
embedding-and-decoder member, i.e. `1q` in a DIVERSITY framing. I had already written `1p` from the
first reading, did not withdraw it after the correction, and later ran a lane on it. **The
misallocation is the finding to keep; the measurement below is a by-product nobody asked for.**
`n_pca = 30` trained on three salts against the `n_pca = 50` members of the SAME salt, and judged as
the entry demanded: not by its own score but by whether its errors land elsewhere. Frame-level
squared-error correlation over 8839 held-out frames:

| pairing | mean correlation |
|---|---|
| within `n_pca=50`, four seeds | **0.602** |
| **across bases, 50 against 30** | **0.625** |

**The different basis is MORE correlated with the existing members than they are with each other.**
The entry's own criterion was "below 0.56 beats another seed"; 0.625 does not beat 0.596, let alone
0.56. A fifth member built on 30 components is worse than a fifth seed.

**And the member is worse on its own too:** S 0.992562 / 0.991791 / 0.992977 on salts 0-2 against
the 50-component ensemble's 0.9930-0.9935 on the same salts. So it fails on both counts, which is
the one combination that leaves nothing to argue about — `seq` earns its place by being worse alone
AND decorrelated; this is worse alone and not decorrelated.

**Why the premise was wrong, which is the part worth keeping.** The 2026-08-13 record showed
`n_pca=30` beating 50 at quality by +0.0043 and losing at production by 0.0014, and that reversal
made it look like `seq`'s profile. It is not: `seq` differs in ARCHITECTURE — a recurrence over the
shot against per-frame regression — while `n_pca=30` is the same model on a truncated version of
the same basis. **Truncation removes information; it does not produce a different view of it.** The
components the two share are the leading ones, which carry almost all the flux and almost all the
predictable structure, so the two models are largely fitting the same thing.

**This does not touch `1q`, and the distinction is exactly why the user's version was the right
one.** A convolutional decoder is a different FUNCTION CLASS; `n_pca=30` is a truncation of the same
one. `1q-resid` measured that PCA leaves 65% of its truncation error as recoverable nonlinear
structure — structure no PCA member of any size can hold. The refutation here is narrow: MORE OR
FEWER PCA COMPONENTS is not a diversity axis.

### 1p. `basis-diversity` — a DIFFERENT basis, the same models, added to the blend
**User's idea, 2026-08-30, and it escapes the bound that caps `1o`.** That bound — 0.00055, from the
representation-free direct net — limits how well any single model can predict the seven FROM THESE
FEATURES. Ensemble diversity is a different account entirely: members sit at 0.9910-0.9912 and the
ensemble at 0.9935, so **0.0024 of the score is member disagreement**, and a decorrelated member
claims part of it without being better at anything on its own.

**It is architecturally free, which is the surprise.** `baseline_model.py:116` already says it:
every artifact carries *"its own PCA basis, its own target scaler and its own coil gains, so their
coefficients are not commensurable"* — and `salts:a+b+c` therefore averages **decoded predictions**,
not coefficients. Mixing bases needs no new code at all. The 27-member blend already contains 27
different PCA bases, one per salt; they simply all happen to be PCA fitted on similar data, so they
agree.

**And there is a ready-made candidate with exactly the right profile.** From 2026-08-13: `n_pca = 30`
beat 50 **at quality on all three salts, +0.0043 with t = 4.84** — and lost to 50 in production by
0.0014. It was closed as a reversal. **But "better on one measure, worse alone in production" is
precisely `seq`'s profile**, and `seq` earns its place: it loses to the MLP 0.9914 against 0.9931
and still adds +0.0007 to the average. `n_pca = 30` has never been tested as a BLEND MEMBER, only as
a replacement.

**What to measure, and it is not the member's own score.** The question is whether it disagrees:
report the frame-level error correlation against the existing members, where seeds reach **0.56** and
feature sets **0.75**. A basis that decorrelates below 0.56 is a better diversity axis than a new
seed, which is the only thing that would justify it. Its own composite is almost beside the point —
`seq` proves a member can be 0.0017 worse alone and still pay.

**Cheap, and cheapest first:**
1. **`n_pca = 30` as an extra member**, 3 salts. One config field, no new code, and the blend syntax
   already accepts it. If the error correlation lands near 0.56 this whole family is worth pursuing;
   near 0.9 it closes.
2. `n_pca = 75` or `100` the same way — both measured neutral-to-negative alone (-0.0010 and 0.0000),
   so again the alone-number is not the point.
3. Only then a structurally different basis — Zernike, Fourier-Bessel, or the learned one from `1o`
   — since those cost real work and the first two settle whether basis is a diversity axis at all.

**The trap worth naming.** A blend of 4 MLPs + seq + an `n_pca=30` member has SIX members, and part
of any gain is simply having one more. The control has to be the same member COUNT — five members
plus a sixth MLP seed against five members plus the 30-component model — or the arm measures
ensemble size and calls it basis diversity.

### 1o. `learned-basis` — an encoder trained on psi AND the seven, instead of PCA
**User's idea, 2026-08-30.** Replace the fixed PCA decomposition with a learned representation whose
objective is both flux reconstruction and the seven derived scalars, so the latent is shaped by what
is scored rather than by explained variance.

**The bound comes first, because it decides how much this can possibly be worth.** `1l` measured, on
one set of held-out frames:

| stage | mean R2 over the seven |
|---|---|
| PCA ceiling, 50 components, coefficients EXACT | 0.99726 |
| **direct net, features -> 7 scalars, NO representation at all** | **0.97776** |
| production ensemble, via the map | 0.97610 |
| head on the trunk's 512-d hidden | 0.97520 |
| head on the 50 PCA coefficients | 0.97501 |

**The direct net has no bottleneck of any kind — it is the representation-free limit — and it reaches
0.9778.** So no representation, learned or otherwise, can pass that from these features. The whole
prize is 0.9778 - 0.9761 = **0.0027 in mean R2, or 0.00055 in the composite.** A learned basis is
competing for the same 0.00055 that `pca-head` took a seventh of and could not confirm.

**And the specific thing this would fix is already measured NOT to be broken.** The 50-dim
bottleneck costs 0.0002 — that is the head reading 50 coefficients (0.97501) against the head
reading the trunk's 512-d hidden (0.97520). Truncation costs 0.0027. The model's own shortfall is
0.021. **Reparameterising repairs the term two orders of magnitude too small to matter.**

**So what WOULD make this worth doing, stated as the hypothesis it has to be.** Not "represent psi
better" — PCA reaches 0.9998 and 30 components already ceiling the seven at 0.990 while the model
reaches 0.976. The only version with a mechanism is: **a basis whose coefficients are EASIER TO
PREDICT FROM THE FEATURES.** PCA orders components by flux variance, which is not the same as by
predictability — measured, components 0-3 come back at 0.022-0.033 sd of error while 38-49 arrive at
1.0-1.5 sd, i.e. as noise. A basis chosen for predictability would spend its dimensions differently.
That is a real and different objective, and it is the one to write into the loss.

**Three costs to price before building anything:**
- **Parseval breaks.** `TargetScaler` divides the psi block by one number because the PCA basis is
  orthonormal, which makes plain MSE exactly `W_PSI * SS_res/SS_tot`. A learned decoder destroys
  that identity, and the loss geometry — including the whole Jacobian metric built on top of it —
  would have to be rederived rather than reused.
- **The decoder ships.** It becomes part of inference and part of every artifact, where today the
  basis is 50 fixed images.
- **Every measurement on record was taken in the PCA basis.** `cons_split`, `cons_gain`,
  `jacobian_delta`, the per-component error profile, `coef-shrink` — all of them would need
  re-reading, and none would transfer automatically.

**The cheap test that comes first, and settles whether to build the expensive thing.** Do not build
an encoder. Ask whether a basis chosen for predictability exists at all: take the current
coefficients, fit the model, and measure how much of the total coefficient error lies in a subspace
that could be rotated away — i.e. whether the error is concentrated in directions PCA happened to
choose or is isotropic. **If the predictable subspace is already 30-dimensional and the remaining 20
are noise in ANY basis, there is nothing to rotate toward and this entry closes for free.** That is
a diagnostic on existing artifacts, hours not days, and it is the honest first step.

### 1l. What actually limits Consistency: the FEATURES, not the basis
**Measured 2026-08-28 on one set of held-out frames, so the levels are comparable to each other.**

| stage | mean R2 over the seven |
|---|---|
| basis ceiling, 50 components, coefficients EXACT | **0.99726** |
| direct net, features -> 7, nothing else | 0.97776 |
| production ensemble, via the map | 0.97610 |
| head reading the trunk's 512-d hidden | 0.97520 |
| head reading the 50 predicted coefficients | 0.97501 |

And by component count, with exact coefficients: n=20 gives 0.95844, n=30 **0.99038**, n=40 0.99503,
n=50 0.99726.

**Thirty components already ceiling at 0.990 and the model reaches 0.976 — it sits two points below
the ceiling of a basis half the size.** The 50-dim bottleneck costs 0.0002 (side head against top
head), truncation costs 0.0027, and the model's shortfall is 0.021. Reparameterising fixes the term
that is two orders of magnitude too small to matter. **This closes the "try another basis"
direction, which was the user's own question, on the user's own evidence.**

What remains is the gap between the production route (0.9761) and everything achievable from the
features (0.9778): **0.0027 in mean R2, or 0.00055 in the composite** — twice the gap to first
place. It is reachable only by changing what the TRUNK learns, since a head reading the frozen
trunk's output cannot find it: coefficients and hidden carry the same functional information to
within 0.0002. That is exactly the mechanism `pca-head`'s gradient tests.

**Caveat on the ceiling, stated rather than buried:** the direct net is ONE net at 512x3 for 40k
steps. It is a lower bound on what the features support, not an upper one, so 0.978 may understate
the room. It is a fair estimate of the ROUTE's loss, which is what the comparison is for.

### 1m-run. RUNNING 2026-08-29 — the gate was OVERRIDDEN, not passed
`pca-head` closed at one-sided **p = 0.1048** against a gate of p < 0.1. **The user's call to run the
sweep anyway.** Recorded as an override rather than folded into the gate, so the record does not
later read as though 0.1048 cleared 0.1 — that distinction is the only thing a pre-registered
threshold is for.

The case for overriding is real and worth stating: the estimate is positive, 9/12 salts favour it,
and the miss is 0.0048. The case against is equally real: this same arm read p = 0.058 at five pairs
and the sixth overturned it, so p near the threshold is exactly where this measurement has already
proved unstable once.

**Arms as agreed:** `w003` (head_weight 0.03) and `w03` (0.3), 12 salts, outer loop over salts so
every variant sits inside the same split. **0.1 needs no re-running — `q_trained_s0..s11` is that
arm — and no-head is `q_base_s0..s11`.** So four weights (0, 0.03, 0.1, 0.3) are compared on 12
shared salts for the cost of 24 new runs rather than 48.

**Two things to check when it reads, both consequences of what the sweep is FOR:**
- **The variant-to-variant sd, measured rather than assumed.** The base-against-trained sd is
  0.000204; variants differing in one scalar and sharing a split should be tighter, and how much
  tighter decides whether the selection-inflation table above is binding.
- **Monotonicity.** Four weights spanning 0, 0.03, 0.1, 0.3 either trace a curve with an interior
  optimum or they do not. A ragged non-monotone pattern across four levels is the signature of
  noise, and would say the sweep is measuring nothing regardless of which level scores highest.

### 1m. `head-weight-sweep` — how hard should the head push?
**User's request, 2026-08-27.** `head_weight` was set to 0.1 by an argument, not a measurement: the
Jacobian metric already carries `W_CONS = 0.20` on those directions, so 0.20 here would double the
emphasis rather than add a signal. That reasoning fixes an ORDER, not a value, and 0.03 or 0.3 are
equally consistent with it.

**Gate, set by the user 2026-08-27 and made numeric 2026-08-28: run it only if `trained` reaches
one-sided p < 0.1 over its 12 pairs.** If the head is worth nothing at 0.1, its best weight is a
question about noise. Candidates `{0.03, 0.3}` against the running 0.1.

**Design, the user's, and it is better than the one this entry first recorded.** Loop over SALTS on
the OUTSIDE and run every weight variant INSIDE each salt. That makes the variants paired with each
other, not merely each paired against its own control: all of them see the same train/val split, the
same shot order, the same everything except the weight, so the salt-to-salt variance — which is the
dominant term, roughly 0.001 against an effect of 0.0001 — cancels out of every variant-to-variant
comparison. The earlier plan of one arm per lane would have compared variants ACROSS salts and paid
that variance in full.

Two consequences worth stating:

- **The selection-inflation table below is now conservative.** It was computed from the base-against-
  trained per-pair sd of 0.000204, which is the right number for comparing a variant to no-head. The
  variant-to-variant differences are paired AND the configurations differ only in one scalar, so
  their sd should be smaller — but by how much is unknown until measured, so **report it from the
  data rather than assuming it**, exactly as the 0.000204 was.
- **The design stays balanced at every salt count.** Because a salt contributes all variants or
  none, the sweep can be read at 4, 6 or 12 salts without the arms drifting out of alignment, and
  a decision to stop early costs comparability rather than validity.

The base (no-head) arm does NOT need re-running: `q_base_s0..s11` already exists and is the same
configuration, so only the new weights cost anything.

**The selection cost, measured on THIS screen rather than assumed.** Per-pair sd of dS is 0.000204,
against `cons-reweight`'s 0.000155: **the mlp-only screen is 1.3x noisier, which is what dropping
seq costs.** If every candidate weight were identical in truth, best-of-K would still score this
far above the field by chance:

| K | 6 pairs | 12 pairs |
|---|---|---|
| 3 | +0.000071 | +0.000050 |
| 4 | +0.000086 | +0.000061 |
| 6 | +0.000106 | +0.000075 |

The screen's own effect is currently +0.000162. So three candidates at 12 pairs carries +0.00005 of
selection noise — a third of the effect, tolerable — while six at 6 pairs carries +0.00011, which is
two thirds and is not. **Three arms, 12 pairs, or use the split-half protocol from 1h-b.**

### 1n-canon. `pca-head` CANONISED 2026-08-30 with the STATIC split, on a cost the user accepted
`head_weight: 0.1` and `head_cons_weights: kappa=3,li=3,...` in all four `torch_mlp` blocks.

**The user's reasoning, and the numbers support it:** not established as a gain, but almost
certainly not a loss. Twelve salts give **+0.000065, 95% CI [-0.000033, +0.000163]** — the interval's
worst case is a third of the point estimate's size, so the downside is bounded and small.

**The cost is not small, and it lands exactly on the priority set an hour earlier.** A run with the
static head takes **+29.6%** longer, paired per salt (2527 s to 3256 s). `0-speed` had just bought
**-30%** by scoring only what is read. **The head gives that back in full**, to the digit. Recorded
because the two decisions were made an hour apart and the arithmetic between them is not obvious
from either entry alone.

Reverting is one line, and a lane that needs the speed more than the 0.000065 can pin
`models.*.head_weight=0.0` per arm without touching the default.

### 1r. `head-weight-003-static` — the one combination the sweep never ran
**User's request, 2026-08-30.** The weight sweep tested 0.03, 0.1 and 0.3 with the EQUAL split; the
split arms tested equal, static and adaptive at 0.1. **0.03 with the STATIC split is the corner
neither covered**, and it is the interesting one for a specific reason: at 0.1 static and equal were
indistinguishable (t -0.30), but static's per-pair sd was the smallest of the three arms
(0.000154 against 0.000209 and 0.000191). A quieter arm at a lower weight is where a small real
effect would show first if there is one.

**And it is the cheap direction.** The canonised head costs +29.6% of every run; 0.03 should cost
the same wall time (the head's expense is the auxiliary forward and backward, not the weight) but
would press the trunk a third as hard — so if it matches 0.1 it is not cheaper, and the reason to
prefer it would be that it perturbs less for the same result.

**Screen:** against the canonised 0.1-static, paired within salt, 6 salts. Both arms carry a head,
so this measures the WEIGHT alone. The no-head control already exists as `q_base_s0..s11`.

**Pre-registered:** the three weights at the static split will be 0.03, 0.1 and (from the sweep, at
equal split) 0.3 — **do not read a winner across them**, since 0.3 was measured at a different split
and the sweep's own lesson was that four levels traced no curve.

### 1n. CLOSED 2026-08-30 — all three head splits land in the same place, none of them anywhere
Twelve salts, `head_weight` 0.1 throughout, every arm paired within salt against the same no-head
control.

| arm | mean dS | sd | SE | t | p | wins |
|---|---|---|---|---|---|---|
| equal | +0.000080 | 0.000209 | 0.000060 | 1.33 | 0.105 | 9/12 |
| static kappa=3 li=3 | +0.000065 | 0.000154 | 0.000044 | 1.46 | 0.087 | 8/12 |
| adaptive | +0.000042 | 0.000191 | 0.000055 | 0.77 | 0.230 | 9/12 |

**Nothing separates them:** static minus equal -0.000016 (t -0.30), adaptive minus equal -0.000038
(t -0.50), static minus adaptive +0.000022 (t 0.44). Three ways of splitting the seven, one answer.

**The adaptive arm produced the finding, and it is not about heads.** The feedback rule converged to
a fixed point reproducible across **32 independent fits** (4 models x 8 salts, different seeds and
splits): kappa 1.48, li 1.82, R_axis 0.96, volume 1.13, Z_axis 0.59, tri_top 0.47, tri_bot 0.55.
That point correlates **0.971** with `1 - R2_j` of the head measured two days earlier by a different
script — i.e. **the loop's answer was computable in advance and the loop added nothing.**

**And the reason it lands there is the real result.** `w_j ∝ MSE_j` is a fixed point only if
reweighting does not change the errors. Raise kappa's weight and its error should fall, dropping its
weight and moving the equilibrium. It does not move: 32 fits settle on the profile of the
UNWEIGHTED head. **So reallocation does not change per-scalar accuracy — the loop is measuring, not
steering.** That is the fourth mechanism to say so (`aux`, the surrogate, `cons-reweight`, this), and
the first to say it structurally rather than through a null.

**Three corrections I made on this arm, all recorded rather than quietly fixed:**
- I reported "adaptive over-presses and damages the map" from `dR2_psi = -0.000042` on 7 salts.
  Paired, that was t -1.06, p 0.33. Reading means without testing them.
- I wrote that a static split inside the head is `cons-reweight` one layer up and not worth running.
  It ran best of the three at 7 salts. The simplification was wrong: in the Jacobian metric the
  split competes with psi through `dc^T M dc`; in the head's loss it acts on a separate term that
  never mixes with psi.
- **p = 0.03 collapsed on the next salt three times in this experiment** — 5 pairs 0.058 -> 6 pairs
  0.292; 5 salts 0.032 -> 6 salts 0.565; 7 salts 0.032 -> 9 salts 0.244. At this sd, p near 0.03
  below about ten salts means nothing, and that is an instrument fact, not a fact about heads.

**The head family is closed.** Weight 0.03 and 0.1 indistinguishable, 0.3 harmful, frozen mildly
harmful, three splits identical, and the best reading of the best arm is p = 0.087 against a
measured direction ceiling of 0.00055 of which it takes about an eighth.

### 1n. `head-cons-weights` — weight the SEVEN inside the head by where the error actually is
**User's proposal, 2026-08-27, and the second half of it is the part that has not been tested.**
The head today weights the seven equally after standardising each by its own spread, which
reproduces the scorer: Consistency is the mean of seven pooled R2 and R2 already divides by SS_tot.

**Static reweighting here is NOT a new mechanism and should not be run as one.** `cons-reweight`
already moved weight between the same seven, in the Jacobian metric, and measured +0.000041 at 12
pairs — a positive estimate the data could not separate from zero. A static split inside the head is
the same idea one layer up and should be expected to measure the same.

**The ADAPTIVE version is different and is the entry worth running.** Weight each scalar by its
CURRENT error during training — `w_j ∝ 1 - R2_j` recomputed every evaluation — so effort follows
where the fit is actually losing instead of where it was losing when the config was written. This
is a feedback loop, not a fixed allocation, and none of the three refuted reweighting arms tested
it. The scorer-reproducing equal split is the null it must beat.

**Two things to get right, both learned the hard way here:**
- Recompute on VALIDATION, not training error, or the weights chase the fit's own overfitting and
  the loop amplifies it.
- Normalise the seven to a constant sum, as `cons_weights` already does, so the arm moves the SPLIT
  and not the head's total strength — otherwise it is confounded with 1m.

**Extending it to the psi error, 2026-08-28 — the natural version is wrong and the reason is
structural.** The user proposed adding psi to the adaptive rule, since psi error is also a sum over
50 scalars. But the two blocks are not the same kind of object:

- **psi is ONE pooled R2** over every pixel: `R2_psi = 1 - sum_k err_k^2 / SS_tot`, one denominator
  for all fifty. A coefficient's contribution to the SCORE is its ABSOLUTE squared error, so equal
  weights are the correct weights — which is exactly what `TargetScaler` does, one scalar divisor
  over the whole block (Parseval).
- **the seven are SEVEN separate R2, averaged.** Each carries its own denominator because the
  scorer computes them independently and means them afterwards.

Weighting the fifty by their own current relative error would turn one pooled R2 into fifty
separate ones, which is not the quantity being scored. The failure direction is predictable and has
already been measured: tail components have large RELATIVE error and negligible absolute
contribution — zeroing components 38-49 entirely moves `R2_psi` by **0.000005** (see `coef-shrink`).
An adaptive relative rule would send most of the weight exactly there.

**The version that IS right, and is better than what was asked for.** Adapt the balance BETWEEN the
score's blocks rather than within psi: psi, q95/betaN, D_LCFS and the seven, each weighted by how
much of the LOST SCORE it currently holds. That is one number per block, it keeps every block's
internal structure intact, and it is the self-tuning form of `1i cons-gain` — which is static and
was itself motivated by the same table:

| block | weight | headroom x weight | share of what is left |
|---|---|---|---|
| R2_psi | 0.55 | 0.000114 | 1.7% |
| R2_qb | 0.15 | 0.000707 | 10.6% |
| D_LCFS | 0.10 | 0.001128 | 16.8% |
| Cons | 0.20 | 0.004749 | 70.9% |

An adaptive block rule would find the same direction 1i sets by hand, and would keep finding it as
the fit moves. **Run 1i first anyway:** a static gain of 2 or 4 answers whether the direction pays
at all, for one field and no feedback loop, and an adaptive scheme that cannot beat a constant is
machinery for nothing.

**Pre-register the reading, since an adaptive scheme can always be tuned after the fact:** equal
split as control, one adaptive arm, 12 pairs, and the per-scalar R2 reported beside the composite so
"it helped kappa and li and lost it elsewhere" stays visible. That pattern has appeared in every
reweighting arm so far — most recently `coef-shrink`, where kappa +0.00069 and li +0.00089 were
buried by Z_axis -0.00459.

### 1j. `pca-head-trunc` — let the head read only the components the model can predict
**Follows from a measurement, 2026-08-28, not from an idea.** Against a trained artifact, the
model's per-component coefficient error, in units of that component's own sd:

| components | error / sd | meaning |
|---|---|---|
| 0-3 | 0.022-0.033 | predicted almost exactly |
| 4-12 | 0.059-0.087 | predicted well |
| 15-25 | 0.138-0.319 | degrading |
| 30-37 | 0.474-0.633 | poor |
| 38-49 | 1.0-1.5 | **worse than predicting the mean — pure noise** |

So a dozen of the head's fifty inputs carry no information at consumption time. **This also explains
the one number the jitter change did NOT move.** Without jitter the pre-fit reaches 0.00827 on true
coefficients (head R2 0.9917 across the seven) but arrives at 0.02716 when fed the model's own
predictions (R2 0.9728) — and WITH the 0.5 jitter it was 0.02736, i.e. identical. The 3.3x
degradation is not off-manifold-ness of the kind jitter models; the leading components barely move.
It is the tail arriving as noise.

**Sharp, falsifiable prediction:** a head reading only the first ~30 coefficients should close most
of the 0.00827 -> 0.02716 gap. If it does not, the explanation above is wrong and the degradation is
something else. Cheap: one field, `head_n_pca` already exists as the boundary the head is told about.

### 1f. `pca-head` CLOSED 2026-08-29 — **+0.000080, t 1.33, one-sided p 0.1048, gate NOT passed**
Twelve pairs, mlp-only screen, trained head, weight 0.1, split pinned to the canonised `x3`.

| | mean d | sd | SE | t | wins |
|---|---|---|---|---|---|
| S | +0.000080 | 0.000209 | 0.000060 | 1.33 | 9/12 |
| R2_psi | +0.000002 | 0.000016 | 0.000005 | 0.41 | 7/12 |
| **R2_qb** | **+0.000133** | 0.000232 | 0.000067 | **1.98** | 7/12 |
| D_LCFS | +0.000092 | 0.000213 | 0.000061 | 1.50 | 6/12 |
| Cons | +0.000251 | 0.001082 | 0.000312 | 0.80 | 9/12 |

95% CI [-0.000052, +0.000213]; Wilcoxon 0.1245. **The user's gate was one-sided p < 0.1 for the
weight sweep; 0.1048 does not pass it, by 0.0048.** Recorded as failed rather than rounded, because
the threshold was set before the data and moving it afterwards is the whole thing a pre-registered
gate exists to prevent. ~22 pairs would be needed for p < 0.05 at this effect and spread.

**The accumulation history is the most useful thing this arm produced, and it belongs in the record
because it is an argument about METHOD rather than about heads:**

| pairs | mean | t | p |
|---|---|---|---|
| 4 | +0.000162 | 1.59 | 0.105 |
| 5 | +0.000158 | 1.99 | **0.058** |
| 6 | +0.000066 | 0.58 | 0.292 |
| 8 | +0.000046 | 0.55 | 0.300 |
| 11 | +0.000059 | 0.95 | 0.182 |
| 12 | +0.000080 | 1.33 | 0.105 |

A random walk around a small positive value. **Had this stopped at five pairs it would have reported
p = 0.058, passed its own gate, and launched a weight sweep built on a number the very next pair
overturned.** That is the same failure `cons-reweight` was stopped into at six pairs, and the reason
the user's instruction to run all twelve was right.

**What survives unexplained: `R2_qb` at t 1.98, 7/12, the strongest single term — in a block the
head does not touch.** Consistency, its actual target, reads t 0.80. Three readings and nothing yet
separates them: chance (one of four terms at t~2 over twelve pairs is roughly what noise gives with
no multiplicity correction); real transfer, the head improving the coefficients generally with
q95/betaN most sensitive to it; or something in the q95/betaN block interacting with the head other
than through the seven. **It is the one thread worth pulling if this direction is revisited.**

**Read alongside 1l:** the direction's measured ceiling is 0.00055, and this took +0.00008 of it,
about a seventh, at a cost of ~20% more fit time. And this is the mlp-only screen, so the number
does not even transfer at full value.

### 1f-frozen. CLOSED at 6 pairs 2026-08-28 — a frozen head is mildly HARMFUL
Stopped by the rule pre-registered below, and the rule fired on the reading it named.

**-0.000054, SE 0.000075, t -0.71, one-sided p 0.746, 2/6**, CI [-0.000247, +0.000140]. Not merely
null: `R2_psi` -0.000031 and Consistency -0.000201, both negative, on an arm whose whole purpose was
to add a Consistency signal.

**The three-way comparison is what this arm bought, and it is the cleanest number of the set:**

| comparison | mean dS | SE | t |
|---|---|---|---|
| trained vs no head | +0.000066 | 0.000113 | 0.58 |
| frozen vs no head | -0.000054 | 0.000075 | -0.71 |
| **trained vs frozen, paired within salt** | **+0.000119** | **0.000063** | **1.89** |

The third is cleaner than either of the others because both arms carry a head, pre-fitted
identically from the same starting point, and differ in exactly one thing: whether it keeps
updating. Split, shot order and head initialisation are all shared, so what is left is
co-adaptation alone.

**What it says.** A frozen head pours into the trunk a gradient calibrated for a trunk that no
longer exists, and that is mild damage. A trainable head at least tracks. But neither beats no head
convincingly, so the value here is not in the auxiliary signal — it is in the head being ABLE to
follow. That is exactly what the `headv/val` trajectory said at salt 0 on 2026-08-28, before any
paired number existed: the ratio held constant at 0.882 with sd 0.024, the head following the map
rather than leading it.

### 1f-stop-rule. Pre-registered 2026-08-28, BEFORE the deciding pairs existed
**The user's call, written down while `frozen` had 3 pairs and `trained` had 6, so it cannot be a
decision taken after seeing the answer.**

- **`frozen` stops at 6 pairs** if it keeps reading as it does now: -0.000003, SE 0.000056, t -0.05,
  1/3, per-salt -0.000097 / -0.000007 / +0.000096. A null this flat does not become interesting with
  six more pairs — at this SE, six pairs already resolve anything worth acting on, and the arm's
  purpose was to ask whether a head that cannot co-adapt still helps. It does not.
- **`trained` runs the full 12.** It is the arm with a live question: +0.000066 at 6 pairs, but that
  is +0.000158 at 5 pairs and +0.000066 at 6, so one salt (-0.000395) moved it by more than its own
  mean. That is exactly the case where stopping early is the mistake `cons-reweight` taught — it was
  halted at 6 pairs on a judgement about cost, which I mislabelled as a measurement, and the user
  corrected it.

**Note what the frozen arm's low SE means, since it is a result rather than an accident.** Its
per-pair spread is half the trained arm's. A frozen head delivers the same auxiliary gradient shape
every step and cannot chase the trunk, so it adds less variance — and also adds nothing. The trained
arm's larger scatter is the head co-adapting, which is where both the hope and the noise live.

### 1f-result-so-far. The head tracks the map instead of adding to it
**Read at salt 0 of the screen, 2026-08-28, before any paired S exists — recorded now so it cannot
be fitted to the answer afterwards.**

The pre-registered co-adaptation test was: if the head's own loss falls much further than the
map-derived score follows, the two met in the middle. **It does not fall further. It does not fall
differently at all.** Over 46 evaluations of the first mlp, Pearson(val, headv) = 0.976, and the
ratio headv/val sits at 0.882 with sd 0.024, drifting -0.019 across the whole fit — less than its
own scatter.

That is not collusion, and it is not encouraging either. A head carrying signal the map loss does
not already contain would show the ratio FALLING as the trunk learned the new thing. A constant
ratio says the head is measuring what the validation loss already measures. **This is what the
transfer rule predicts for a null, and it is visible before the paired number exists.**

The direct check agrees: per-model best validation loss, same salt, head against no head, is
+1.58%, +1.58%, -4.96%, -0.58% across the four MLPs — a within-run spread of 6.5 points that
swallows any systematic shift. One salt cannot answer this; 12 pairs of S can. **Caveat kept
explicit: val is MSE in target space while the score reads the map and the seven functionals, which
is the entire reason the Jacobian metric exists. A head could move the composite without moving
val.** So this sharpens the expectation toward zero without settling it.

### 1f-screen. `pca-head` measured on the MLP ensemble ALONE, 2026-08-28
**User's call, and the arithmetic backs it.** The head is attached to `mlp`/`mlp1`/`mlp2`/`mlp3`
only, so `seq` is bit-identical in both arms — confirmed on salt 0, where both recorded seq best
0.030951. It was 45.1 min of the 66.0 min of fitting, **68%**, spent producing the same numbers
twice. Dropping it takes fitting to 20.9 min and the whole run from 92 min to roughly 45, so 12
pairs cost ~18 h instead of ~37.

**This is a SCREEN and its number does not transfer to production, which is the point to keep
straight.** The transfer rule says a change measures zero when its work is already done by anything
else in the configuration — and `seq` is a different architecture reading the same shots, so in the
full ensemble it plausibly supplies some of what the head supplies. Removing it removes that
competition. **The mlp-only difference bounds ONE thing and not another, and conflating them is a
mistake I made until the user caught it on 2026-08-28.** It bounds how much THIS configuration —
`head_weight` 0.1, equal weights over the seven, reading all 50 coefficients, trained rather than
frozen — transfers to the full ensemble, because a competitor has been removed. It does NOT bound
what the head DIRECTION can reach: not one of those four choices has been measured. 0.1 came from
an argument about ORDER (the Jacobian metric already carries `W_CONS = 0.20`, so 0.20 here would
double rather than add), the seven are weighted equally by default, a dozen of the fifty inputs
arrive as noise (1j), and the frozen arm has not run. **The direction's ceiling is the 0.00055
measured in 1l, not the screen's number** — and that is itself a lower bound, the direct net being
one untuned fit. A positive screen still has to be confirmed with `seq` back in before anything is
canonised. A null screen, by
contrast, is close to conclusive: if the head cannot move an ensemble built entirely of the models
it is attached to, it will not move one where it is diluted.

Records to `x_pca_head_screen.csv`, 12 pairs, arms `q_base` / `q_trained`, split pinned. The one
completed full-ensemble run is kept as `x_pca_head_full_1pair.csv` rather than mixed in — two
different configurations must not share a file.

### 1f. `pca-head` — layers over the PCA output trained on the TRUE kappa and li, values only
**User's design, 2026-08-26. The gradient-fidelity gate that closed the surrogate arms DOES NOT
APPLY here, and the distinction is the whole point of the entry.**

**The design.** The net outputs the 50 coefficients as now; a small head `g` sits on top of them and
predicts kappa and li; the extra loss is `||g(c) - kappa_true||^2` against the CACHED GROUND TRUTH.
Trunk and head train together. No derivatives anywhere.

**Why the gate does not apply.** In `aux-surrogate-frozen`, `g` STANDS IN for the true functional:
the model is taught by `g`'s derivative, so if `dg/dc` disagrees with `df/dc` the model is pushed the
wrong way, and L3's threshold is the right test — which it failed at 0.610 against 0.90. Here `g` is
not standing in for anything. It is a readout trained on real labels, and the requirement is not
"reproduce `df/dc`" but "let `c` carry enough for a small net to read kappa off it". A head can be a
useful training signal while being a poor approximation of `f` away from the data.

**Why it is not plain `aux` either.** In `aux` the auxiliary scalars are extra ROWS of the same
output layer, each with its own weights, so the trunk can satisfy them by routes the coefficients
never see — and at inference only the coefficients are read, so that route buys nothing. Here the
head reads ONLY `c`, so the only way to lower the auxiliary loss is to put the information into the
coefficients themselves. That is the property plain `aux` cannot enforce and the frozen surrogate
tried to buy with a gradient it did not have.

**What it shares with the closed arms, and therefore what to watch.** Co-adaptation: head and trunk
can meet in the middle, with the auxiliary loss falling while the map does not improve. **Read the
head's own loss beside the map-derived `R2_cons`** — if the first falls much further than the second
follows, that is the failure, named in advance. It is the same diagnostic written for
`aux-surrogate-trained` and it costs nothing.

**Cost:** a head and one loss term. No cache beyond `.aux_targets/`, which is already built for
salts 0-5; no derivatives, so none of the Jacobian machinery. Cheaper than either surrogate arm.

**Gate:** production pairs, t >= 2 and >= 9/12 signs, AND kappa and li must move by >= +0.005 of R2
each in the map-derived block — the same rule as `aux`, for the same reason: the mechanism has to
move the scalar it names.

### 1i. `cons-gain` — raise Consistency AGAINST psi in the training metric
**User's proposal, 2026-08-27, and it is a better target than the one I named an hour earlier.** I
had written that the machinery should be aimed at `R2_psi`, "weight 0.55, no averaging
denominator". That reasoning counted the weight and forgot to check the HEADROOM, and the headroom
says the opposite:

| block | weight | mean level | headroom | x weight | share of what is left |
|---|---|---|---|---|---|
| R2_psi | 0.55 | 0.999793 | 0.000207 | 0.000114 | **1.7%** |
| R2_qb | 0.15 | 0.995284 | 0.004716 | 0.000707 | 10.6% |
| D_LCFS | 0.10 | 0.988720 | 0.011280 | 0.001128 | 16.8% |
| Cons | 0.20 | 0.976256 | 0.023744 | 0.004749 | **70.9%** |
| | | | | 0.006698 | = 1 - S, checks |

**psi is saturated.** Its 0.55 is the largest weight in the metric and it is buying 1.7% of what
remains: even a PERFECT map from here is worth +0.00011. Consistency holds 71% of everything left
on 20% of the weight. Per unit of metric weight the headroom ratio is **114x** in Consistency's
favour. The fit is spending most of its capacity defending a term that has almost nothing left to
win.

**This also reframes `cons-reweight`'s 35x attenuation.** That number said a gain inside
Consistency arrives at the composite divided by 35. True — but Consistency's headroom is 44x psi's
in absolute score, so even after the division it is the larger prize. `cons-reweight` moved weight
BETWEEN the seven, which is a zero-sum move inside one block. This moves weight INTO the block, and
the two are not the same experiment.

**Implementation is one factor, and the metric is already written for it.** target_metric.py:433
builds `m = eye(n) + (psi_ss_tot / W_PSI) * m_cons` — the identity IS the psi term (TargetScaler
having already divided the psi block by `sqrt(psi_ss_tot / W_PSI)`), and the ratio between the two
is therefore pinned at exactly the scorer's. A `cons_gain` multiplier on the second term is the
whole change. Note line 438 scales the BOUNDARY term by the same factor, so `cons_gain` must apply
to `m_cons` alone or D_LCFS moves too and the arm measures two things.

**Pre-registered, so the reading cannot be chosen afterwards:** `cons_gain` in `{2, 4}`, both
against the accepted `x3` baseline, 6 pairs each. Two candidates and not eight — 1h's table says
best-of-8 at 6 pairs carries +0.00007 of selection noise, and two is the most that can be afforded
without the split-half protocol. **The gate is `R2_psi`:** it must not fall by more than 0.00002
(which is 10% of psi's entire remaining headroom), or the arm is buying Consistency with the one
term that has nothing to give.

**The honest risk.** psi being at 0.9998 does not prove it is cheap to hold there — it may be
saturated precisely BECAUSE the metric defends it, and the seven may be at 0.976 because they are
genuinely harder rather than underweighted. That is what the two gain levels test: if the gate
trips at gain 2, the answer is the former and this whole direction closes.

**Sibling, same argument, different block:** D_LCFS holds 16.8% of the headroom on 0.10 of the
weight, and target_metric.py:438 already has its own separate term. `bnd-gain` is the same one-line
change on `m_bnd` and should be run only after `cons-gain` reports.

### 1h. `cons-split-sweep` — search the seven weights instead of naming one
**User's request, 2026-08-27, and the entry exists mostly to say how NOT to run it.** `x3` was one
split, named in advance from the test-set headroom. A sweep asks the obvious next question — is
`kappa=3,li=3` the best of them — and in asking it destroys the exact property that made `x3`
worth accepting at p = 0.138: that no selection happened.

**The inflation is quantified, not feared.** Per-pair sd of dS is 0.000125, measured. If every
candidate split were IDENTICAL in truth, the best-of-K would still score this far above the field
by chance alone:

|  K splits | 6 pairs | 12 pairs | 24 pairs |
|---|---|---|---|
|  4  | +0.000052 | +0.000037 | +0.000026 |
|  8  | +0.000073 | +0.000051 | +0.000036 |
| 16  | +0.000090 | +0.000064 | +0.000045 |

Eight splits at six pairs each is 96 fits, three days of the box, and returns a winner carrying
+0.00007 of pure selection noise — **larger than the entire +0.000052 effect being hunted.** Even
sixteen splits at twenty-four pairs, which is a month, does not get the inflation below the effect.
A naive sweep cannot answer this question on this machine.

**The precedent is already on disk and it is the same shape.** `lambda_cv` swept one scalar knob,
picked the in-sample optimum, and measured it on held-out shots: in-sample +0.000054, out-of-sample
**-0.000300**. The sweep reported a gain five times its own true value with the sign flipped.

**So run it in this order, and stop at the first one that answers:**

**1h-a. `cons-split-derived` — DERIVE the weights, do not search them.** One arm, no selection, no
inflation. Set each scalar's weight proportional to its own headroom `1 - R2_j` measured on the
public test set (li 0.0398, kappa 0.0346, and the five others an order of magnitude smaller). That
is a single pre-registered hypothesis with a stated reason, exactly like `x3` but with the ratio
taken from the data rather than rounded to 3. **Cheap: 6 pairs, ~18 h, comparable directly against
`x3`.** If the derived split beats `x3`, the question "is 3 the right number" is answered without a
sweep ever running.

**1h-b. If a sweep still happens, it must be split-half, like `lambda_cv`.** Choose the winner on
salts 0-5, MEASURE it on salts 6-11, and report only the second number. Nothing else is honest at
this sd, and the cost is doubled on top of the table above — which is itself the argument for
trying 1h-a first.

**What a sweep would be for, if it clears.** Not the +0.00004 on the composite: `cons-reweight`'s
own decomposition shows the transfer is real (kappa +0.00145) and attenuated 35x by Consistency
being a mean of seven under a 0.20 weight. A better split still pays that 35x. The reason to look
is that the SAME machinery aimed at `R2_psi` — weight 0.55, no averaging denominator — would keep
what this loses.

### 1g. `cons-reweight-all-9` — move weight BETWEEN the q95/betaN block and the seven
**User's proposal, 2026-08-26. `cons_split` only moves weight inside `W_CONS`; this opens the split
across all nine scalars the loss actually contains.**

**What the board says, and it is not what I expected.** Per-scalar, from the 2026-08-25 test set:

| scalar | R2 | unclaimed R2 | weight now | S available |
|---|---|---|---|---|
| li | 0.960156 | **0.039844** | 0.0286 | 0.001138 |
| kappa | 0.965434 | 0.034566 | 0.0286 | 0.000988 |
| R_axis | 0.978202 | 0.021798 | 0.0286 | 0.000623 |
| volume | 0.984592 | 0.015408 | 0.0286 | 0.000440 |
| Z_axis | 0.984992 | 0.015008 | 0.0286 | 0.000429 |
| tri_bot | 0.989229 | 0.010771 | 0.0286 | 0.000308 |
| tri_top | 0.990932 | 0.009068 | 0.0286 | 0.000259 |
| **betaN** | 0.996402 | **0.003598** | **0.0750** | 0.000270 |
| **q95** | 0.996677 | **0.003323** | **0.0750** | 0.000249 |

**q95 and betaN carry 2.6x li's weight with 12x less unclaimed R2.** That is the scorer's own
arithmetic — `W_QB = 0.15` over two scalars against `W_CONS = 0.20` over seven — not a mistake of
ours, and the loss inherits it.

**Why NOT simply weight by headroom, which is the obvious move.** `aux` has just measured the
exchange rate inside Consistency at roughly **1:1**: kappa +0.0029 and li +0.0031 against R_axis
-0.0024 and Z_axis -0.0024, even though those pairs differ two- to three-fold in headroom. So
marginal return does NOT track headroom the way the table invites you to assume, and a blanket
proportional reweight would be betting on exactly the thing just refuted.

**What is still open is the EXTREME ratio.** Between q95 and li the headroom differs 12x, not 2x —
a regime `aux` never probed. "A nearly saturated scalar has little marginal return" is untested
there and is the only version of this hypothesis still standing.

**The asymmetry that sets the size of the move.** q95 and betaN are SUBMITTED directly and carry
0.15 of the score. The whole prize on li and kappa is 0.002126. If a starved qb slipped from 0.9965
to 0.99, that is **-0.00098** — three times the gap to first place. There is far more to lose here
than to win, so the arm is deliberately small.

**PRE-REGISTERED:**

    arm      W_QB   W_CONS   what moves
    control  0.15   0.20     as shipped
    qb10     0.10   0.25     a third of the qb block, given to li and kappa only

Not the proportional split (which would cut q95/betaN to 0.10x): that is a bet on headroom-tracking
that `aux` argues against, and its downside is the steep one.

**Gate:** production pairs, t >= 2 and >= 9/12 signs. **Plus a hard stop that outranks the gate:**
if `R2_qb` falls by more than 0.001 the arm is abandoned whatever `S` does, because that is the
regime where the loss is unbounded and the gain is capped.

**Code:** `jacobian_form` already takes `cons_weights`; this needs the same for the qb block and a
`qb_weight` config key. Half a day, no new targets, no cache.

### 2b. `one-vs-four`, the original entry — replace the four-seed ensemble with ONE net carrying ema + huber
**The user's proposal, and the converse of tonight's two zeros.** `ema` and `huber` both measured
exactly zero in production *because* the four-seed ensemble had already taken their gain. If that is
the reason, a single net that gets both should capture much of what the ensemble buys — at a QUARTER
of the training cost. The ensemble is worth +0.0009 (0.9931 -> 0.9940); the two screened at +0.0019
combined on ONE net. They will not add cleanly, but the order is the same. Queued; the control arm
is free, since `h_base_s0..s5` already are exactly "four seeds, no ema, no huber" on the same salts.

### 2c. ~~`coef-expansion`~~ — **REFUTED 2026-08-23, free.** A test-time scale on the PCA
coefficients. The in-sample sweep peaked at lambda 1.005 for +0.000075, next to D5's predicted
median of 1.0095; two disjoint halves of the same tail then disagreed on the DIRECTION (1.0050 vs
1.0000) and the mean out-of-sample gain was **-0.000150**. The bias of best-of-eleven was larger
than the effect. Every future free sweep reports two columns: in-sample, and on shots the sweep
never saw.

### 2d. ~~`seq-capacity-dropout`~~ — **h512 closed at +0.00004, t=1.6, 7/10 over 10 pairs**; the MLP variant (`wide+dropout`) stopped at 1 pair to focus on `aux`. Neither is where the test set puts the loss.

<details><summary>the original entry</summary>

### `seq-capacity-dropout` — PRE-REGISTERED 2026-08-23, before `h512` returned
**The user's proposal, written down while the arm it depends on is still running.** If
`seq_hidden` 256 -> 512 does not pay on its own, run 512 WITH dropout.

**Both branches are interesting, for opposite reasons, and saying so in advance is the point:**
- **If `h512` is flat or negative**, the natural reading is that the extra capacity overfits — 5633
  sequences against 1.25M rows for the MLP, three orders of magnitude fewer examples per parameter.
  Then dropout is the fix, and 512+dropout should beat both 256 and 512 alone.
- **If `h512` pays**, 512+dropout is still worth one run, because a larger model has more to
  regularise and dropout at 0.1 was only ever measured at 256 — where it starved a model that had
  no capacity to spare.

**What the record already says, and it cuts both ways.** `seq-dropout` at 256 measured -0.0039 and
made the model UNDERFIT: train stuck at 0.030 against the baseline's 0.012, val 0.0445 against
0.0320. Note precisely what that does and does not show — dropout always raises TRAIN loss, since it
is noise injection; the damning part is that VAL rose too. So dropout is bad at 256. It says nothing
about dropout at 512, which is the question here.

**Honest counterweight:** the same combination was refuted for the MLP (A15) — width -0.0006/-0.0011,
dropout 0.02->0.1 at -0.0025 — and its conclusion was "this model is not short of parameters, the
ceiling is in the INPUTS". If that generalises to `seq`, this dies with it.

**SWEEP SMALL VALUES, not just 0.1 — added on the user's instruction 2026-08-25, and the record
supports it.** `seq` has only ever been tried at **0.1**, where it was catastrophic (-0.0039, the
model UNDERFIT: train stuck at 0.030 against 0.012). Everything below that is untested on this
member. A15's MLP grid measured the shape of the curve: **0 -> 0.02 is FLAT, 0.02 -> 0.1 costs
-0.0025** — so the harm climbs steeply and the whole interesting region sits under 0.05. Grid to run:

    0.01  0.02  0.03  0.05   (and 0.1 already measured at -0.0039)

Pre-registered reading: if the curve is monotone down from 0.01, dropout is simply wrong for this
model and the family closes. If there is an interior optimum near 0.02-0.03, the capacity-limited
diagnosis was right and 512+that value is the arm to confirm. A15's flat 0->0.02 on the MLP is a
weak prior AGAINST finding anything, and it is weak precisely because `seq` sees 5633 sequences
where the MLP sees 1.25M rows — three orders of magnitude fewer examples per parameter.

</details>

### 2e. `recheck-a15` — the MLP capacity grid, whose own write-up calls itself provisional
A15's record says, in as many words: *"Every cell of this grid was fitted at a learning rate that was
later measured to blow up in 15 of 28 runs at these shares... **Re-run it at `grad_clip: 1.0` before
spending anything on the strength of it**"* — and each cell therefore carries a coin flip worth about
0.002, against marginal means of -0.0001 to -0.0011. **`grad_clip` is now 1.0.** The stated
precondition is met and the re-run has never happened, so the whole "capacity is not the constraint"
conclusion — which demoted every "make it bigger" idea in the file — rests on a grid its author
asked to be redone.

### 2f. `seq-weight` — **NOT ESTABLISHED 2026-08-25: two artifacts, opposite answers** (+0.000174 on `lanes/a`, −0.000039 on `lanes/c`). The optimum is a property of the fit, not of the design. Confounded — `a` has cosine and `c` does not, and they are different salts — so a third sweep is needed to separate them, not an argument.
**Currently every member carries 1.0**, so `seq` gets exactly 20%. Those weights were fixed before
`seq` gained its cosine schedule (+0.00010 confirmed, t = 5.1), and nothing has revisited them since.

**This is NOT A21, which is closed.** A21 FITTED all five weights by optimisation and
cross-validated at **−0.00002** — worse than equal. Five free parameters on a handful of shots
overfit, exactly as that entry says. Setting ONE weight by hand is a different object: one parameter,
swept coarsely, and with a mechanism behind it rather than a fit.

**The mechanism is measured.** `mlp` alone is 0.9931, four seeds 0.9940, `mlp+seq` **0.9943** — one
extra architecture beats three extra seeds, because seeds correlate at 0.56 and a different
architecture does not. If `seq` contributes more per member than an MLP seed does, 20% may be low.

**Free: this is a deterministic RESCORE, no training.** `--models` accepts weighted combinations
(`a*0.3+b*0.7`), and every member is already in the artifact.

**Gate, and it is not optional:** the λ sweep taught this the hard way — an eleven-point sweep
returned +0.000075 in-sample and **−0.00015** on shots it had not seen. So report two columns: the
best weight found on one half of the clean tail, and what that weight earns on the other half.
Refuted if the out-of-sample column is ≤ 0, or if the two halves disagree on the direction.

### 2g. `more-seeds` — 4 → 8 → 12, RE-OPENED 2026-08-25 on changed grounds
**Priced at +0.0002…+0.0003 and DECLINED on 2026-08-20** — for double the MLP training cost, when
cheaper wins still looked available. **Nine production arms later, one survived at +0.00010.** The
benefit side of that ledger has not changed; the alternatives have. +0.0002 for double cost is now
among the better deals on the list, and three of them close the 0.0006 gap to first place.

**What the decline was based on, kept because it bounds the upside honestly.** A pure 1/n variance
model predicts 4→8 seeds at +0.0007, diluted to +0.0006 by the 4-of-5 member share. But an empirical
bound already exists here: the shipped trio averages **12 MLPs plus a seq across THREE feature sets**
and scores 0.9950 against prod's 0.9945 — so tripling members WITH feature diversity on top is worth
+0.0005 in total. Seeds inside ONE feature set must be worth less, because seeds sharing features
share systematic error; the measured saturation ("it saturates at three") says the same thing.

**So the honest centre is +0.0002, not +0.0007**, and the 1/n law is the wrong model. Run 8 first;
12 only if 8 pays, since the increment shrinks as 1/n.

### 3. `disagreement-weight` — weight frames by how much the seeds disagree
**Oracle +0.00159 measured; expected +0.0004…+0.0008. One prediction pass, ~6.5 h.** The highest
oracle of anything unbuilt, and it needs no new physics — the seeds already tell us where they are
unsure. `my_experiments/frame_disagreement.py` is scaffolded only.

### 3b. ~~`shot-tail-features`~~ — **REFUTED 2026-08-21, free.** The vessel integrals do nothing for
the tail: absolute last-decile cost is 0.00119 / 0.00118 / 0.00119 across `prod` / `vesseld` /
`vesselp`. Shares would have said otherwise, but only because `vesselp` is a worse model overall.
This raises `shot-tail-weight`, now the only remaining candidate for the measured 0.00106 — and it
should be run TWO-SIDED: the first decile is also expensive (15.1% of cost, density 1.48x).


### 4-6. MOVED 2026-08-30 to [single_model_hypotheses.md](single_model_hypotheses.md)
`seq-ema` + `seq-huber`, `seq-sweep` and `ema-patience` each change ONE model and share nothing
with the other members, so they are screened on that model rather than through the ensemble. The
numbering keeps its gap deliberately: entries here are keyed by slug, and renumbering would break
every reference written before today.

### 7. `mlp2-block0` — one seed, one block, an order more damage than typical
Found 2026-08-21 in passing: in block 0 of the clean tail, `mlp2` scores **0.977925** where the
other three seeds score 0.9825–0.9881 and every other block has all four at 0.990+. Free to chase —
find the shot, then check whether it is the same one the tail-expert diagnostics keep landing on.

### 1. ~~`aux`~~ — **REFUTED 2026-08-26: −0.00016, SE 0.00005, t = −3.1, 0/3 signs.** Not a null: li rose on all three salts and kappa on two, so the mechanism works — but R_axis and Z_axis fell by the same amount and q95/betaN on top. A one-for-one trade inside an equally-weighted seven is worth zero. This is the `cons-reweight` objection arriving early through another door.

<details><summary>the original entry</summary>

### `aux` — teach `kappa` and `li` directly, PROMOTED FROM 8 ON TEST-SET EVIDENCE, 2026-08-25
**The leaderboard breakdown moved this from the bottom of the list to the top, and the evidence is
external rather than argued.** `li` sits at R2 0.960156 and `kappa` at 0.965434 on the TEST set —
together **0.002126 of unclaimed S, 6.6x the 0.00032 gap to first place**, and 51% of everything
Consistency is missing. Meanwhile R2_psi is 0.999832 of a measured 1.0000 ceiling: the map is
finished and every remaining point is in scalars derived FROM it.

**Why it is not redundant with the Jacobian metric, which is the argument that demoted it.** The
entry was ranked down because "the loss already up-weights the geometry directions 1434x". That is
true and it is the wrong object. The metric knows which DIRECTIONS in flux space matter — components
20-49, collectively. It cannot express that two of the seven SCALARS the scorer reads carry two
thirds of the loss, because it never sees those scalars. An auxiliary target does exactly that.

**Why the fit is what is failing, not the representation — C2, 2026-08-15, unused until now.** "Of
the 0.0055 remaining, 0.0045 is reachable by FITTING and 0.0010 is the basis… the basis can produce
those scalars to 0.99+; the fit does not." `kappa`'s basis ceiling is 0.9908 against 0.9654 achieved:
0.025 of its R2 is lost to fitting alone.

**PRE-REGISTERED, before any of it runs:**

    arm 1   loss.aux_scalars=kappa,li   loss.aux_weight=0.1     (the entry as written)
    control the same base, aux off

- **Gate:** production pairs, 12 salts. Accept at t >= 2 AND >= 9/12 signs, on `S`.
- **The mechanism must move its named scalar.** C10's lesson: if `S` twitches but kappa's and li's
  own R2_cons do not, the arm is not doing what it claims and does not ship. Read `--terms` and the
  per-scalar block, not just the composite.
- **Refuted if** kappa and li move less than +0.005 of R2 each — that is 0.00014 of S apiece, below
  what 12 pairs resolve, so a smaller move cannot be acted on anyway.
- **If it pays, sweep `aux_weight` SECOND, never first.** A weight sweep before the mechanism is
  established is the λ mistake: eleven points, a pretty curve, and −0.00015 out of sample.

**Honest counterweight:** `aux` measured FLAT once already. But that was at the old resolution,
where the gate was 0.00065 and the effect being sought is 0.0003 — the same "indistinguishable
recorded as zero" class this whole `recheck` section exists for.

**Cost:** the per-frame targets are ~31 ms a frame, about 13 core-hours per training split, cached
by the exact file list. Caches for salts 0-5 built 2026-08-25 on cores the GPU lane leaves idle.

### 8. ~~`aux`~~ — the original entry, superseded by position 1 above
**Re-priced DOWN 2026-08-20 and kept only for honesty.** The loss already up-weights the geometry
directions **1434×** (components 20–49 hold 0.021% of the flux variance and take 30% of the loss),
which is most of what `aux` would say. An auxiliary task also regularises, so it is not refuted —
but it measured flat once, needs per-salt caches (~24 h), and ranks below `disagreement-weight` by
an explicit decision recorded so it is not re-litigated by intuition.

### 9. Cheap diagnostics, no training
`perm-importance` (permutation importance on the artifact we already have) · `residual-time` (the
residual's structure in time) · `seq-memory` (how far back the recurrence actually reaches) ·
`mast-transfer` (does anything here transfer) · `board-gap` (read the 704-shot rescore as the
local-to-board gap, not as a new fold).

### 10. Expensive and unpriced
`hyperparam-search` (~40 runs) · `diverse-ensemble` (8–10 members, ×10 training) · `scorer-calibration`
(~200 scoring passes) · `gs-cleanup` (Grad-Shafranov at inference, ~1 day) · `qb-head` (dedicated
q95/βN head — but R²qb is already 0.9967 of a 1.0000 ceiling) · `catboost-prod` · `frames-axis`
(more frames per shot at fixed shots) · `thomson-derivative` · `reversed-ip` · `per-shot-offsets` ·
`shot-aggregates` · `m-pool-reweight` · `functional-surrogate` · `regime-loss` (needs the loss
reworked).

## Closed — one line each, so nothing is re-proposed

| slug | closed by |
|---|---|
| `huber` | **ACCEPTED** +0.00088, 13/16. Threshold two-sided: 1.0 → −0.00058, 4.0 → −0.00036 against 2.0 |
| `ema` | accepted on the screen (+0.00101, 16/16, t=7.2), **zero in production** — the ensemble already did it |
| `seq-dropout` | −0.0039 on the first pair. Made the model UNDERFIT: train never below 0.0298 against 0.012 |
| `stopcomp` | +0.00009, 3/15 — the composite monitor's sample size was not what held it back |
| `jac` | +0.00044 — the metric weights were already right |
| `bnd` | +0.00040, and it does not move D_LCFS at all |
| `more-seeds` | seed correlation 0.56 → an infinite ensemble floors at 56%; 4 seeds took three quarters |
| `more-feature-sets` | correlation **0.75**, worse than seeds → floor 75%; a 4th set is 83%→81% |
| `bigger-basis` | kappa's ceiling climbs ~0.003 per ten components against a 0.027 gap; `npca75` −0.0015 |
| `reduced-rank` | the badly-predicted directions carry **0.00%** of the flux variance |
| `geometry-loss-tilt` | already done, by 1434× |
| `bias-correction` | there is no bias: bias/sd between −0.09 and +0.08 on all seven scalars |
| `fitted-ensemble-weights` | optimised weights cross-validate to **−0.00002** — worse than equal |
| `more-shots` | there are no shots left: 7041 in the config, 7041 in the split |
| `capacity`, `depth`, `batch-size`, `regularisers` | every arm negative across three grids |
| `stream-005ms` | 99.0–99.9% of coil-current variance sits below 50 Hz |
| `transductive-selection` | the proxy is **inverted** — disagreement rises where the model is right |
| `retrieval` | 8-NN reaches R² 0.87–0.96 against the ensemble's 0.96–0.99 |
| `ampere-constraint` | per-frame Spearman **−0.175** |
| `eight-seeds` | priced and declined |
| `shrink-to-mean`, `temporal-filter`, `functional-weights`, `axis-coords`, `xpoint-decomp`, `ip-band-loss`, `dimensionless-currents`, `coil-shear`, `beta-p-thomson` | all pre-priced below resolution or refuted in `ideas.md`'s kill list |

## Convention

**The transfer rule, sharpened 2026-08-24 — PRE-REGISTERED, before `seq-ema` closed.** The rule so
far was: a change measures zero in production when the ENSEMBLE already does its work. `ema` and
`huber` died that way; `seq-cosine` transferred because `seq` is a single member with nothing to
dilute it.

`seq-ema` is now fading on the same member that cosine paid on — +0.8, +0.1, +0.0 over three salts,
t = 1.1 — and the averaged weights are demonstrably active, winning 99 and 123 evaluations a run. So
the mechanism fires and the gain does not arrive. If that holds, the rule needs widening:

> A change measures zero when its work is already being done by ANYTHING already in the
> configuration — the ensemble, or another change measured earlier.

Cosine and EMA both stabilise where the trajectory ends. Cosine was measured against a base without
it; EMA is being measured on top of it. Paying twice for one job buys nothing.

**The operational consequence, which is the point of writing this down.** Every arm from here is
measured on top of the accepted stack, so each new result is an INCREMENT, not a standalone value —
and increments do not add up to the sum of the standalone screens. When estimating whether a set of
arms closes the 0.0006 gap, price them against each other, not independently.

**If instead `seq-ema` comes back positive at six pairs, this paragraph is wrong** and the narrow
rule stands. Recorded now so that either outcome costs the same to admit.


**Stop an arm when it is decided, not when the salt list ends.** The protocol says 12 pairs because
that is what resolves +0.00024 from zero. It does not say to keep running after the answer is in.
Twice now a settled arm has blocked an unanswered one: `frame_ends` ran to 8 pairs at +0.00002 while
a submission build waited, and `seq-cosine` ran to 8 pairs at **t = 5.1** while the `seq-ema` lane
sat idle for ten hours. Both were stopped early once noticed, and both should have been stopped
sooner. The queue's job is to spend machine time on questions whose answers are unknown; a t of 5
is not a question.


- **Pre-register the gate before the run.** Mean and sign count, both fixed in advance.
- **Read every arm PAIRED inside a salt.** Absolute S across splits has sd 0.0013–0.0032 at 70
  shots — larger than most effects here. Measured 2026-08-21, not assumed.
- **A screen result is not a production result** until the mechanism says it should survive
  ensembling, and preferably not even then.
- **A decomposition must sum to the total it decomposes.** That check caught the `D_LCFS` sign error.
