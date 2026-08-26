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

### 1c. `cons-reweight` — the metric splits its Consistency weight SEVEN WAYS EQUALLY. Found 2026-08-26
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

**Pre-registered gate:** production pairs, and the named scalars must move — same rule as `aux`.
Refuted if kappa and li gain less than +0.005 of R2 each, or if the other five lose more than the
two gain.

### 1d. `aux-surrogate-frozen` and `1e. aux-surrogate-trained` — RESTORED 2026-08-26 at the user's
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

### 2d. `seq-capacity-dropout` — PRE-REGISTERED 2026-08-23, before `h512` returned
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

### 4. `seq-ema` + `seq-huber` — **CODE WRITTEN AND PROVEN WIRED 2026-08-21, ready to run**
`seq` had neither. Both are now plumbed, Huber scaled by 2 so an arm differs from its control in
outlier handling alone rather than also in how often `grad_clip` fires. The EMA is proven, not
assumed: at decay 0.9 the averaged weights won 4 evaluations, at 0.999 they won 0 and the score was
byte-identical to base — which is why the model now REPORTS its win count, since zero wins and
never-wired-in are otherwise the same number. The mechanism won 965–3019
evaluations per run on the MLP, so it is not architecture-specific, and `seq` is the only ensemble
member not getting it. Caveat that must be respected: `ema` measured ZERO in production for the MLP,
so this is a screen-only expectation until confirmed in production.

### 5. `seq-sweep` — the rest of programme Q
`seq` is a full production member never swept, and it earns its place — `mlp+seq` = 0.9943 against
0.9940 for `mlp` plus three extra seeds, so one extra architecture beats three extra seeds. Arms
left: `learning_rate` (3e-4 / 3e-3), `seq_hidden` 512, `seq_layers` 2, `patience_steps`. Dropout is
already refuted (below). ~72 min a run, ~19 h a 16-salt arm.

### 6. `ema-patience` — the patience was tuned for a trajectory that no longer happens
`ema` reaches its best at 65669 steps against the baseline's 76884 — **15% earlier** — and
`patience_steps` was never revisited. One line.

### 7. `mlp2-block0` — one seed, one block, an order more damage than typical
Found 2026-08-21 in passing: in block 0 of the clean tail, `mlp2` scores **0.977925** where the
other three seeds score 0.9825–0.9881 and every other block has all four at 0.990+. Free to chase —
find the shot, then check whether it is the same one the tail-expert diagnostics keep landing on.

### 1. `aux` — teach `kappa` and `li` directly. PROMOTED FROM 8 ON TEST-SET EVIDENCE, 2026-08-25
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
