# Ideas, ranked

Things worth trying, written down before anyone has measured them. An idea leaves this file in one
of two directions: it gets measured and moves to [experiments_history.md](experiments_history.md)
with a number attached, or it gets argued away and moves there as refuted. Nothing here is a
claim — the whole point of a separate file is that these have **not** been tested.

Ranked best-first by expected gain in S divided by cost and risk. **Group A** is anything one or a
few production runs can settle (~9 min each, under an hour per idea). **Group B** multiplies the
compute — a much bigger model, repeated retraining, or per-frame iterative optimisation — and gets
tested separately, not in the ordinary loop.

Rewritten 2026-08-13 against the production budget below; the dates on individual entries say when
each was first written down.

## The arithmetic every rank is read against

Production headroom is **0.0205** of S: Consistency 0.0158, R²qb 0.0027, D_LCFS 0.0017, R²ψ 0.0004.
Each Consistency scalar is worth 0.20/7 ≈ 0.0286 of S, so the per-scalar budgets are li 0.0031,
kappa 0.0031, volume 0.0029, R_axis 0.0026, Z_axis 0.0023, and the two triangularities 0.0006 each.

**Anything aimed at R²ψ or at the triangularities is capped below the noise floor** and is not on
this list.

Seed noise at production is σ = **0.0009** of S (3 salts × 3 seeds, measured), against the 0.0060
that was carried over from quality scale. A paired comparison resolves **0.0013** on one run and
**0.0008** on three seeds — which is what makes most of Group A worth running at all. Almost all of
that noise sits in Consistency, and inside it in `R_axis` (0.018) and `Z_axis` (0.023), so the
threshold is per-scalar, not global. Full tables in experiments_history.

Two facts about the scorer that several entries lean on:

- **Failures are expensive and quantized.** A frame whose predicted map yields no LCFS contributes
  D = 1.0 to that shot's mean (`local_score.py:363`) against ~0.01–0.02 for a good frame, and a
  non-finite scalar is replaced by the fold mean (`local_score.py:374`), which is R² 0 for that
  frame. Both rates are printed by every run — and at production both are 0.0%, which is why the
  "hunt the failing frames" idea is already in experiments_history as refuted rather than here.
- **The seven scalars are `f_j(ψ_pred)`, not regressions.** The only lever on them is the map — and
  the map is scored frame by frame independently, while every functional it feeds is smooth on the
  current-diffusion timescale. That asymmetry is what the first entry exploits.

MAST / Challenge 2 is deliberately absent: it moves `G_ratio`, not S, and this list ranks by ΔS.

---

## Where to start on a bigger machine — 2026-08-14

Ranked by what twenty measured iterations actually showed, not by what looked promising before
them. Every physics idea below survived only because it is untested, not because it is likely: of
the twenty iterations, the four that paid were capacity, an independent measurement, averaging and
more data, and every physics construction was refuted.

1. **Train the ensemble members in parallel.** Built and measured on 2026-08-14 — 13.1x on the
   fit, score unchanged at quality — and then **removed**: it needed the architecture written down
   a second time, in a form that would rot the first time anyone added a layer. The gain is real
   and the entry in experiments_history.md has the numbers; rebuilding it means doing it through
   `torch.func.stack_module_state` + `vmap`, which stacks an ordinary `nn.Sequential` and carries
   buffers and per-seed randomness, so batch norm and dropout survive it. Read that entry first.
   What the exercise did establish, and what outlived it: the GPU epoch is **launch** overhead, not
   arithmetic — which is what makes A14 below the bigger lever.
2. **More shots, past 0.60.** +0.0018 at the last step and no sign of flattening; memory was
   always the obstacle and the cache removed most of it.
3. **More ensemble members**, once they train in parallel. Four seeds were worth +0.0042 and the
   curve of that has never been probed at all.
4. **Frames per shot.** Production trains on every tenth frame. That ratio was set at a fixed row
   budget in a much smaller regime and never revisited.
5. **MAST.** `predict_row` still raises `NotImplementedError`, so Challenge 2 scores zero — the
   largest structural gap left, and worth more than any refinement on DIII-D.

# Group A — one to a few production runs

**A principle several of these share.** The model sees one frame at a time, 21 instantaneous
numbers, and a fully-connected net has no way to differentiate them — it cannot see the previous
frame or the neighbouring radius. So wherever the physics acts through a *rate* rather than a
level, that quantity is simply not in the inputs, however well its level is represented: the loop
voltage `dI_sol/dt`, the ramp phase `dIp/dt`, the eddy currents driven by `dI_coil/dt`, the
Grad-Shafranov source term `p'(ψ)`. Each of those is a feature we can compute and do not.

Two traps common to all of them. **Differentiate on the signal's own grid, before any resampling** —
a derivative taken after interpolation onto `efit_times` measures the interpolator. And **divide by
the actual step**: the EFIT step is nominally 20 ms, and measured it is 20 ms for 98.0% of
intervals — but 31% of shots carry a gap over 100 ms and the largest is 900 ms. A difference passed
off as a derivative would be wrong by up to 45x on those frames.

## A1. Average several MLP seeds — *2026-08-13*

**Hypothesis.** Averaging four MLPs (seeds 0–3) beats the single-seed mean by ≥ 0.0015.

**Mechanism.** σ ≈ 0.0013 of S across seeds is pure optimisation noise, and nearly all of it is in
Consistency. Averaging coefficients is averaging flux maps, which removes the independent component
of per-frame error in exactly the directions the nonlinear functionals punish. This is variance the
3168-shot model demonstrably does **not** remove for itself — σ was measured on that model.

**Test.** Four `mlp` blocks in `params.yaml` differing only in `seed`, all listed in
`ensemble.members`. One run; the MLP fit is about half of the nine minutes, so ~25 min. Weight
averaging (SWA/EMA) inside a single fit is a nearly free second arm in `TorchMLPModel.fit`.

**Refuted if** the four-seed ensemble is within 0.001 of the single-seed mean — that would say the
per-frame error is shared bias rather than seed noise.

**ΔS ≈ +0.0015…+0.003, confidence 0.8.** The most certain item; ranked second only because its
ceiling is the seed σ itself.

## A2. Flip `calibrate_scalars` on — *2026-08-13*

**Hypothesis.** Dividing each scalar's block in `M` by its measured linear/actual ratio gains
≥ 0.0015, mostly on `li` and `kappa`.

**Mechanism.** `M ∝ JᵀJ / var(f)`, so an over-stated Jacobian is an over-weighted scalar. The
linearisation check measured `li` at 1.65× and `tri_bot` at 1.51×; `li` therefore holds about
1.65² ≈ 2.7× the loss budget it deserves, starving `kappa`, `volume` and the axis — and `li` is
both the worst scalar and the one the Jacobian metric made worse.

**Test.** Already wired: one knob in `params.yaml`, two or three paired production runs.

**Open problem, and the reason confidence is low.** The random full-vector probes give ratios
1.08–1.28 for all seven, nearly uniform, where the model's real error direction gives li 1.65,
tri_bot 1.51, R_axis 0.98. So the ratio is a property of the error *direction*, not of the
functionals, and the switch as built is close to a uniform rescale. Either measure the direction
from a trained model and feed it in (two-pass), or accept that this entry is really about the
former.

**Refuted if** paired ΔS ≤ 0, or `li` improves while `kappa`/`volume`/axis lose as much.

**ΔS ≈ +0.001…+0.0025, confidence 0.4.**

## A3. The MLP's capacity and schedule — *2026-08-13*

**Hypothesis.** At ~63k training frames, 21 → 256 → 256 → 52 with constant lr 1e-3 and no weight
decay is under-fitted, and one of {512×512, 256×256×256, cosine or step decay, wd 1e-5…1e-4} beats
the baseline by ≥ 0.002.

**Mechanism.** The nearest-neighbour ceiling says instantaneous currents determine the targets to
R² 0.97–0.997 while the model reaches 0.89–0.93 — the **regression** is the binding constraint, not
the features. This surface has never been swept once since the starter kit. The "the big model does
it for itself" trap does not apply here, because this *is* enlarging the model. Constant-lr Adam
also leaves the final iterate noisy, which feeds Consistency jitter directly.

**Test.** `hidden_sizes` plus a new schedule key in `params.yaml`; five or six production runs,
about an hour.

**Refuted if** no variant clears +0.002 paired — which would say 63k rows saturate this function
class, and only the Group B capacity ideas remain.

**ΔS ≈ +0.002…+0.005, confidence 0.6.**

## A4. More shots, and absorb the validation window — *2026-08-13*

**Hypothesis.** Training on ~4200–4900 shots, then refitting on train + validation with the epoch
count frozen from the stopped run, gains ≥ 0.0015.

**Mechanism.** 1408 → 3168 shots bought +0.0040 with every term up, and nothing says the curve has
flattened. The validation window is 15% of the shots that the fit reads but never trains on; a
frozen-epoch refit converts them into training data at no methodological cost.

**Test.** Fit the PCA on a subsample of frames — it does not need 140k rows to estimate 50
components, and that is what drove the machine into swap at `0.45/0.2` — then one run at
`0.60/0.1 0.05/0.1 0.01`, and one refit with `patience: 0` and `epochs` set from the stopped run.

**Refuted if** the paired gain is ≤ +0.0005.

**ΔS ≈ +0.0015…+0.003, confidence 0.7.**

## A5. CatBoost at production scale, back in the ensemble — *2026-08-13*

**Hypothesis.** CatBoost trained at production scale, averaged with the MLP, beats the MLP alone by
≥ 0.0015.

**Mechanism.** On the old 14-shot fold the ensemble scored 0.9319 against 0.9304 for CatBoost alone
and 0.9030 for the MLP (README, "scaling"), and CatBoost has simply never been *run* at production
scale. Axis-aligned splits against smooth ReLU surfaces is real family diversity, which is what
averaging is paid for.

**Adversarial note.** Both of this project's reversals happened because 3168 shots let the MLP
catch up with something we were supplying by hand. The gap may have closed — that is exactly what
the run measures.

**Test.** `enabled: true`, `iterations` raised to ~8000, weights mlp 0.6 / catboost 0.4; one run,
40–60 min, the CatBoost fit dominating.

**Refuted if** CatBoost alone lands ≥ 0.004 below the MLP and the ensemble is within 0.001 of it.

**ΔS ≈ +0.001…+0.003, confidence 0.4.** Borderline Group A on cost.

## A6. A dedicated q95/βN head — *2026-08-13*

**Hypothesis.** A separate small model for the two submitted scalars recovers ≥ 0.001 of the 0.0027
R²qb budget.

**Mechanism.** The scorer treats q95 and βN as independent channels — `common.py:6` calls them "the
ONLY two separately-predicted scalars ... not recoverable from psi(R,Z)". The Jacobian metric
measurably taxed them (R²qb paid 0.0028 when the ψ block's share of the loss grew), because the
shared trunk trades them against the map by construction. Their ceiling from currents alone is
0.997 against 0.981 reached.

**Test.** One extra model in the zoo predicting the 2-vector; `predict_row` takes ψ from the
ensemble and the two scalars from it. One production run.

**Refuted if** the dedicated R²qb is within 0.005 of the shared one.

**ΔS ≈ +0.001…+0.002, confidence 0.5.**

## A7. Post-hoc de-biasing of the derived scalars — *2026-08-13*

**Hypothesis.** The scalars of the predicted maps carry a systematic component, and a fixed
correction — `δc = J⁺ · (mean bias)`, or a small affine map fitted on the validation window —
recovers ≥ 0.001.

**Mechanism.** A pooled R² pays for bias in full, and the linearisation check found bias shares up
to 25% of the miss (R_axis). Predicting a conditional-mean map is not the same as predicting the
map whose functionals are correct: `f(E[ψ]) ≠ E[f(ψ)]` for an argmax or a contour extreme. `J` and
its pseudo-inverse already exist in `jacobian_form`.

**Adversarial note.** The model already minimises the same quadratic loss on the training set, so
it does part of this for itself. Hence the modest estimate.

**Test.** Diagnostic first and free: mean and slope of `f_j(ψ_pred)` against `f_j(ψ_gt)` on
validation frames. If biased, apply `δc` in `predict_row` and re-score the existing artifact.

**Refuted if** the biases are under 5% of each scalar's RMS miss, or the correction helps the
validation window and not the scored tail.

**ΔS ≈ +0.001…+0.002, confidence 0.35.**

## A8. More frames per shot at fixed shots — *2026-08-13*

**Hypothesis.** Production trains on every tenth frame (`0.45/0.1`); raising the frame share to 0.2
or 0.3 at unchanged shots gains ≥ 0.001.

**Mechanism.** The 08-12 result — more shots and fewer frames beats the reverse at a fixed row
budget — set the ratio, but it never said the frame axis is worthless, only that it is the cheaper
one to spend. Between-shot and within-shot variance split roughly 53/47, so frames are
near-duplicates but not duplicates. This is the one axis of the split we have never varied on its
own at production.

**Test.** One run each at `0.45/0.2` and `0.45/0.3`, paired against `0.45/0.1`. Needs A4's
subsampled PCA first — `0.45/0.2` is exactly what drove the machine into swap.

**Refuted if** the paired gain is ≤ +0.0005, in which case frames are settled and only shots matter.

**ΔS ≈ +0.0005…+0.002, confidence 0.4.**

## A9. Auxiliary supervision on the scored functionals — *2026-08-13*

**Hypothesis.** Adding auxiliary target columns — the shipped `efit_li` / `efit_r_axis` /
`efit_z_axis` labels, or better, `f_j(ψ_gt)` precomputed with the scorer's own `derive_frame` over
the training frames — shapes the trunk so that the *derived* scalars of the predicted map improve
by ≥ 0.001.

**Mechanism.** The Jacobian metric encodes only the linearised response of the functionals;
auxiliary heads supply the nonlinear version as gradient signal to the shared layers.

**Adversarial note.** This is precisely the category that burned us twice — the Jacobian loss
already hands the model most of this information, so the marginal effect is a good candidate for
vanishing at production.

**Test.** Widen the target vector by three to seven columns, weighted like the qb block, dropped at
inference. One production run.

**Refuted if** paired ΔS ≤ 0.001 — the expected outcome, but the test is a single run.

**ΔS ≈ 0…+0.002, confidence 0.3.**

## A10. Derivatives, wherever the physics is a rate — *2026-08-13*

**Hypothesis.** The model is handed levels of quantities whose *rates* are what act physically, and
supplying the rates explicitly gains ≥ 0.001.

**Mechanism.** Three places where the rate is the physical quantity and the level is not:

- **The solenoid.** `dI_sol/dt` is the loop voltage — the thing that actually drives the plasma
  current, where `I_sol` itself drives nothing. Measured: it correlates −0.665 with `dIp/dt`
  against +0.515 for the level, so the derivative carries the relationship and the level dilutes it.
- **The plasma current.** `dIp/dt` separates ramp-up, flat-top and ramp-down, which are physically
  different regimes with different current profiles — and `li` is a current-profile quantity.
- **The vessel.** High-passing each coil current at the vessel's L/R timescale (~2–20 ms) is a
  proxy for the eddy currents it induces. Those produce real poloidal flux and are **structurally
  absent** from the shipped signals: `coil_field` cannot compute them, because the vessel geometry
  is not in the dataset.

This is the one feature family aimed at information the nearest-neighbour ceiling cannot contain.
That ceiling was measured on instantaneous currents, so it bounds only what instantaneous currents
determine — and the missing 0.03 on `li` could be exactly what they do not.

**Test.** Extend `features_for_row` with the derivatives and two or three high-passed copies per
signal, computed on the raw magnetics time base and filtered before interpolation onto
`efit_times`. Identical at training and inference. One run per time constant.

**Refuted if** ΔS ≤ 0.001 paired, or the gain appears only on scalars already at their ceiling.

**ΔS ≈ 0…+0.002, confidence 0.35.** In Group A because widening the first layer by a few columns
is not a measurable change in training time.

## A11. Thomson scattering, and its radial derivative — *2026-08-13*

**Hypothesis.** Reducing the Thomson profiles to a few numbers per frame — including their radial
gradients — gains ≥ 0.001, most of it on βN and `li`.

**Mechanism.** Pressure is one of the two free functions of Grad-Shafranov, and it is the one
physical quantity the magnetics genuinely cannot see: coil currents constrain the flux at the
boundary, not how the pressure is distributed inside.

**And it is the radial DERIVATIVE that enters, not the profile.** Grad-Shafranov reads
`Δ*ψ = −μ₀R² p'(ψ) − FF'(ψ)`: the source term is `p'`, so `dTe/dr` and `dne/dr` are the physically
right quantities and the profiles themselves are their integrals. Two features follow that a raw
profile does not give:

- `d(n_e T_e)/dr`, the pressure gradient, at a few normalised radii;
- the **pedestal position**, the radius of maximum `|dTe/dr|`, which locates the edge transport
  barrier — and the edge is where the boundary-shape scalars are decided.

**Adversarial note.** The nearest-neighbour ceiling from currents alone is 0.997 on βN against
0.981 reached, so the currents nearly determine it already. And an earlier check found the per-shot
R² correlating −0.07 with Te and 0.00 with ne over 200 shots: the shots we fail on are not the ones
with unusual profiles. Both point the same way.

**Test.** A handful of robust numbers per frame — core value, edge value, width, peaking factor,
plus the gradient features above — never the raw profile, which is a different length per shot and
mostly noise at the edge. Differentiate on the profile's own radial grid, before any resampling.

**Refuted if** ΔS ≤ 0.001 paired. Run the gradient arm even if the level arm fails: they are
different hypotheses, and only the gradient one follows from the equation.

**ΔS ≈ 0…+0.0015, confidence 0.25.** In Group A because it costs no training time, not because it
looks likely.

## A12. Reversed-Ip augmentation — *2026-08-13*

**Hypothesis.** Reversed-current shots (~10% of the data) still score materially worse *for the
MLP*, and mirroring them as augmentation (X → −X with mirrored targets) closes part of the gap.

**Mechanism.** Two polarities are two regimes and the model sees ten times less data for one of
them. This is **not** the polarity *normalization* refuted on 08-11: that one collapsed the two
regimes into one at tiny scale on the whole zoo; augmentation adds data instead of removing a
degree of freedom, and the refutation predates the big model.

**Test.** Step 0 is free: group per-shot R²ψ and D_LCFS from the last production run by sign(Ip).
Only if reversed shots are ≥ 0.01 worse is the augmentation run worth it — and it needs care that
the q95 sign convention survives mirroring, which the labels can settle first.

**Refuted if** the per-polarity gap for the MLP is under 0.005 of per-shot S. Likely.

**ΔS ≈ 0…+0.001, confidence 0.2.** Last in Group A on purpose.

## A14. Raise `batch_size`, and refit the schedule with it — *2026-08-14*

**Hypothesis.** `batch_size: 512` was never chosen; it is the value the fit has always had. A
larger batch is several times cheaper per epoch on a GPU, and the score is at worst unchanged.

**Mechanism, measured.** The training loop is bound by kernel launches, so epoch time tracks the
NUMBER of batches and barely notices their size. On production shapes, 93736 rows on a V100:

| batch | batches/epoch | s/epoch |
|---|---|---|
| 512 | 183 | 0.307 |
| 1024 | 91 | 0.152 |
| 2048 | 45 | 0.080 |
| 4096 | 22 | 0.033 |
| 8192 | 11 | 0.020 |

Halving the batch count halves the time, almost exactly, up to about 4096.

**Why this is not a free speed-up, and belongs here rather than in a speed change.** It changes the
optimisation, not the arithmetic. At 8192 an epoch is 11 Adam steps instead of 183, so:

- `patience: 100` becomes an eighteenth of the leash it is now, in gradient steps. The measurement
  that fixed patience at 100 (experiments_history, 08-12) was made at 512 and does not carry over.
- The learning rate has to move with it. The usual linear or square-root scaling is a starting
  point, not an answer.
- Gradient noise falls with the batch, and for a 75k-parameter net on ~90k rows that noise is
  plausibly doing regularisation work.

**Test.** A grid over `batch_size` x `learning_rate` at production shares with `--only mlp` — which
is exactly what `make quality` now is, at a fifth of a production run's cost. Patience must be
converted to steps rather than left at 100 epochs, or the comparison measures the leash and not the
batch.

**Refuted if** the best cell at any batch above 512 does not beat 512's own best cell by more than
the single-net sigma of 0.0013.

**ΔS ≈ −0.002…+0.002, confidence 0.3.** It is on this list for the wall clock, not for the score:
even at ΔS = 0 exactly, a 4x cheaper production fit makes everything else on this list cheaper to
test, and that is worth a grid.

## A17. Settle the frames axis past 0.2 — *2026-08-14* — **DONE, and it reversed**

**Done 2026-08-14, on eight salts.** +0.0018 on average, every sign positive, t = +5.02. `0.80/1.0`
is the production share now.

The interesting part is why it had read as flat: grid G ran the 1.0 arm with
`pca_frame_share: 0.1` against the 0.2 arm's `1.0`, so the treatment arm's PCA was estimated from
half the sample. That was a memory measure, taken without noticing it was also an experimental
change — the confound was in the CONTROL and it cost the effect about two thirds of its size. The
lesson is narrower than "hold everything fixed": it is that a setting adjusted to make a run FIT
is still a setting, and belongs in the same audit as the one being tested.

### The original entry

**Not a hypothesis, a resolution problem.** `0.80/1.0` against `0.80/0.2` measured +0.0005 /
−0.0002 / +0.0025 on salts 0 / 3 / 4. Mean +0.0009 with disagreeing signs, which is 1.2 sigma on
the standard error of three runs — too small for three salts to call, in either direction.

**The cost of settling it.** Resolving +0.0009 at two sigma needs the standard error down to
0.00045, so n = (0.0013 / 0.00045)² ≈ **8 salts**. Salts 0, 3 and 4 already have both arms, so it
is five more salts x two arms = **ten runs, about 75 minutes** — the grid is written and ready at
`grids/h_frames_8salts.json`. Both arms per salt, because the comparison is paired within a salt
and absolute scores are not comparable across them.

**Why it is not ranked higher.** Even if the full +0.0009 is real it costs **2.2x the wall clock**
and 2x the memory for every run thereafter, against +0.0037 for the step to 0.2 at one extra
minute. So the honest position is that 0.2 is where the knee is for practical purposes, and this
entry exists so that "we never checked" does not get mistaken for "we checked and it was zero".

**Cheaper variant worth doing first:** 0.4 and 0.6 on three salts, which would say whether the
curve rises smoothly or the 1.0 arm's spread is noise. Six runs, ~40 min.

## A16. Count the schedule and the patience in STEPS, not epochs — *2026-08-14* — **DONE**

**Done 2026-08-14.** Built as designed below, with `lr_t_max_steps` left at 0 (meaning `max_steps`)
rather than pinned, since no schedule is on by default. Accepted on the reproduction criterion:
salt 0 gave 0.9906 against a recorded 0.9905 and salt 3 gave 0.9886 against 0.9890 — +0.0001 and
−0.0004, both a third of a sigma. The shares→patience table is deleted from params.yaml.

The tail measurement below is now the open half of this entry: **60–70% of the fit runs after the
best step is already found**, and cutting the leash is worth a 2x screen. It is a measurement, not
an edit — see the warning in the last paragraph — and it belongs in the ten experiments.

**Not a hypothesis about the score.** This one removes a class of bug that has already fired, and
it deletes a lookup table that AGENTS.md forbids. Rank it as infrastructure, not as ΔS.

**The problem.** `patience`, `epochs` and `lr_t_max` are all counted in EPOCHS, and an epoch is
`rows / batch_size` optimizer steps — so every one of them silently changes meaning when the batch
or the data changes. On 2026-08-14 that forced three manual rescalings:

| change | rescaling it forced | factor |
|---|---|---|
| batch 512 → 4096 | patience 100 → 800 | 8x |
| data 0.60/0.1 → 0.80/0.2 | patience 800 → 302 | 2.7x |
| data → 0.80/1.0 | patience 800 → 60 | 13x |

and **one of them was got wrong**: `lr_t_max` was driven by the `epochs` CEILING, which sits far
above any real fit, so a 533-epoch cosine run traversed only the flat first quarter of its curve.
The rate fell from 1e-3 to 8.4e-4 and never went below 84% of its start. It was recorded as an
annealing experiment for an hour before the arithmetic was checked. In steps that mistake cannot be
made, because the horizon would not be expressed in a unit that depends on the dataset.

Worse, params.yaml now carries this, which is exactly the second-value-to-keep-in-sync the
conventions ban:

    #   0.60/0.1 -> patience 836, epochs 16727
    #   0.80/0.2 -> patience 302, epochs 6033
    #   0.80/1.0 -> patience  60, epochs 1207

Moving to steps deletes the table rather than documenting it better.

**It is also what everyone else does.** `transformers` schedules take `num_training_steps` and
`warmup_steps`; Lightning's `val_check_interval` is in steps; large-scale training plans in steps
and tokens. Epochs survive as a unit mainly in vision benchmarks, where the dataset is fixed and
the question never comes up.

**The design.** Four settings, none of which depends on the data:

| now | after |
|---|---|
| `epochs: 6033` | `max_steps: 368000` |
| `patience: 302` | `patience_steps: 18400` |
| `lr_t_max: 0` | `lr_t_max_steps: 18400` |
| — | `eval_every_steps` |

The last one is new and necessary: validation is currently computed once per epoch, so early
stopping lives on that grid whatever unit the patience is in. Making the cadence explicit is a
small win of its own — at batch 4096 the validation set is currently evaluated every 23 steps,
which is far more often than the curve needs.

**Acceptance is a reproduction, not an improvement.** A configuration expressed in steps must
reproduce the score of the same configuration expressed in epochs, on one salt, to four decimals or
within the 0.0013 sigma. Without that check there is no way to tell a translation error from a real
effect.

**What it unlocks.** With the units fixed, the patience can finally be tuned once and stay correct.
It needs tuning: measured on 2026-08-14, at batch 4096 the best epoch arrives after 7-13 thousand
steps while the leash is 18400, so **60-70% of the fit runs after the best point is already
found**. That leash was calibrated at batch 512, where the optimum took 50-115 thousand steps and
it was a 20-30% tail. Cutting it is plausibly a 2x speed-up — and it is exactly the change that
cost 0.5521 against 0.7273 when it was done carelessly on 08-12, so it is a measurement, not an
edit.

## A18. The ten, pre-registered — *2026-08-14*

Ten arms declared BEFORE any of them runs, five on the architecture and five on the features and
the loss. Written here first because thirty screened configurations have already taught this fork
what happens otherwise: the best of grids A+B+C was +0.0013 against the +0.0029 selection alone
produces over that many arms, and the two that were carried to confirmation both went NEGATIVE on
unseen salts.

**The arithmetic, once, for all ten.** Sigma is 0.0013 for a single net. Over ten arms selection
alone is expected to produce a best of 0.0013 * sqrt(2 ln 10) = **+0.0028**. So an arm that beats
the control by less than that on the screening salt has shown nothing, whatever its rank in the
table. Only an arm above it earns the two confirmation runs on salts nothing selected against, and
only agreeing signs there earn a place in the submission.

They are ordered by expected value over cost, and the tail is the part that gets dropped if the
clock runs out — which will be recorded as dropped, not quietly omitted.

### Architecture

| # | arm | why it might pay | costs |
|---|---|---|---|
| 1 | **Shorter leash**: `patience_steps` 18400 → 6000 | measured on 08-14: the best step arrives at 7-13k while the leash is 18400, so **60-70% of every fit runs after the best point is already found**. The leash was calibrated at batch 512, where the same tail was 20-30% | config only |
| 2 | **LayerNorm** on the hidden pre-activations | batch norm was refuted at four learning rates, but its mechanism is batch COUPLING — LN normalises within a sample and has none of it. The refutation of one is not evidence about the other | a `norm` knob in `build_mlp` |
| 3 | **Residual trunk**, 4-6 blocks with skip connections | the model is bias-limited, not variance-limited (three regularisers all closed the train/val gap while validation stayed flat), so it wants effective depth — and plain depth 3 already went slightly negative, which is what untrainable depth looks like | a `residual` knob |
| 4 | **Pyramid** [1024, 256] against [512, 512] | equal parameters, different allocation. If the mapping is "lift 122 inputs into a wide space, then compress", the first layer is where the capacity belongs | config only |
| 5 | **Split heads** for the PCA block and the two scalars | one linear layer produces all 52 outputs today, so the flux coefficients and q95/betaN share a final representation although the metric weights them 71/29 and they have different statistics | a small head change |

### Features and loss

| # | arm | why it might pay | costs |
|---|---|---|---|
| 6 | **`derivatives: both`, `derivative_signals: poloidal`** | A10 screened this at 0.60/0.1 and got +0.0007 at best, below the selection floor. The data is 5-10x bigger now, and a feature that needs data to pay is exactly the kind that a small screen refutes wrongly | config only |
| 7 | **`n_pca: 75`** | saturated at the OLD data scale (75 gave −0.0010 at production). More components need more frames to estimate and to fit; there are five times as many now | config only |
| 8 | **Thomson profile groups** added to the four in use | the profile block is computed and cached and simply not selected. The cheapest untried feature in the repo | config only |
| 9 | **Huber loss** on the scaled targets | the loss is MSE over 52 scaled outputs, so one badly reconstructed frame moves the gradient as much as thirty ordinary ones. Huber at 2-3 sigma keeps the quadratic centre and bounds the tail | a loss selector |
| 10 | **Per-frame weighting by the Jacobian** (B6) | `jacobian_form` already measures how the seven scored functionals respond to each coefficient, and the loss charges every frame equally although equal coefficient error costs unequal scalar error across frames | the largest of the ten |

## A15. Capacity and light dropout TOGETHER — *2026-08-14* — **DONE, refuted**

**Done 2026-08-15**, twelve cells at `0.80/1.0`. Every axis flat or negative on the marginal means:
width −0.0006 / −0.0011, depth −0.0001 / −0.0011, dropout 0→0.02 nothing and 0.02→0.1 −0.0025.
The numbers are in experiments_history.

**What it changes about everything below it.** The bias-limited diagnosis was right and its usual
corollary was wrong: this model is not short of parameters. It can already express more than 46
numbers from one frame determine, so the ceiling is in the INPUTS, not in the hypothesis class.
Every remaining "make it bigger" idea drops in rank and everything that adds information —
the sequence model, the feature arms of A18 — rises.

### The original entry


**Queued after the nonlinearity and depth arms of grid C.**

**Hypothesis.** A much wider net (1024 or 2048 per layer, possibly deeper) with dropout 0.05-0.2
beats both of its halves, because the dropout makes the extra capacity usable and doubles as an
implicit ensemble over sub-networks.

**Why this is not already refuted, when both halves are.** The two experiments that look like they
settle it tested opposite corners and never the same cell:

| | dropout 0 | dropout 0.05-0.2 |
|---|---|---|
| 512x512 | the baseline, 0.9873 | **grid C: −0.0015 to −0.0050, monotone in strength** |
| 1024x1024 | **it-11: 0.9866 against 0.9868, refuted** | **never run** |

it-11 grew the width at ZERO regularisation and found nothing; grid C regularised at the BASE width
and made things worse. Neither says what happens when the capacity has something holding it
together. The 2x2 has one empty cell and it is the interesting one.

**Mechanism, and why it is worth a run despite the diagnosis.** Grid C's three-way result —
batch norm, dropout and weight decay all close the train/val gap and all fail to move validation or
actively hurt it — says the model is limited by BIAS, not variance. That is an argument FOR
capacity, not against it: bias is what capacity buys down. What the same result says is that
capacity has to arrive without paying the regularisation tax that the base-width runs paid, which
is the whole question here. Dropout at 0.05 is a much smaller tax than at 0.2, and grid C's own
curve is monotone in strength, so the low end is where to look.

**What makes it affordable now.** Two things that were not true this morning: the fit runs on the
GPU, and batch 4096 with a sqrt-scaled rate is score-neutral at **1.89x** (grid B). A wider net
also uses the GPU better than the current one, which is launch-bound rather than arithmetic-bound —
though whether 2048-wide is still launch-bound is a measurement, not an assumption, and it is
step 0.

**Test.** Step 0, free: measure s/epoch at 1024 and 2048 wide, at batch 4096, before committing to
a grid — if a wide net is no longer launch-bound the whole thing costs several times more and the
grid has to shrink. Then width in {1024, 2048} x dropout in {0.0, 0.05, 0.1}, with 0.0 included as
the control that reproduces it-11 at the current recipe. Everything else at whatever grid B and the
nonlinearity arms leave as the best configuration.

**Refuted if** the best cell fails to beat the base width by more than the 0.0013 single-net sigma,
on salt 0 and then on salts nothing selected against.

**ΔS ≈ 0…+0.004, confidence 0.35.** The confidence is not higher because the nearest-neighbour
ceiling measured earlier (R² 0.97-0.997 against the 0.90-0.93 the model reaches) says the inputs
themselves may be the binding constraint, in which case no amount of capacity helps and the answer
is features or A13's loss, not width.

## A13. Stop on the composite, not on the validation MSE — *2026-08-14*

**Hypothesis.** Early stopping picks the epoch with the lowest validation MSE, which is not the
epoch with the highest S, and the difference is worth something.

**Mechanism.** The composite weighs `W_PSI 0.55, W_QB 0.15, W_LCFS 0.10, W_CONS 0.20`
(`fusion_scoring/common.py:57`). With `loss.metric: jacobian` the loss now covers 0.90 of that,
which is far better than the 0.70 it covered under `parseval` — but four gaps remain, and they are
gaps in kind, not in size:

- **D_LCFS, 0.10, is structurally absent.** It is a distance, zero at the ground truth, so a
  central difference sees no derivative at any probe step.
- **Consistency, 0.20, enters as a linearisation** measured at one finite step. `calibrate_scalars`
  in params.yaml records how badly: `li` overstated 1.65x, `tri_bot` 1.51x.
- **`jacobian_delta` is fitted at the error the model makes** — but the model gets better during
  training, so by the last epochs the probe stands where the model no longer is.
- **The aggregations differ.** R²ψ and R²qb are pooled over the whole fold against a single scalar
  mean; D_LCFS is averaged per shot; MSE is a mean over frames. None is a monotone function of
  another, so two models with equal loss can score differently.

**Why now.** This was unaffordable before: evaluating the real composite means decoding PCA to flux
maps and running `derive_frame` per frame. At 0.0158 s per seed-epoch it is now the only expensive
thing left in the loop, so it can be budgeted deliberately — every K epochs, on a fixed subsample.
The machinery exists: `scorer_context` and `derive_frame`, as the Jacobian probe already uses them
at `jacobian_frames: 300`.

**Test.** Step 0 is a measurement, not a change: record for one production fit both curves — the
validation MSE per epoch, and the composite on the validation shots every K epochs — and see how
far apart their peaks are and how much S the MSE-chosen epoch leaves behind. Only if that gap
clears the 0.0009 seed sigma is the switch worth building.

**Refuted if** the two peaks land within a few dozen epochs of each other, which after the move to
`jacobian` is entirely possible — the loss became a much better surrogate than it was when README
first recorded that they disagree.

**ΔS ≈ 0…+0.003, confidence 0.35.** The evidence it rests on is old: the observation that the
validation curve keeps improving past the score's peak was made under `parseval`, when the loss
was blind to Consistency entirely.

---

# Group B — multiplies the compute

Tested on their own, not inside the ordinary iteration loop. Each says roughly how much more it
costs.

## B1. A real hyper-parameter search — *2026-08-13* — ~40 runs, overnight

**Hypothesis.** A 30–60-trial search over width, depth, learning rate and schedule, weight decay,
batch size and patience beats A3's hand sweep by a further ≥ 0.002.

**Mechanism.** Same as A3 — the regression is the binding constraint and this surface is unexplored.
σ = 0.0013 makes ~0.002 effects rankable, and pairing trials within one salt keeps it clean.

**Test.** Optuna or a plain grid driving `train_eval.py` at production shares, one fixed salt, the
best three re-validated on a second. ~6 h unattended on this machine.

**Refuted if** the incumbent is within noise of the search optimum. **ΔS ≈ +0.002…+0.004 beyond A3,
confidence 0.6.**

## B2. Scale the ensemble to 8–10 diverse members — *2026-08-13* — ×10 training, ~2 h

**Hypothesis.** Seeds × widths × CatBoost (if A5 survives) adds a further ≥ 0.001 over A1.

**Mechanism.** Variance reduction saturates as 1/N in the independent component, and architectural
diversity keeps the components independent longer than seeds alone do. Inference cost at submission
— 874 shots × 10 small nets — stays trivial.

**Test.** Grow `params.yaml` after A1, A3 and A5 report; one long run.

**Refuted if** members 5–10 add under 0.0005 over the four-seed ensemble. **ΔS ≈ +0.001…+0.002 on
top of A1, confidence 0.7.**

## B3. All the data — *2026-08-13* — ×3–5 in memory work and time

**Hypothesis.** After A4, going to ~5600 shots with a 0.2–0.4 frame share gains a further ≥ 0.001.

**Mechanism.** Extrapolation of the shots curve, plus whatever A8 finds on the frame axis.

**Test.** float32 targets end to end, subsampled PCA, a chunked metric transform — the RAM ceiling
is the real engineering, roughly half a day — then one run per point. A preprocessed cache of ψ
plus features would be ~18 GB against 88 GB of parquet and would remove the decode cost entirely;
profile the run first to see whether that is where the time actually goes.

**Refuted if** A4's paired gain is already ≤ +0.0005 — that closes the whole axis. **ΔS ≈
+0.001…+0.003, confidence 0.5.**

## B4. Zeroth-order calibration against the exact scorer — *2026-08-13* — ~200 scoring passes, 4–6 h

**Hypothesis.** Optimising a low-dimensional post-hoc correction (per-scalar gains, smoothing
window, an inflation factor — 10 to 20 parameters in all) directly against true S on validation
shots beats the surrogate-tuned version of A7 by ≥ 0.001.

**Mechanism.** The quadratic surrogate is linearised and provably miscalibrated for three of the
seven (correlation 0.45–0.77), and it is blind to the Hausdorff maximum. The true scorer costs
1–2 min per evaluation, which is affordable as an objective when the parameter count is tiny.

**Test.** Wrap `local_score` as an objective over the validation shots; CMA-ES or Nelder-Mead;
confirm the winner on the untouched tail of two salts. Overfitting the validation fold is the
failure mode, hence few parameters and the second-salt confirmation.

**Refuted if** the tail gain is under half the validation gain. **ΔS ≈ +0.001…+0.003,
confidence 0.35.**

## B5. A sequence model over the shot — *2026-08-13* — days of work, 2–5× training cost

**Hypothesis.** A temporal model (TCN or GRU, or the present MLP with a learned causal filter bank
on its inputs) mapping the whole feature sequence to the coefficient sequence beats per-frame
prediction by ≥ 0.002.

**Mechanism.** A sequence model learns state — integrating exactly the eddy-current and
current-diffusion dynamics that instantaneous features miss — rather than filtering the output of a
memoryless one.

**Downgraded 08-13, and honestly.** This entry used to lean on "post-hoc smoothing has a ceiling
that a learned filter would beat". That argument is gone: smoothing was measured and the prediction
turned out to be *twice as smooth as the ground truth*, so there is no jitter for a better filter
to remove. What survives is the other half — a memoryless model cannot represent the vessel's
current state at all, whatever bandwidth it is given — and that is a claim about A10's features as
much as about architecture. Test A10 first: if explicit derivatives and high-passed currents buy
nothing, the missing state is not the binding constraint and this becomes hard to justify at days
of work.

**Test.** A new `TargetModel` consuming per-shot batches, which is a loader rework: the pipeline
concatenates frames today, and the training unit would become the shot (3168 of them, not 63k).

**Gate.** Run A10 first, for the reason above. **ΔS ≈ +0.001…+0.004, confidence 0.3.**

### What the interface already gives, and the one thing it does not — *2026-08-14*

Three facts about the pipeline, checked before designing anything:

- `predict` needs **no change at all**. `predict_row` (`baseline_model.py`) is called with one
  shot's frames, in efit time order — so a sequence model gets its natural unit at inference for
  free.
- The **ensemble is free too**. `_predict_targets` is a weighted sum over the members' `predict` in
  the scaled target space, so a sequence model earns a weight in `params.yaml` beside the MLPs and
  the two can be averaged without either knowing about the other.
- **`fit` is the only thing that has to change.** `_read_shots` concatenates every shot into one
  block and drops the boundaries, and `TargetModel.fit(X, Y, X_val, Y_val)` has nowhere to say
  which rows belong to which shot. It gets an optional per-shot boundary argument; ridge, CatBoost
  and the MLP ignore it.

### The frame clock, measured — *2026-08-14*

202 shots, 44729 steps of `efit_times`:

| step | share |
| ---: | ---: |
| 20 ms | 97.86% |
| 40 ms | 1.59% |
| 60 ms | 0.24% |
| ≥ 80 ms | 0.31% |

Every step is a multiple of 20 ms, **77% of shots contain at least one gap** (median 2.0% of that
shot's steps), the largest is 2580 ms, and a shot is 2 to 373 frames long (median 234). So the
clock is not per-shot: it is one 20 ms clock with dropped frames.

Two consequences for this entry, both settled:

- **Train it at `frame_share = 1.0` only.** Training on `0.80/0.2` takes a stride-5 subsample —
  ~100 ms apart — and inference sees every frame at 20 ms. For a memoryless MLP that is only
  "fewer samples"; for a model over time it is learning the dynamics at one dt and applying them
  at another. The model raises rather than accepting a smaller share, per AGENTS.md.
- **Feed Δt as an input regardless**, because 1.0 still is not a regular grid — the gaps remain.
  In units of the base step, not milliseconds, so a gap reads as "this step was 3 frames long":
  Δt to the previous frame for the forward pass, Δt to the next for the backward one.

Windowing — thinning by whole windows instead of by frames inside them — was the third option and
is worse than it looks: a stride on a clock that already has holes makes the spacing vary from two
mixed causes at once, where `1.0` plus a Δt column leaves exactly one.

### The screen, pre-registered — *2026-08-14*

Six arms on one salt, declared before any of them runs. None of the hyper-parameters here has ever
been measured on this problem — the MLP's were tuned over thirty runs and none of that transfers to
a recurrence — so the first number this model produces is worth nothing as a verdict on the
architecture, and this exists so that it is not read as one.

**The control is the MLP at the same shares and the same salt**, not the MLP at its own best
shares: `81_f10_s0 = 0.9907` at `0.80/1.0`, salt 0.

| arm | what it varies | why |
|---|---|---|
| `q1` | the block as it stands: lr 1e-3, `seq_hidden` 256, one layer, 32 shots | the reference the other five are read against |
| `q2` | lr 3e-4 | recurrences are usually run below the rate a feed-forward net of the same width takes, and the MLP's 2.8e-3 was tuned for a batch of frames, not of shots |
| `q3` | lr 3e-3 | the other side of it — the argument above is a convention, not a measurement |
| `q4` | `seq_hidden` 512 | how much state the shot actually needs is the question the architecture exists to ask |
| `q5` | `seq_layers` 2 | depth in the recurrence rather than width |
| `q6` | `dropout` 0.1 | **the one arm with a real prior.** The training set is 5633 SEQUENCES where the MLP sees 1.25M independent rows, so this model has three orders of magnitude fewer examples per parameter and the overfitting argument that failed for the MLP — measured bias-limited — need not fail here |

**Thresholds.** Six arms, sigma 0.0013, so selection alone clears +0.0023 over the best. But this
is not a selection problem yet: the question is whether the best arm beats the MLP control at all.
Read it as **refuted if no arm reaches the control**, since the residual form means the model can
always fall back to being the MLP and failing to is evidence the recurrence actively hurts.
Anything that beats the control by more than +0.0023 goes to two unseen salts like everything else.

**And a second reading that costs nothing.** The `delta` branch starts at zero, so the size of its
output at the end of training measures how much the model decided to use the shot. If the best arm
matches the control with a `delta` that stayed near zero, the recurrence found nothing; if it
matches with a large `delta`, it found something and paid for it elsewhere. Those are different
results and the score alone cannot tell them apart.

## B6. A per-frame heteroscedastic metric — *2026-08-13* — ~half a day

**Hypothesis.** The fixed `M` (JᵀJ averaged over 300 probe frames) under-weights high-sensitivity
frames; weighting each training frame by a cheap surrogate for `tr JᵀJ` gains ≥ 0.001.

**Mechanism.** A pooled R² weights frames equally, but equal coefficient error costs unequal scalar
error across frames, and the loss should charge the frames where `J` is large. Averaging `M` blurs
exactly that.

**Test.** Fit the surrogate from the probe output we already compute (free), apply as per-row sample
weights in the MLP loss, one run per strength.

**Refuted if** per-frame `tr JᵀJ` varies by less than ~2× across probe frames. **ΔS ≈
+0.001…+0.002, confidence 0.3.**

## B7. Grad-Shafranov cleanup at inference — *2026-08-13* — ~1 day

**Hypothesis.** Projecting each predicted map onto "no toroidal current outside the predicted LCFS"
— `Δ*ψ = −μ₀ R j_φ = 0` there, using the existing `grad_shafranov_form` — improves LCFS extraction
and the contour scalars by ≥ 0.001.

**Mechanism.** Spurious vacuum current wrinkles the flux exactly where `lcfs.py` binary-searches for
the boundary, and it is physics the loss never imposed. The most physical idea left.

**Adversarial note.** R²ψ = 0.9993 says the maps are already very clean, and the Jacobian loss
suppresses the functional-relevant part of the noise. The model may effectively do this itself,
which is why it sits low despite the physics.

**Test.** The fixed-mask subspace version first — one eigendecomposition of the form on the PCA
subspace and a damping sweep, testable inside A7's harness. The per-frame masked version only if
that shows a sign.

**Refuted if** the damping sweep's optimum is zero. **ΔS ≈ 0…+0.002, confidence 0.25.**

## B8. A boundary term that is an honest maximum — *2026-08-13* — ~half a day

**Hypothesis.** Fine-tuning the MLP with an auxiliary log-sum-exp (soft-Hausdorff) penalty over
per-frame boundary displacements `δψ/|∇ψ|` improves `1 − D_LCFS` by ≥ 0.001 without paying
elsewhere.

**Why the refutation does not cover it.** The refuted term was a quadratic *mean* inside `M`, and
its gradient-floor clip removed exactly the X-point neighbourhood that the Hausdorff **maximum** is
made of. A soft maximum is not a quadratic form, so it needs a torch loss with per-frame boundary
sampling — a different mechanism, untested.

**Honest limit.** The whole `D_LCFS` budget is 0.0017 and the failure frames that would have been
the easy part of it are already at zero. The worst return on the list, which is why it is last.

**Test.** Per-frame boundary maps for ~2k sampled frames, the penalty added for a short fine-tune,
one run per weight.

**Refuted if** Consistency pays more than `1 − D_LCFS` gains. **ΔS ≈ 0…+0.001, confidence 0.2.**

---

## Suggested order

Free diagnostics first, since two of them have already closed an entry without a run: the polarity
split (A12 step 0) and the scalar bias and slope (A7 step 0). Then the zero-retrain sweep A7 on the
artifact we already have; then A1 and A2, one run each; then A3, A4, A5, A6, A8.

Group B only after A1, A3 and A10 report — B1, B2 and B5 are their scaled versions, and B4 needs
the harness A7 builds.

**Dropped, with reasons.** Temporal smoothing of the predictions, which used to head this list —
measured 08-13 and refuted: the prediction is *twice as smooth* as the ground truth, and even
smoothing the tail components alone buys 2.7% of the loss-metric error, i.e. ΔS ≈ +0.0004 against a
resolution of 0.0013. Anything touching `n_pca` (saturated at production, measured both ways).
The quadratic boundary term (refuted; B8 is its only honest descendant). Per-component
standardization and ridge-in-the-ensemble (refuted). Polarity *normalization* (refuted; A12 is a
different operation and gated on a free diagnostic). Repairing frames where extraction fails
(refuted — both failure rates are 0.0% at production). Thomson (the correlation with the failing
shots was measured at ≈ 0 — but it costs no training time, so it is A11 now, not dropped).
A CNN or pixel decoder
(the coefficient bottleneck is saturated: Consistency stops improving at 50 components, so capacity
belongs in the regression — A3, B1, B5 — not in the decoder).
