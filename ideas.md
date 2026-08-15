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

**Rewritten 2026-08-15 from C1 and C2, which measured it rather than assuming it.** The model
scores 0.9945 and the ceiling of this pipeline is **0.9990** — so of the 0.0055 remaining, **0.0045
is reachable by fitting and 0.0010 the 50-component basis has already spent.**

Where the reachable 0.0045 sits: Consistency **0.0034**, R²qb 0.0005, D_LCFS 0.0005, R²ψ 0.0001.
Inside the geometry terms, by share of their cost: kappa **23.3%**, li **18.7%**, the LCFS distance
18.4%, R_axis 11.3%, volume 10.8%, Z_axis 6.8%, tri_top 4.5%, tri_bot 5.4%.

**Anything aimed at R²ψ or at the triangularities is capped below the noise floor** and is not on
this list. R²ψ is now measured at 0.9998 against a ceiling of 1.0000 — it is finished.

**And the shape of the loss, which is new.** The worst 1% of frames carry 22.4% of the geometry
cost and the worst 10% carry 57.9%; the last decile of a shot costs 2.4× its share and the first
1.5×. The loss weights every frame equally. That gap is the single most concrete thing this file
knows about where the remaining 0.0045 is hiding.

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

## A2. Flip `calibrate_scalars` on — *2026-08-13* — **DONE 2026-08-15, refuted**

**S 0.9945 against 0.9945** at production on salt 0 — exactly flat, against a resolution of 0.0013.
The seven scalars move by ±0.0012 with no pattern, and `kappa` — the one C1 said carried the most
cost and the one this entry expected to gain — went DOWN by 0.0010. Those moves are a twentieth of
the per-scalar seed noise and mean nothing individually.

**The "open problem" paragraph below called it, and that is the lesson worth keeping.** The probe's
ratios are 1.11–1.39, near enough uniform that dividing by them is close to a uniform rescale of
the Consistency block against psi — and `TargetScaler` renormalises the whole vector to unit average
variance afterwards, so most of even that is absorbed. A knob written down as promising, with a
paragraph underneath explaining why it probably would not work, did not work. The paragraph should
have been the rank.

**What is NOT refuted:** measuring the ratio along the MODEL's own error direction rather than from
isotropic probes, which is the two-pass version the paragraph proposes. That is a different
quantity — the model's error is not isotropic — and it remains untested.

### The original entry

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

**Read against C1, 2026-08-15, and the mechanism sharpens even though the ratios did not.** At
production the probe gives R_axis 1.11, li 1.15, kappa 1.16, volume 1.19, tri_top 1.22, Z_axis
1.23, **tri_bot 1.39** — a 1.6× spread in `ratio²` between the extremes, not the 2.7× the paragraph
above hoped for. But C1 now says which end of that spread matters: **the two most over-stated
Jacobians belong to the two triangularities, which carry 9.9% of the geometry cost, while `kappa`
and `li` — 42% of it between them — sit at the bottom of the range.** So the uncalibrated loss
spends its Consistency budget on the cheapest scalars and calibration moves it toward the dearest.
That is a smaller effect than the entry was written on and a much better argued one.

**Refuted if** paired ΔS ≤ 0, or `li` improves while `kappa`/`volume`/axis lose as much.

**ΔS ≈ +0.0005…+0.0015, confidence 0.45.** One boolean, no new code — the cheapest run on the list.

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

**The first half is done and the second half is now the whole entry — 2026-08-15.** Production
trains on 0.80 of the shots and holds 0.19 back to stop on. Those 1338 shots are read every
evaluation and never fitted, so **the refit converts a quarter more data into training data at no
methodological cost at all**, and the comparison stays clean because the scored tail is the same
1% either way. More data is also the most reliable winner this fork has: 1408 → 3168 shots bought
+0.0040 with every term up, and 0.60 → 0.80 bought +0.0018 with no sign of flattening.

**Exactly how**, so nothing is decided twice: run production as usual, read `best step` out of the
fit report, then refit at `0.99/1.0` with `patience_steps: 0` and `max_steps` set to that number
scaled by the data — the same number of PASSES, not of steps, since a step is a fixed batch and
there are now 24% more rows. Nothing to stop on and nothing that needs stopping.

**Refuted if** the paired gain is ≤ +0.0005.

**The honest risk.** The step count is transferred from a fit on less data, and the optimum moves
with the data — probably later, which the pass-count scaling is meant to absorb, but it is an
assumption and not a measurement. If the refit loses, the first thing to check is a step sweep
around the transferred count, not the idea.

**ΔS ≈ +0.0015…+0.003, confidence 0.7.** The highest-confidence open entry in this file, and the
one whose mechanism has already paid twice.

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

## A10. Derivatives, wherever the physics is a rate — *2026-08-13* — **DONE, accepted 2026-08-15**

**Reverted on 2026-08-13 at 0.60/0.1** (+0.0007 at best, below its floor) and **accepted on
2026-08-15 at 0.80/1.0** (+0.0010 on the selection salt, +0.0023 and +0.0013 on two unseen ones).
Same feature, same code, twenty-five times the frames.

The lesson is not "screen bigger", which is unaffordable. It is that a REFUTATION at a small scale
is only a refutation at that scale, and a feature whose value is an extra dimension the model has
to learn to use is exactly the kind that needs data before it pays. Entries refuted below the
production scale should say so, and this one now does.

### The original entry


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

**Provisional, and the reason is worth reading.** Every cell of this grid was fitted at a learning
rate that was later measured to blow up in 15 of 28 runs at these shares — training loss to 67.9
against unit-variance targets, then a recovery into a different place. So each cell carries a coin
flip that is worth about 0.002, and the marginal means average three or four of them. The four
readings agreeing in sign is what keeps the conclusion standing; it is not what a clean grid would
look like. **Re-run it at `grad_clip: 1.0` before spending anything on the strength of it** — and
note the deep-wide cell that "diverged" was the same instability, not a property of depth.

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

## A13. Stop on the composite, not on the validation MSE — *2026-08-14* — **DONE 2026-08-15, refuted**

**S 0.9940 against the control's 0.9940**, four seeds, production, salt 0, with the composite gated
behind a 10000-step loss plateau and measured over 400 validation frames. Per term the treatment is
0.9998 / 0.9965 / 0.9899 / 0.9780 against 0.9998 / 0.9962 / 0.9901 / 0.9785 — inside noise
everywhere, and marginally BEHIND on both geometry terms, which are the ones it was built to serve.

**And it closes its own confound without a third run.** The gated rule necessarily trains far past
where loss stopping ends — best step 84000 / 94000 / 98000 / 108000 against 58900 / 39150 / 80250 /
88550, over runs 35% to 100% longer — so a length-matched loss arm was pre-registered in case it
won. It did not, so the extra training bought nothing either. Both halves flat.

**What was actually wrong with the entry.** It rested on the loss and the score disagreeing, and the
disagreement is real at small scale and absent at production: over 68000 steps the validation loss
falls 0.0597 → 0.0324 and the composite rises 0.9852 → 0.9922 together, both flattening near step
42000. The fits are sitting on a plateau where nothing distinguishes one step from another, and no
stopping rule can find a peak that is not there. The evidence the entry was written on — a
sequence model with a lower loss and a worse composite — was about comparing two DIFFERENT
ARCHITECTURES, not two steps of one trajectory, and those are not the same claim.

**What survives, and it is not nothing.** `monitor.py` computes the real composite inside training
at 2.3 s per 150 frames, which is a tool this project did not have. B5's outstanding screen wants
exactly that for the sequence model, where the loss/score disagreement WAS measured between
architectures rather than assumed.

### The pre-registration, kept because it is the part that made the result readable

**Pre-registered before the score was seen**, because the first MLP's trace is already in hand and
it is not what the smoke run suggested.

At 0.005/1.0 the two measures diverged cleanly: the loss fell while the composite turned around. At
**production they do not**. Over 68000 steps the validation loss falls 0.0597 → 0.0324 and the
composite rises 0.9852 → 0.9922, together, and both flatten around step 42000–48000. So the
headline claim this entry was written on — that the loss keeps improving past the score's peak —
is a small-data effect, and at production the disagreement is much weaker than advertised.

**What the trace does show is a selection problem in the monitor, and naming it correctly matters.**
Adjacent evaluations swing ±0.0015 (0.9905, 0.9848, 0.9895 at steps 26000/28000/30000). That is NOT
resampling noise — the 150 frames are FIXED for the life of the fit, so consecutive evaluations
differ only because the weights do. It is real variation of the model on those 150 frames, which
makes the swing genuine and the problem worse: the rule keeps the argmax over ~34 evaluations of a
150-frame sample, so it selects the step that best fits **those** frames, and how that transfers to
the fold is exactly the selection-bias arithmetic this fork applies to everything else.

**And the third model shows the sharp edge of it.** `mlp2` recorded its best composite at step
**2000** — the very first measurement — and 18000 steps later had only matched it. With a patience
of 18400 steps counted on a 2000-step grid, an early lucky sample can end a fit that had not
started, and keep the weights that produced it. Loss-based stopping cannot do this, because the
validation loss at step 2000 is genuinely and hugely worse than at 20000.

**Read it this way.** Accepted if paired ΔS ≥ +0.0013 on salt 0 and it survives two unseen salts.
Refuted if ΔS ≤ 0. **Anything between is a monitor result, not an A13 result**, and the follow-up
is fixed in advance and has two parts, because the trace above shows two distinct faults:

- raise `loss.monitor_frames` from 150 to 600, so the thing being selected on is closer to the
  fold — four times the monitor cost, about 20 minutes on a four-MLP production run;
- **do not let the first measurement set the bar.** A warm-up before the composite is allowed to
  decide anything, or a floor on the step it may select, or simply judging on the composite only
  once the validation loss has stopped improving. The last is the most honest of the three: it uses
  the loss for what it is reliably good at — telling an undertrained fit from a trained one — and
  the composite for what it is uniquely good at, which is choosing among fits that are all trained.

That third option is arguably what A13 should have been from the start, and it is now the version
worth running.

**The confound in the gated version, written down before its score is known.** It changes two things
at once. Measured on the first three seeds: best step 84000 / 94000 / 98000 against the control's
58900 / 39150 / 80250, and the fits run 104000 / 114000 / 118000 steps against 77300 / 57550 /
98650. That is not an accident of the criterion, it is built into the rule — the composite phase
cannot begin until the loss has been flat for 10000 steps and then spends its own 18400 of patience,
so a gated fit trains at least 28400 steps past the loss peak where the control stops at 18400.

So **if it wins, the win is not attributed until a third arm runs**: loss stopping with the patience
raised to match the treatment's step counts. If that alone matches the gated composite, the finding
is "the patience was too short" — a one-line change, and a much cheaper one than a scorer in the
training loop. Only if the composite beats a length-matched loss control does A13 own the result.

### The original entry

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

## A19. A multi-scale derivative bank, not one half-width — *2026-08-15*

**Hypothesis.** Replacing the single 1 ms centred derivative with a bank at ~0.2, 1, 5 and 20 ms,
plus a few leaky integrals of the loop voltage at 10, 50 and 200 ms, gains ≥ 0.001 over A10's
accepted `both/poloidal`.

**Mechanism, and it is the sharp part.** `RAW_DERIV_HALF_MS = 1.0` was chosen as "short against the
equilibrium, long against the noise" — a reasonable guess, never measured, and there is no reason
one number serves every process. The vessel's eddy currents decay over milliseconds to tens of
milliseconds; current diffusion in the plasma runs to hundreds. A derivative at one scale is a
band-pass at one frequency, and the two physical timescales that matter sit on either side of it.
The **leaky integrals** are the more interesting half: an exponential moving average of `dI_OH/dt`
with time constant τ *is* a first-order estimate of the vessel current driven by that voltage, so a
bank of them is a hand-built state observer — exactly the state a memoryless MLP cannot form, given
as a feature instead of as an architecture. If it pays, it is B5's mechanism at a hundredth of the
cost; if it does not, that is evidence against B5's mechanism too.

**BUILT 2026-08-15, and pre-registered here before it runs.** `features.vessel` is `none` /
`driving` / `poloidal`, four leaky integrals per signal at 20 / 100 / 500 / 2500 ms — the bank set
by C6's measurement of what the sequence model chose for itself, not by the L/R argument alone.
FEATURE_WIDTH 124 → 208, so the first run rebuilds the shot cache; measured at 2.2–2.8 s of feature
work per shot, about 20 minutes over the corpus on 24 processes.

**Two arms, and both are run before anything is concluded, because they test different claims.**
`poloidal` is the eddy-current claim over every flux-producing coil and is what won for the
derivatives; `driving` is the two signals the physics names and is the sharp version. A10's history
is the reason both go: the broad set won there, and a broad set losing does not refute the narrow
one.

**Read against 0.9940**, which is `mlp+mlp1+mlp2+mlp3` on the tail from the production artifact —
four seeds, loss-stopped, no vessel block. Accepted at +0.0013 paired and confirmed on two unseen
salts. **And read the last decile separately**, with `diagnose_frames.py`: this entry exists because
of A22, so an arm that lifts the total by nothing while cutting the last decile's cost is a
different and more interesting result than an arm that does neither.

**The specific way it could fail, written down first.** 80 new columns against 84 existing ones
doubles the input width, and this fork has already measured that nearly-constant columns are not
free — `frame_gaps` cost 0.0012 for exactly that reason. If `poloidal` loses and `driving` does not,
that is dilution rather than a refutation of the physics.

**Refuted if** the bank is within 0.0005 of the single scale, which would say the 1 ms guess is
already on the flat of a broad optimum.

**ΔS ≈ +0.0005…+0.002, confidence 0.4.** Ranked above the architecture ideas that share its
mechanism because it is one cache rebuild and no new model.

## A20. Average the weights, not only the seeds — *2026-08-15*

**Hypothesis.** An exponential moving average of the weights over the last ~10% of steps, or a
Polyak average from the plateau onward, buys a share of the four-seed gain at zero extra training.

**Mechanism.** A1 measured σ ≈ 0.0013 of S across seeds and called it optimisation noise, and
averaging four fits paid +0.0042. Some of that noise is *between* seeds and only more seeds fix it;
some of it is the last few thousand steps rattling inside one basin, and a weight average removes
that part for free. The two are additive — an EMA of each member, then the ensemble of the four.

**Test.** One `torch.optim.swa_utils.AveragedModel` in the MLP fit, evaluated at the same stopping
point as the raw weights, so it is a paired comparison inside a single run.

**Refuted if** the averaged weights score within 0.0003 of the raw ones — which would say the
plateau is already flat and early stopping is landing in the middle of it rather than on a rattle.

**ΔS ≈ +0.0003…+0.0015, confidence 0.5.** The cheapest entry on the list: no extra run, no extra
cache, no new hyper-parameter that has to be tuned.

## A22. The last tenth of a shot — *2026-08-15* — **the target C1 and C7 actually found**

**The measurement, before the hypothesis, because this entry exists only because of it.** The final
decile of every shot carries **24.1% of the geometry cost over 10.2% of the frames**, and C7 says
this is not the metric's doing: sensitivity there is 0.98× the median while the implied coefficient
error is **2.22×** the fold average. Every other decile sits between 0.51× and 1.03×. The model is
worse at the end of a shot than anywhere else, by a factor of two, and nothing in this project had
noticed.

**Hypothesis.** Something about ramp-down is either absent from the inputs or under-represented in
the fit, and addressing it wins ≥ 0.001 — perfect prediction there would be worth 0.0011 of the
0.0039 reachable in the geometry terms.

**Three candidate mechanisms, and they are separable.**

- *Missing state.* The current is decaying fastest, so the vessel eddy currents are largest and
  most transient exactly here — the one thing 21 instantaneous levels cannot express. This is
  A19's leaky integrals of the loop voltage, and A22 gives them a place to show up. **The cheapest
  test in the entry: rerun C1 with the A19 features and read the last decile alone.**
- *Under-representation.* The tail may simply be a small, differently-distributed slice — low Ip,
  a small or deforming plasma — that a flat loss lets the flat-top dominate. Then a per-frame
  weight by position is the fix, and unlike B6's Jacobian weight it is aimed at the frames that
  actually cost.
- *Label noise.* EFIT's own reconstruction is least constrained as the plasma terminates, so part
  of that 2.2× might be irreducible. **STILL OPEN — the measurement below does not test it, and an
  earlier version of this entry claimed that it did.** What was measured is that the ceiling
  decomposed onto the same frames is FLAT across the shot — 0.000086 to 0.000120 per decile, the
  last at 0.000120 against a fold mean of 0.000100, with only **10.1% of the last decile's cost
  coming from the basis** against 21–33% in the middle. That refutes a different hypothesis: the
  tail is not harder to REPRESENT. A noisy label is represented perfectly well — the basis
  reconstructs whatever it is given — so a flat ceiling is exactly what label noise would also
  produce. **The seed test has now been run and it narrows the question without closing it.** Mean
  pairwise correlation of the four seeds' residuals over the seven scalars is **+0.550 in the last
  decile against +0.615 in the middle** of the shot. So a majority of the tail's error is SHARED
  across seeds — bias of this model-and-feature class, which more seeds cannot remove — and the
  tail is if anything slightly MORE variance-driven than the middle, not less. Two consequences:
  averaging is bounded here (it can only touch the ~45% that is independent), and the shared 55%
  is either a missing input or an unlearnable label, which this test still cannot separate.

| decile | model | ceiling | reachable | ceiling as a share of the cost |
|---|---|---|---|---|
| 0.0–0.1 | 0.000742 | 0.000110 | 0.000632 | 14.9% |
| middle eight | 0.000316–0.000429 | 0.000088–0.000110 | — | 20.5–33.1% |
| **0.9–1.0** | **0.001182** | 0.000120 | **0.001062** | **10.1%** |

**So the bound is 0.00106 of the 0.00390 reachable — 27.2% of everything left, in 10% of the
frames.**

**The sequence model already had its chance here and did not take it.** It saves 15.2% of the last
decile's cost against 24.9% of the first decile's, so a bidirectional pass over the shot — which
sees the termination coming — is not what closes this.

**Refuted if** weighting and features both leave the last decile within 0.0003 of where it is. The
representation-floor branch has already been tested and refuted, above.

**ΔS ≈ +0.0005…+0.0015, confidence 0.45.** Bounded above by 0.00106, and that bound is measured —
which is more than any other open entry in this file can say.

## A23. Spend the confidence: ensemble weights that vary, and shrinkage — *2026-08-15*

**Both halves need NO retraining at all**, which is what puts this above everything else outstanding
except A22. C8 measured a per-frame confidence that predicts the cost at Spearman +0.565 and costs
nothing, and two ways to spend it fall straight out of measurements already taken.

*Weights that vary with position or with confidence.* A21 below fits ONE weight per member. But the
optimal weight demonstrably is not constant: the recurrence removes **+2.4%** of the cost in the
first decile of a shot and **+14.5%** in the eighth — a six-fold spread on a member that carries
0.20 everywhere. Fitting `w(phase)` or `w(disagreement)` on the reserved holdout is the same tiny
least-squares problem as A21 with a few more parameters, and `_predict_targets` already averages in
the scaled target space, so the code touched is one function.

**The GLOBAL weight was swept first, and it is already right.** Ten weightings on a selection set
give a smooth unimodal curve peaking at 0.33–0.43 against production's 0.20 — worth +0.0002 there
and **exactly nothing on the tail**, where 0.20, 0.33 and 0.43 all score 0.9945. So there is no
constant-weight gain to collect, and what is left of this half is only the VARYING version, which
has to beat a flat weight that is already sitting on its optimum. That is a harder bar than the
decile table made it look.

*Shrinkage where the model is unsure.* R² is squared error, and under squared error the optimal
estimate of an uncertain quantity is shrunk toward the prior mean. `ĉ' = ĉ − λ·f(disagreement)·(ĉ −
c̄)`, one scalar λ swept on the artifact that already exists — the same zero-retrain harness A7
builds. The 50 coefficients have a natural `c̄`: the PCA basis is centred, so it is zero.

**The adversarial note, stated before either runs.** Both are fitted on a holdout and both add
parameters to a decision that is currently free of them; the fork's own history says selection at
this scale costs about 0.0013 unless confirmed on salts nothing selected against. Neither is
exempt from that rule because neither costs a training run.

**The trap worth writing down, because it is the textbook answer and it is wrong here.** Predicting
a per-frame σ and minimising a Gaussian NLL — the standard way to model aleatoric uncertainty —
optimises in the WRONG DIRECTION for this metric. NLL down-weights frames with large predicted σ,
so the model learns to ignore what it cannot predict, while a pooled R² charges for those frames
regardless. A heteroscedastic head trained that way should be expected to LOSE score, and the
version that could help is the opposite sign: up-weight, i.e. hard-example mining, which is the
third use and carries B6's own caveat that hard is not the same as under-weighted.

**Refuted if** the fitted varying weights land within 0.0005 of the flat ones on unseen salts, or
the shrinkage sweep's optimum is λ = 0.

**ΔS ≈ +0.0005…+0.0015, confidence 0.45.**

## B11. A loss metric that changes with the regime — *2026-08-15* — needs the loss reworked

**The axis nobody has looked at.** The `jacobian` metric is ONE 50×50 matrix, `JᵀJ` averaged over
300 probe frames. C7 measured how its TRACE varies frame to frame — 10× — and found that the
variation does not explain the cost. What it did not measure is whether `J` points in different
DIRECTIONS: a plasma on ramp-down has a different shape, so a different combination of coefficients
moves its `kappa`, not merely a larger one. Averaging `JᵀJ` blurs direction as well as scale, and
only scale has been tested.

**The free diagnostic, and it decides the entry.** Partition the 300 probe frames by phase into
three regimes, build `M` on each, and measure the principal angles between the resulting subspaces
(or simply how much `L_regime` differs from the pooled `L`). If the directions agree, this is only
the scale again and the entry dies for the cost of three eigendecompositions.

**Honest cost if it survives.** `TargetScaler` folds `L` into the target transform ONCE, globally,
which is what makes the loss a plain MSE afterwards and what makes ensemble averaging in the scaled
space valid. A per-regime `L` cannot live there — it would make the target space depend on the row.
It has to move into the MLP loss as a per-row quadratic form instead, which is a real rework of the
training step and of the early-stopping metric with it.

**Refuted if** the principal angles between the three regime subspaces are small, which after C7's
result is the way to bet. **ΔS ≈ 0…+0.002, confidence 0.2.**

## A21. Fit the ensemble weights instead of setting them — *2026-08-15*

**Hypothesis.** Solving for the ensemble weights on a holdout, rather than writing 0.6/0.4 in
`params.yaml`, gains ≥ 0.0005 — and per-block weights (one set for ψ, one for q95/βN) gain more.

**Mechanism.** The members are not equally good and are not equally good *at the same things*: the
sequence model loses on geometry and the dedicated q95/βN question has already come up in
conversation. Averaging in the scaled target space is linear, so the optimal weights are a tiny
least-squares problem — and with 50 coefficients over thousands of holdout frames, the fitting risk
is small enough to bound by leaving the weights on the simplex.

**Test.** `_predict_targets` already parses per-block combinations; the missing piece is a solve on
the holdout that `reserve_holdout` already carves. No retraining at all.

**Refuted if** the fitted weights land within noise of the hand-set ones, which is likely for two
similar MLPs and much less likely once CatBoost and the sequence model are in.

**ΔS ≈ +0.0003…+0.0015, confidence 0.45.**

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

## B5. A sequence model over the shot — *2026-08-13* — **BUILT AND MEASURED 2026-08-15**

**Built, and it loses.** 0.9914 against the MLP's 0.9931, worse on every term of the composite,
with a LOWER validation loss and a recurrence that is demonstrably in use — the correction branch
grows to 31.6% of the per-frame prediction, so it is not falling back to being the MLP.

**The diagnosis, which is worth more than the number.** This entry was downgraded on 08-13 because
smoothing had been refuted: the prediction is already half as rough as the truth, 0.021 against
0.039. That refutation turns out to apply to the architecture too, and more sharply than to the
post-hoc filter — a bidirectional pass over the shot is a smoother, however much state it carries,
and the terms it costs most are exactly the ones that read the map's geometry (R2_qb −0.0040,
Consistency −0.0049) rather than its values (R2_psi −0.0001).

**What is NOT refuted by this**, and what a second attempt would have to change: the model was
given the sequence and the same loss. A sequence model whose loss charged it for per-frame geometry
— A13's composite-aware stopping, or B6's per-frame weighting — is a different experiment, and the
one this result argues for. What is refuted is "read the shot, keep everything else".

### The original entry


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

### How far back does it actually reach? — *2026-08-15*

**Unmeasured, and the architecture cannot answer it.** Nothing about a GRU fixes a horizon: the
update gate decides per channel and per frame how much of the state survives, so the reach is a
property of the trained weights. The page describing this model says "in principle unbounded",
which is true and useless.

**The cheap measurement, which needs no training.** Run the fitted model over the validation shots
and collect `z` for all 256 channels at every frame. A channel holding `z` steady has a time
constant `tau = -1/ln(z)` frames, and a frame is 20 ms:

| z | tau | in shot time |
|---|---|---|
| 0.5 | 1.4 frames | 0.03 s |
| 0.9 | 9.5 frames | 0.19 s |
| 0.99 | 99.5 frames | 2.0 s |
| 0.999 | 999 frames | 20 s |

So the histogram of `tau` over the 256 channels is a direct read of what memory the model chose to
build. Two outcomes and they mean opposite things: a spread reaching seconds says the recurrence
is carrying slow plasma state, which is the mechanism this model was built for; everything piled
near `z ~ 0.5` says it learned a two-frame smoother and the +0.0015 in the ensemble comes from
something else entirely.

**Cross-check, because a gate value is an inference and not an effect.** Perturb frame `t-k`, watch
the prediction at `t`, sweep `k`. That measures the reach end to end, including whatever the
encoder and `delta` do to it, where the gate statistics only measure the recurrence.

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

### The screen was never run — *2026-08-15*

**Five of the six arms are outstanding**, and this is the largest unpaid debt on the sequence model.
Arm `q1` stopped at `best step 40000 of 40000` — it measured the step budget, not the
hyper-parameter — and the grid was abandoned there rather than spending 2.4 h sweeping five
learning rates against a ceiling. So every number this architecture has produced, including the
0.9914 that refuted it as a replacement and the +0.0007 that accepted it into the ensemble, comes
from **one untuned configuration**, where the MLP beside it was tuned over thirty runs.

**Order matters here.** Raise `max_steps` until the best step lands strictly inside the budget on
`q1`, and only then run `q2…q6`. Arms read against a truncated control measure the truncation. The
same applies to the two knobs the original six do not cover and that this model has more reason to
want than the MLP does: `weight_decay`, since 5633 sequences against 1.25M rows is three orders of
magnitude fewer examples per parameter, and the split of capacity between `enc`/`gru` and the
feed-forward `head` — 789k of the 1.59M parameters are in the recurrence, a ratio nothing chose.

**ΔS unknown, and that is the point.** A tuned recurrence could still lose to the MLP, but the
current verdict does not distinguish "the architecture is wrong" from "the architecture was run at
someone else's learning rate".

## B9. Learn the 50-dimensional subspace instead of taking PCA's — *2026-08-15* — **PREMISE REFUTED**

**C2 ran the free diagnostic this entry asked for, the same day, and it came back against.** The
argument below rests on the geometry terms having 0.0175 of headroom that a variance-optimal basis
cannot reach. Measured: ground truth through the artifact's own 50 components scores **0.9990**, so
the basis costs **0.0010** of S in total — 0.0005 on `D_LCFS` and 0.0005 on Consistency. The
refutation clause at the bottom of this entry is met exactly as written.

**What is left of it, honestly.** Representability is settled and dead. *Predictability* — the
reduced-rank half, choosing directions that the 86 magnetics numbers can actually determine — is
untouched by C2, which only asked what the basis can represent. But it has lost the argument it was
ranked on, its ceiling is now the 0.0045 the regression is leaving rather than anything larger, and
it competes against B6, which C1 measured *for*. Keep it below B6 and below A13.

### The original entry

**The gap this points at.** PCA chooses its 50 directions to maximise explained PIXEL VARIANCE.
The composite does not score pixel variance: 30% of it is `D_LCFS` and Consistency, which read the
map's GEOMETRY and not its values, and the `jacobian` loss metric exists precisely because of that
— it reweights the loss ON the coefficients. But the SUBSPACE is still variance-optimal. A
direction that barely moves the pixels and moves the magnetic axis a long way can be absent from
the basis entirely, and no reweighting of what is there can recover it.

**Two different objectives, and the second is the stronger one.**

*Representability.* Can 50 directions reconstruct the map as the METRIC measures it, rather than
as pixel variance does? That is a generalised eigenproblem on `M` and the flux covariance, both
already computed.

*Predictability, which is the one that matters.* A direction can be important to the score and
simply not be inferable from 86 magnetics numbers — and a slot spent on it is wasted, while a
slightly less important direction the model predicts perfectly is worth more. PCA is blind to this
by construction: it never looks at `X` at all. What does look at both is **reduced-rank
regression** — choose the subspace that maximises the metric-weighted variance of the FITTED
values rather than of the targets. Same shape of computation, different matrix: the covariance of
`X B` instead of the covariance of `Y`.

Stated as one criterion: **spend the 50 directions on what is both worth predicting and possible
to predict.** The `jacobian` loss metric was reaching for the first half of that and could only
reweight what was already in the basis; this decides the basis.

**The free diagnostic, before any training.** Fit the full-rank map from features to flux — ridge
already does exactly this and is in every artifact — and compare three subspaces by the
metric-weighted error they leave: PCA's 50, the metric-optimal 50, and the reduced-rank 50. Three
eigendecompositions, no fitting. If PCA is within noise of both, this entry dies for the price of
an afternoon's arithmetic. If the reduced-rank subspace is clearly better, that number is an upper
bound on what the change could pay, before anything is retrained. **Do this first.**

**What it collides with, and this is the important half.** Two things that currently pay depend on
the decoder being AFFINE:

- **The ensemble.** "Averaging coefficients and averaging flux maps are the same thing — the PCA
  decoder is affine and the weights sum to 1." With a nonlinear decoder that identity breaks:
  averaging latents is no longer averaging maps, and the ensemble would have to average decoded
  65x65 maps instead. Workable, but it is worth +0.0015 to +0.0042 here, so it is not a detail.
- **The Jacobian loss metric.** "Any quadratic functional of the map error is a fixed matrix on
  those coefficients" — true because the decoder is affine. A nonlinear decoder turns that fixed
  matrix into something that depends on where you are, and the +0.0097 that metric is worth would
  have to be re-derived or given up.

**So the sharp version is a LINEAR encoder-decoder**, trained to minimise the metric rather than
the pixel error. It keeps both properties above — still affine, still one matrix — and differs from
PCA only in which 50 directions it picks. All of the potential gain, none of the collateral.

**The nonlinear version is a separate, later entry**, and it should be judged against having to
rebuild the ensemble and the loss metric around it, not against PCA alone.

**Refuted if** the diagnostic shows PCA's subspace already near-optimal in the metric — which is
plausible, since 50 components reconstruct ~100% of the variance and the headroom on `R2_psi` is
0.0004 of S. The hope rests entirely on the geometry terms, where the headroom is 0.0175.

## B10. A second sequence model, on the 0.05 ms stream — *2026-08-15* — a cache rebuild plus a new encoder

**The observation this rests on, and it is the only one of its kind left.** Every idea in this file
rearranges the same 21 numbers per EFIT frame. The magnetics are not sampled at 20 ms — they are
sampled at **0.05 ms**, four hundred samples per frame, and the pipeline throws all of them away
except through one 1 ms centred difference. That is the only place in the project where genuinely
unused *information* sits, as opposed to unused capacity or a better arrangement of what is already
in.

**Hypothesis.** A two-rate model — a small strided convolutional encoder over each frame's ~400 raw
samples, whose 32-dimensional output joins the existing per-frame features feeding the shot-level
GRU — beats the frame-rate sequence model by ≥ 0.002.

**How the two clocks join, which is the question the idea arrives with.** Not by resampling
anything to a common grid: the fast encoder is applied *per frame* over the window `[t − 20 ms, t]`
and returns one vector per frame, so it is a feature extractor whose output already lives on the
EFIT clock. The shot-level GRU is unchanged. This keeps the two rates in the two places they
belong and needs no new alignment logic — and it composes with everything, because the frame-rate
interface (`predict_row` over a shot's frames) never changes. The alternative, one recurrence
running at 20 kHz over the whole shot, is 94k steps per shot and is not worth considering until the
cheap version says there is something down there.

**Gate, and it is a hard one: run A19 first.** A19's fixed multi-scale bank asks the same question —
is there anything in the sub-frame stream? — with four numbers per signal instead of a learned
encoder. If a bank spanning 0.2 to 20 ms buys nothing, the claim that a conv net over the same
samples finds something is a claim that the useful structure is not in any band, which is possible
but is a much thinner reed than "the vessel has fast dynamics".

**Cost, honestly.** The decoded-shot cache is 25 GB and holds interpolated features; the raw stream
is ~8 MB per shot, so caching it for 5633 shots is ~44 GB — the same order as the parquets it comes
from, but it is the dominant cost of the entry and it lands before any model exists.

**Refuted if** A19 is refuted, or if the encoder's output is driven to near-zero variance in
training — the same delta-share reading B5 already uses, applied to the fast branch.

**ΔS ≈ 0…+0.004, confidence 0.25.** The widest interval in this file, and deliberately: it is the
only entry whose upside is new information rather than better use of old.

## B6. A per-frame heteroscedastic metric — *2026-08-13* — **DEMOTED by C7; A22 replaces it**

**C7 measured the premise and it points the other way.** Sensitivity does vary — 10× between the
99th percentile and the median, so the refutation clause below is not met. But it explains almost
none of the cost (Spearman +0.209, and only the top quintile is elevated at all), and the
decile table is decisive: the frames a Jacobian weighting would up-weight are the FIRST decile,
where the model's coefficient error is already half the fold average. The frames that actually cost
— the last decile, 24% of the geometry loss — have ordinary sensitivity and 2.2× the error, and a
`J`-based weight leaves them exactly where they are.

**Keep it, below A22, for the one thing it would still buy.** Ramp-up costs 1.57× its share on
sensitivity alone, which is ~0.0004 of S, and that part is genuinely a mis-weighting. It is a
smaller prize than this entry was written for and it is not first any more.

### The original entry, and the reasoning that C1 alone seemed to support

**Hypothesis.** The fixed `M` (JᵀJ averaged over 300 probe frames) under-weights high-sensitivity
frames; weighting each training frame by a cheap surrogate for `tr JᵀJ` gains ≥ 0.001.

**Mechanism.** A pooled R² weights frames equally, but equal coefficient error costs unequal scalar
error across frames, and the loss should charge the frames where `J` is large. Averaging `M` blurs
exactly that.

**C1 measured the premise, 2026-08-15, and it is larger than this entry assumed.** The refutation
clause below asked for a 2× spread in per-frame sensitivity. What the score actually shows is the
worst **1% of frames carrying 22.4%** of the geometry cost — a 22× concentration — with the worst
10% at 57.9%, and the ends of the shot at 2.4× and 1.5× their share. The loss is flat across all of
it. This is now the best-evidenced entry in the file and it moves to the front of Group B.

**Two versions, and the cheap one comes first.** The Jacobian surrogate is the principled one and
needs the probe. But C1 hands over a weighting that needs no Jacobian at all — position within the
shot, or `|dIp/dt|` — and a sample weight is one line in the MLP loss. Run the cheap one to find
out whether frame weighting helps *at all* before deciding how cleverly to weight.

**The adversarial reading, which has to be stated before the run.** A frame that costs more may
cost more because it is HARDER, not because it is under-weighted, and up-weighting hard frames
trades the other 90% away for them. The 22× is a fact about the residual, not yet about the
gradient. This is exactly the shape of argument that has been refuted twice in this fork.

**Test.** Per-row sample weights in the MLP loss, one run per strength, paired on one salt, then
confirmed on two salts nothing selected against.

**Refuted if** no strength clears +0.0005 paired — the flat weighting is then correct and the tail
is difficulty rather than mis-weighting. **ΔS ≈ +0.001…+0.003, confidence 0.45.**

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

# Group C — diagnostics, which cost no training at all

Two entries in this file have already been closed by a free measurement rather than by a run, and
the project's two reversals both happened because a number was taken from the wrong scale. These
cost an afternoon each, none of them needs a fit, and every one of them changes what is worth
running next. **They come before Group A.**

## C1. Where the score is actually lost — *2026-08-15* — **DONE**

`my_experiments/diagnose_frames.py`, 70 held-out shots, 14830 frames, the two geometry terms
decomposed onto every frame on one additive scale that sums to the pooled terms exactly.

**The loss is heavy-tailed.** The worst **1% of frames carry 22.4%** of the geometry cost, the worst
5% carry 45.8%, the worst 10% carry 57.9%. So it is neither uniform nor a handful of disasters: a
long tail, 22× over its share at the top and still 2× at the tenth percentile.

**It is the ENDS of the shot.** By position: the last decile **24.1%** of the cost over 10.2% of the
frames, the first decile 15.1%, and the middle eight flat between 6.4% and 8.7%. Ramp-down and
ramp-up carry 39% of it over 20% of the frames. `|dIp/dt|` says the same more weakly — 31.5% in the
top quartile against 17.8% in the bottom — which is what you would expect if the ends are the
mechanism and the current rate is only a proxy for them.

**The holes barely matter in total.** A frame after a gap of 2 costs 3.6× its share and one after a
gap of 3+ costs 6×, but they are 1.3% of the frames and 5.5% of the cost together. That is a
consistent third measurement against `frame_gaps` as a feature: the effect is real per frame and
too small to pay for two nearly-constant input columns.

**Which scalar.** kappa **23.3%**, li **18.7%**, the LCFS term 18.4%, R_axis 11.3%, volume 10.8%,
Z_axis 6.8%, tri_top 4.5%, tri_bot 5.4%. So the elongation and the internal inductance alone are
42% of everything the geometry terms cost, and the triangularities are 9.9% — which confirms the
note at the top of this file that anything aimed at them is below the noise floor.

**What it points at, and it is not what was expected.** The frames are not exotic: no failure
class, no extraction breakdown (0.0% both ways), nothing that a targeted repair could catch. What
there is instead is a distribution the loss ignores — a pooled R² charges every frame the same, and
this is a 22× spread. That is **B6** stated as a measurement rather than a hypothesis, and B6's own
refutation clause ("refuted if per-frame variation is under ~2×") is answered the other way.

## C2. The representation floor, per term — *2026-08-15* — **DONE, and it kills B9's premise**

`evaluate.py --mode basis`: ground truth pushed through the artifact's own 50 components and back,
scored on the same 70 shots. q95 and betaN are handed over exactly, since they are regressed
directly and never pass through the basis.

| term | model | **ceiling** | reachable | basis floor |
|---|---|---|---|---|
| R²ψ | 0.9998 | **1.0000** | 0.0001 | 0.0000 |
| R²{q95,βN} | 0.9967 | **1.0000** | 0.0005 | 0.0000 |
| 1 − D_LCFS | 0.9902 | **0.9950** | 0.0005 | 0.0005 |
| Consistency | 0.9802 | **0.9974** | 0.0034 | 0.0005 |
| **S** | **0.9945** | **0.9990** | **0.0045** | **0.0010** |

**Of the 0.0055 remaining, 0.0045 is reachable by fitting and 0.0010 is the basis.** And 76% of the
reachable part is in Consistency alone.

**This refutes B9's premise as stated.** That entry was written on "the headroom on the geometry
terms is 0.0175", and the measurement says the 50 PCA directions reproduce the geometry the scorer
reads to within 0.0010 of S. Its own refutation clause — "refuted if the diagnostic shows PCA's
subspace already near-optimal in the metric" — is met. What survives is the narrower half:
reduced-rank regression is about which directions are PREDICTABLE, not which are representable, and
nothing here measures that. But it has lost the argument it was ranked on, and it cannot be worth
more than the 0.0045 the regression is leaving on the table.

**The finding that reorders the rest.** The model's ψ is essentially perfect as pixels — 0.9998
against a ceiling of 1.0000 — and the scalars derived FROM that ψ are off by 2–4% of variance. The
basis can produce those scalars to 0.99+; the fit does not. So this is not representation and it is
not capacity: it is that the loss and the score disagree about where on the map the error matters.
`kappa` is the sharpest case — the worst ceiling of the seven (0.9908) *and* the most expensive
scalar (23.3% of the geometry cost).

## C3. Permutation importance on the artifact we already have — *2026-08-15*

Permute one input block across frames, re-score, restore, repeat: 21 signals, two derivative blocks,
the gap columns, the Thomson block. No fitting, one scoring pass each. Answers questions the
feature entries currently guess at — whether the accepted derivatives are used or merely harmless,
whether Thomson is dead weight, whether the model is essentially reading Ip and the shaping coils —
and it prices A19 and B10 before either is built. Caveat worth writing down: permutation measures
what *this* fit uses, not what is usable, so a zero is weaker evidence than a large value.

## C4. The residual's structure in time — *2026-08-15*

Split the metric-weighted error into a per-shot constant, a slow within-shot component and the
rest. A per-shot offset has been noticed before and "turned out not to be explained by current
history at all", but its *share* was never measured. If it is large, a bidirectional pass over the
shot should have removed it almost for free — and B5 measurably did not, which would mean the
constant is not inferable from the inputs at all and bounds every architecture equally. If it is
small, B5's failure is about geometry and A13's composite stopping is the fix.

## C5. Does anything transfer to MAST? — *2026-08-15*

`predict_row` raises `NotImplementedError`, so Challenge 2 currently scores **zero** — and the
top-of-file ranking already calls this the largest structural gap left. Before any modelling: which
of the 21 signals exist on MAST at all, what the flux grid and the frame clock are, and what the
present DIII-D artifact scores when pointed at MAST shots with whatever features do exist. That
number can only be embarrassing or encouraging, and either one is worth more than a refinement on a
challenge already won.

## C8. Does the model know where it is wrong? — *2026-08-15* — **DONE, and the answer is yes, cheaply**

The four seeds are already fitted, so their per-frame disagreement is an uncertainty estimate that
costs nothing to compute. Taken as the spread of the seven derived scalars across the four seeds,
normalised per scalar and summed, against the ensemble's own per-frame cost on 14830 frames:

**Spearman = +0.565**, against **+0.209** for the Jacobian sensitivity C7 measured. And unlike
sensitivity the relation is monotone — mean cost by quintile of disagreement is 0.20× / 0.32× /
0.52× / 0.99× / **2.97×**, with the top fifth of frames carrying **59.3%** of the cost.

So the model's own disagreement locates the loss two and a half times better than the metric's
linearisation does, and it is free.

**The blind spot, and it sits exactly on the money.** By decile the disagreement is 1.55× at the
start of a shot against a cost of 1.54× — calibrated — but **1.19× at the end against a cost of
2.61×**. It under-reports the tail by more than a factor of two, and the reason is already measured:
seed residuals correlate +0.550 there, so a majority of the tail's error is SHARED, and shared error
is invisible to disagreement by construction. Four seeds that are all wrong the same way look
confident. **Ensemble disagreement measures variance and is blind to bias.**

## C7. Is an expensive frame under-weighted, or simply harder? — *2026-08-15* — **DONE, and it redirects B6**

2998 probed frames over the same 70 shots, joined to C1's costs.

**Sensitivity does vary — 10×.** p99/median = 10.1, p90/p10 = 6.8. B6's refutation clause asked for
less than 2× and it is not met, so the premise stands.

**But it explains almost none of the cost.** Spearman(sensitivity, cost) = **+0.209**, and the
relation is not a gradient — it is a step at the very top. Mean Consistency cost by quintile of
sensitivity: 0.74, 0.79, 0.65, **0.56**, 2.25. The fourth quintile has 1.38× the median sensitivity
and the LOWEST cost of the five.

**And the two ends of the shot are expensive for opposite reasons.** Dividing the decile's mean cost
by its median sensitivity gives the coefficient error the model is actually making there (a
decile-level ratio, not a per-frame quantity):

| decile | sensitivity | cost | implied error² |
|---|---|---|---|
| 0.0–0.1 | **3.05×** | 1.57× | **0.51×** |
| 0.1–0.9 | 0.84–1.24× | 0.62–0.91× | 0.54–1.03× |
| 0.9–1.0 | **0.98×** | **2.17×** | **2.22×** |

**Ramp-up is B6's case exactly** — the model predicts it unusually WELL (error² half the average)
and it costs 1.57× only because the functionals are three times as sensitive there. **Ramp-down is
the opposite and it is the bigger lump**: ordinary sensitivity, 2.2× the coefficient error, 24% of
the whole geometry cost. Nothing about the metric's weighting touches it. The model is simply wrong
at the end of a shot.

**So B6 aimed the wrong way.** A Jacobian-weighted loss up-weights the first decile, which is
already the best-predicted region, and leaves the last decile untouched. See A22, which is what
this measurement actually argues for.

**One more reading — first taken CONFOUNDED, then run properly, and the answer flipped.** Comparing
`mlp` alone against the ensemble gave 0.00113 and was written up as "the recurrence helps least
where its physical argument is strongest". That comparison mixed four seeds with the recurrence.
With `mlp+mlp1+mlp2+mlp3` as the control instead:

| | geometry cost | saves |
|---|---|---|
| the four seeds alone, mean | 0.00700 | |
| averaging them | 0.00529 | **0.00171** |
| adding `seq` on top | 0.00490 | **0.00039** |

So seed averaging does four fifths of it and the recurrence one fifth — and where it does that fifth
reverses the earlier reading. By decile the saving climbs monotonically toward the end of the shot:
+2.4% at 0.0–0.1, ~5–8% through the middle, **+14.5% at 0.8–0.9 and +10.1% at 0.9–1.0**. In absolute
terms **34% of everything the recurrence buys comes from the last decile and 52% from the last two**,
out of 20% of the frames.

**That is evidence FOR the missing-state branch of A22.** The one member that carries state helps
most exactly where the memoryless models are worst, which is what "the vessel current is largest
and most transient during ramp-down" predicts. It also raises B5's outstanding screen from a debt
to a live question.

`my_experiments/diagnose_sensitivity.py`. **The diagnostic that decides B6**, and it exists because
C1 by itself cannot: a frame's Consistency cost is `|| J_i (ĉ_i − c_i) ||²`, the product of the
frame's SENSITIVITY and the model's coefficient ERROR there, and C1 measured only the product.

- Large `J_i` — the frame is sensitive, equal coefficient error buys more scalar error there, and a
  loss built on one `M` averaged over 300 probe frames charges it the same as everywhere else.
  That is a mis-weighting, and re-weighting fixes it.
- Large `(ĉ_i − c_i)` — the frame is simply harder. Up-weighting it then trades the other 90% away
  for nothing.

The two are indistinguishable in C1's output and call for opposite work. This measures `J_i`
per frame with the same central differences and the same probe step the training loss is built
from, joins it to the costs C1 already wrote out, and reports the mean cost by quintile of
sensitivity. **A strong positive relation confirms B6's mechanism; a flat one kills it before a
single training run.**

## C6. The memory the sequence model learned — *2026-08-15*

Already specified under B5 ("How far back does it actually reach?"): the histogram of `tau` over the
256 update-gate channels, cross-checked by perturbing frame `t − k`. Listed here so the diagnostics
are in one place.

---

## Suggested order

**Rewritten 2026-08-15, after C1 and C2 ran.** They cost an afternoon and between them they closed
one entry, promoted another, and replaced the budget table at the top of this file with a measured
one. That is the third and fourth time a free diagnostic has settled something a run would have
spent hours on, after the polarity split (A12 step 0) and the scalar bias and slope (A7 step 0).

The order the two measurements imply:

1. **A23, spending the confidence** — weights that vary with position or with disagreement, and
   shrinkage where the model is unsure. **Neither needs a training run at all**, and both are
   pointed at by numbers already measured: the recurrence is worth 2.4% of the cost at the start of
   a shot and 14.5% at the eighth decile while carrying one flat weight, and seed disagreement
   predicts the per-frame cost at Spearman +0.565.
2. **A22, the last tenth of a shot.** 27.2% of everything reachable, in 10% of the frames, and the
   explanation that would have killed it is already refuted. Its own next step is A19's features
   read on that decile alone rather than on the composite.
3. **A13, stop on the composite.** The model's ψ is at its ceiling and its geometry is not, which
   is the same disagreement between loss and score in a different place.
4. **C3, permutation importance** — still free, and it prices A19 and B10 before either is built.
5. Then the zero-retrain sweep A7; then A1, A19, A3, A4, A5, A6, A8.

**Demoted or closed by the same measurements:** A2, measured flat; B6, whose weighting is aimed at
the frames the model already predicts best; B9, whose premise C2 refuted; and anything aimed at
R²ψ or the triangularities.

**Not first any more:** B9, whose premise C2 refuted, and anything aimed at R²ψ, which is measured
at 0.9998 of a 1.0000 ceiling and is finished.

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
