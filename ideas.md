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

# Group A — one to a few production runs

## A1. Temporal smoothing of the predicted outputs — *2026-08-13*

**Hypothesis.** Low-pass filtering the predicted 52-vector along each shot's time axis raises
Consistency, `1 − D_LCFS` and R²qb, and some window in 40–150 ms beats no smoothing by ≥ 0.002.

**Mechanism.** We predict every frame independently from instantaneous currents, so the predicted
coefficients carry frame-to-frame jitter that an equilibrium evolving on the current-diffusion time
(hundreds of ms, against 20 ms between frames) forbids. The functionals read geometry — a sub-pixel
argmax in `o_point.py`, contour extremes in `derive._shape_from_contour` — and turn small map
jitter into large scalar jitter, which is pure SS_res in a pooled R². Filtering coefficients equals
filtering flux maps, the decoder being affine. **The model cannot do this for itself: it never sees
an adjacent frame.**

**Test.** A filter step on `P` in `predict_row`, knob in `params.yaml`; sweep {off, 3, 5, 7 frames}
× {order 0, 1} by **re-scoring the existing production artifact** — minutes per setting, no
retraining — then confirm the winner with one paired production run.

**Refuted if** the sweep's optimum is "off", or the predicted coefficients' power spectrum above
~5 Hz is already at the ground truth's own level. That spectrum is a 20-minute read-only diagnostic
and should come first.

**ΔS ≈ +0.002…+0.006, confidence 0.6.** Best expected gain per minute on the list.

## A2. Average several MLP seeds — *2026-08-13*

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

## A3. Flip `calibrate_scalars` on — *2026-08-13*

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

## A4. The MLP's capacity and schedule — *2026-08-13*

**Hypothesis.** At ~63k training frames, 21 → 256 → 256 → 52 with constant lr 1e-3 and no weight
decay is under-fitted, and one of {512×512, 256×256×256, cosine or step decay, wd 1e-5…1e-4} beats
the baseline by ≥ 0.002.

**Mechanism.** The nearest-neighbour ceiling says instantaneous currents determine the targets to
R² 0.97–0.997 while the model reaches 0.89–0.93 — the **regression** is the binding constraint, not
the features. This surface has never been swept once since the starter kit. The "the big model does
it for itself" trap does not apply here, because this *is* enlarging the model. Constant-lr Adam
also leaves the final iterate noisy, which feeds Consistency jitter directly.

**Test.** `hidden_sizes` plus a new schedule key in `params.yaml`; five or six production runs, ~1 h.

**Refuted if** no variant clears +0.002 paired — which would say 63k rows saturate this function
class, and only the Group B capacity ideas remain.

**ΔS ≈ +0.002…+0.005, confidence 0.6.**

## A5. More shots, and absorb the validation window — *2026-08-13*

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

## A6. CatBoost at production scale, back in the ensemble — *2026-08-13*

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

## A7. A dedicated q95/βN head — *2026-08-13*

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

## A8. Post-hoc de-biasing of the derived scalars — *2026-08-13*

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

## A9. More frames per shot at fixed shots — *2026-08-13*

**Hypothesis.** Production trains on every tenth frame (`0.45/0.1`); raising the frame share to 0.2
or 0.3 at unchanged shots gains ≥ 0.001.

**Mechanism.** The 08-12 result — more shots and fewer frames beats the reverse at a fixed row
budget — set the ratio, but it never said the frame axis is worthless, only that it is the cheaper
one to spend. Between-shot and within-shot variance split roughly 53/47, so frames are
near-duplicates but not duplicates. This is the one axis of the split we have never varied on its
own at production.

**Test.** One run each at `0.45/0.2` and `0.45/0.3`, paired against `0.45/0.1`. Needs A5's
subsampled PCA first — `0.45/0.2` is exactly what drove the machine into swap.

**Refuted if** the paired gain is ≤ +0.0005, in which case frames are settled and only shots matter.

**ΔS ≈ +0.0005…+0.002, confidence 0.4.**

## A10. Auxiliary supervision on the scored functionals — *2026-08-13*

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

## A11. Eddy-current proxy features — *2026-08-13*

**Hypothesis.** Appending high-passed coil currents (vessel L/R timescales, ~2–20 ms) and `dIp/dt`
to the 21 features gains ≥ 0.001.

**Mechanism.** Vessel eddy currents produce real poloidal flux and are structurally absent from the
shipped signals — `coil_field` cannot compute them, because the vessel is not shipped — and the
high-passed coil currents are their driving term. `dI_sol/dt` is the loop voltage and correlates
−0.665 with `dIp/dt` against +0.515 for the level, so the derivative is the informative quantity.

This is the **only** feature idea aimed at information the nearest-neighbour ceiling cannot contain:
that ceiling was measured on instantaneous currents, so it bounds only what instantaneous currents
determine, and the missing 0.03 on `li` could be exactly this. Every other feature idea stays
weighed down by it.

**Test.** Extend `features_for_row` with two or three filtered copies per signal, filtered before
interpolation onto `efit_times`, identically at training and inference. One run per τ.

**Refuted if** ΔS ≤ 0.001 paired, or the gain appears only on scalars already at their ceiling.

**ΔS ≈ 0…+0.0015, confidence 0.25.**

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

---

# Group B — multiplies the compute

Tested on their own, not inside the ordinary iteration loop. Each says roughly how much more it
costs.

## B1. A real hyper-parameter search — *2026-08-13* — ~40 runs, overnight

**Hypothesis.** A 30–60-trial search over width, depth, learning rate and schedule, weight decay,
batch size and patience beats A4's hand sweep by a further ≥ 0.002.

**Mechanism.** Same as A4 — the regression is the binding constraint and this surface is unexplored.
σ = 0.0013 makes ~0.002 effects rankable, and pairing trials within one salt keeps it clean.

**Test.** Optuna or a plain grid driving `train_eval.py` at production shares, one fixed salt, the
best three re-validated on a second. ~6 h unattended on this machine.

**Refuted if** the incumbent is within noise of the search optimum. **ΔS ≈ +0.002…+0.004 beyond A4,
confidence 0.6.**

## B2. Scale the ensemble to 8–10 diverse members — *2026-08-13* — ×10 training, ~2 h

**Hypothesis.** Seeds × widths × CatBoost (if A6 survives) adds a further ≥ 0.001 over A2.

**Mechanism.** Variance reduction saturates as 1/N in the independent component, and architectural
diversity keeps the components independent longer than seeds alone do. Inference cost at submission
— 874 shots × 10 small nets — stays trivial.

**Test.** Grow `params.yaml` after A2, A4 and A6 report; one long run.

**Refuted if** members 5–10 add under 0.0005 over the four-seed ensemble. **ΔS ≈ +0.001…+0.002 on
top of A2, confidence 0.7.**

## B3. All the data — *2026-08-13* — ×3–5 in memory work and time

**Hypothesis.** After A5, going to ~5600 shots with a 0.2–0.4 frame share gains a further ≥ 0.001.

**Mechanism.** Extrapolation of the shots curve, plus whatever A9 finds on the frame axis.

**Test.** float32 targets end to end, subsampled PCA, a chunked metric transform — the RAM ceiling
is the real engineering, roughly half a day — then one run per point. A preprocessed cache of ψ
plus features would be ~18 GB against 88 GB of parquet and would remove the decode cost entirely;
profile the run first to see whether that is where the time actually goes.

**Refuted if** A5's paired gain is already ≤ +0.0005 — that closes the whole axis. **ΔS ≈
+0.001…+0.003, confidence 0.5.**

## B4. Zeroth-order calibration against the exact scorer — *2026-08-13* — ~200 scoring passes, 4–6 h

**Hypothesis.** Optimising a low-dimensional post-hoc correction (per-scalar gains, smoothing
window, an inflation factor — 10 to 20 parameters in all) directly against true S on validation
shots beats the surrogate-tuned versions of A1 and A8 by ≥ 0.001.

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
prediction plus post-hoc smoothing by ≥ 0.002.

**Mechanism.** A1's ceiling is that of a fixed filter. A sequence model learns state — integrating
exactly the eddy-current and current-diffusion dynamics that instantaneous features miss — and
picks the per-component bandwidth jointly with the regression. This is the structural fix for the
single largest line in the budget.

**Test.** A new `TargetModel` consuming per-shot batches, which is a loader rework: the pipeline
concatenates frames today, and the training unit would become the shot (3168 of them, not 63k).

**Gate.** Run A1 first. If a fixed filter buys nothing, this buys little. **ΔS ≈ +0.002…+0.006,
confidence 0.4 — the highest ceiling in Group B.**

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
subspace and a damping sweep, testable inside A8's harness. The per-frame masked version only if
that shows a sign.

**Refuted if** the damping sweep's optimum is zero. **ΔS ≈ 0…+0.002, confidence 0.25.**

## B8. A boundary term that is an honest maximum — *2026-08-13* — ~half a day

**Hypothesis.** Fine-tuning the MLP with an auxiliary log-sum-exp (soft-Hausdorff) penalty over
per-frame boundary displacements `δψ/|∇ψ|` improves `1 − D_LCFS` by ≥ 0.001 without paying elsewhere.

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

Free diagnostics in one afternoon: the polarity split (A12 step 0), the scalar bias and slope
(A8 step 0), the jitter spectrum (A1 step 0). Then the zero-retrain sweeps A1 and A8 on the
artifact we already have; then A2 and A3, one run each; then A4, A5, A6, A7, A9.

Group B only after A1, A2 and A4 report — B1, B2 and B5 are their scaled versions, and B4 needs the
harness A1 and A8 build.

**Dropped, with reasons.** Anything touching `n_pca` (saturated at production, measured both ways).
The quadratic boundary term (refuted; B8 is its only honest descendant). Per-component
standardization and ridge-in-the-ensemble (refuted). Polarity *normalization* (refuted; A12 is a
different operation and gated on a free diagnostic). Repairing frames where extraction fails
(refuted — both failure rates are 0.0% at production). Thomson (the correlation with the failing
shots was measured at ≈ 0, and βN's ceiling from currents alone is 0.997). A CNN or pixel decoder
(the coefficient bottleneck is saturated: Consistency stops improving at 50 components, so capacity
belongs in the regression — A4, B1, B5 — not in the decoder).
