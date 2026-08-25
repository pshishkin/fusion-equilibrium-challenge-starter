# MAST / Challenge 2 — what we tested, and what came of it

The DIII-D half of this fork keeps its own record in
[experiments_history.md](experiments_history.md); this file is Challenge 2 only. The split is by
machine because almost nothing crosses it — that is itself one of the measured results below.

The rule is the same one AGENTS.md sets for the other file: an idea lives in
[ideas_mast.md](ideas_mast.md) until it has a number, then it moves here with a verdict, whichever
way it went.

## The leaderboard, which is the only unbiased instrument we have

Three demo shots is all the local truth there is, and they are consecutive numbers from one
campaign. The board scores 1206 shots and is **noiseless** for MAST — the solve is deterministic
and the fold is fixed — so a board difference of any size is real, unlike Challenge 1 where 0.0013
is the resolution of a paired run.

| # | what changed | S | R²ψ | R²qb | D_LCFS | Consistency |
|---|---|---|---|---|---|---|
| — | `predict_row` raised `NotImplementedError` | 0.0000 | — | — | — | — |
| 1 | the free-boundary Grad-Shafranov solve | 0.5566 | 0.8895 | 0.0574 | 0.4121 | 0.0 |
| 2 | Thomson-driven pressure source | 0.5170 | 0.8149 | — | 0.5155 | 0.0 |
| 3 | `edge = 0.03`, single calibration | 0.564865 | 0.881124 | 0.18299 | 0.472011 | 0.0 |
| 4 | + the three-member leave-one-out ensemble | 0.565247 | 0.880696 | 0.197091 | 0.486991 | 0.0 |
| 5 | + MAST's real first wall | 0.56777 | — | — | — | 0.0 |

**What entries 3, 4 and 5 decompose into** — they were built as a deliberate ladder so that each
step could be priced on its own, and the prices are very uneven:

- `edge = 0.03` (letting the source run 3% of the flux span past the last closed surface):
  **+0.0083**, and the largest single gain Challenge 2 has had after the solve itself.
- the leave-one-out ensemble: **+0.00038**. Locally it looked worth +0.0013, and locally two of
  its three members had seen each demo shot. On the fold none of them has seen anything, and the
  variance reduction is a quarter of what three shots advertised.
- MAST's real first wall, taken from the public archive: **+0.00252**, i.e. **six times the
  ensemble**, for a change that costs nothing at inference and is a fact rather than a fit.

The ensemble also traded terms rather than lifting them together: R²qb 0.183 → 0.197 while D_LCFS
went 0.472 → 0.487, which is worse. It survives on the net, barely.

## The Consistency term has never once been non-zero, and it is 20% of the composite

Every board entry above reads **Consistency 0.0**, and entries 3 and 4 report why: all seven
pooled R² are negative, so all seven floor at zero.

| scalar | entry 3 | entry 4 |
|---|---|---|
| R_axis | −10.69 | −10.13 |
| Z_axis | −26.27 | −27.30 |
| kappa | −0.52 | −0.66 |
| tri_top | −3.94 | −4.26 |
| tri_bot | −2.78 | −2.92 |
| volume | −7.21 | −6.85 |
| li | −0.40 | −0.35 |

This is the largest unclaimed pool in Challenge 2 by a wide margin: **0.20 of S, entirely
unrealised**, against the 0.15 that R²qb carries and the 0.10 in D_LCFS. Two of the seven — kappa
and li — sit within one decimal of zero and would start paying immediately; Z_axis at −27 says the
magnetic axis is not merely imprecise but systematically displaced, which `calibration.py` already
measures independently as a **−0.036 ± 0.026 m** bias, over half a grid cell.

Note the ordering trap: a scalar at −0.35 and a scalar at −27 are worth the same to the composite
today (both floor to 0), but they are nowhere near equally far from paying.

## The scorer was grading MAST on DIII-D's geometry — 2026-08-18

`local_score.py` had `MACHINE = "DIII-D"` hard-coded and loaded `d3d_envelope.npz` regardless of
the shots it was handed, so every MAST number this fork produced before 2026-08-18 was scored on
the wrong grid, the wrong envelope and the wrong axis sign. It surfaced because
`diagnose_boundary.py` computed D_LCFS 0.3317 where the scorer claimed 0.0996.

**What it invalidated:** D_LCFS and Consistency are 30% of the composite, and for three days both
were meaningless, so every sweep in that window optimised R²ψ alone and the profile constants were
chosen against a metric that could not see two of its four terms. The fix reads the machine off
the shots and validates it; `--mode perfect` now returns exactly 1.0000 on both machines, which is
the check that should have existed from the start.

The `edge` sweep was re-run against the corrected composite immediately afterwards and the optimum
had **moved** — which is why it was re-swept rather than assumed.

## q95 had an extra power of R, and the toroidal field is what caught it

`q = (F/2π) ∮ dl / (R|∇ψ|)` — a **single** power of R, because `B_p = |∇ψ|/R`. The first
implementation squared it. Nothing in the score said so; what said so was the fitted TF constant
coming out at **0.00117 T·m/kA**, implying a toroidal field four times below the 0.4–0.55 T MAST
actually runs at. After the fix the constant landed at **0.003668**, i.e. B0 = 0.366 T at an 85 kA
feed, and the correlation between the contour integral and the truth went **−0.29 → +0.43**.

Keeping the TF constant separable from the affine is what made this checkable, and it is worth
preserving for that reason alone: fitting both at once sent it to 0.0096 (a 0.96 T field) and the
number stopped being a check on anything.

## Scalars transfer, flux maps do not — the rule the board established

Between entry 1 and entry 5 of the table: **R²qb went 0.0574 → 0.2157 while R²ψ went 0.8895 →
0.8805.** The scalar calibration kept 91% of its local gain on 1206 unseen shots. Every change
aimed at the flux map lost on the fold what it gained on three shots.

Entry 2 is the clean demonstration. Driving the pressure term from Thomson raised local R²ψ from
0.9158 to 0.9370 and was submitted; on the fold it went **0.8895 → 0.8149** with D_LCFS 0.4121 →
0.5155, worth **−0.0514 of S**, while its scalar half was *positive* at +0.0118. A global
`p_scale` tuned on three NBI-heated discharges is not a property of the machine, and Thomson's own
liveness varies across the fold in a way three shots cannot show. The mechanism stays implemented
(`p_scale > 0` switches it on) because what failed was fitting its one constant on three shots.

## Leave-one-out says MAST's flux calibration does NOT overfit — and DIII-D's does

Refit everything on two demo shots, score the third through the real scorer:

- **flux map: held-out R²ψ 0.9328 ± 0.0020 against 0.9316 in sample.** An in-sample bias of
  −0.0012, i.e. none. The reason is identifiable: 812 conductor rows give the solenoid a shape the
  fit can separate from a constant.
- **R²qb is the term that overfits**: 0.2771 in sample against 0.1663 held out.

DIII-D asked the same question, where shots are plentiful, answers **−0.601 ± 0.120**. So the MAST
result is a property of this dataset, not a general licence.

Shrinking the affines toward their means was tried and **refuted**: the composite floors R²qb at
zero *after* averaging q95 and βN, so shrinkage gives up real gain to insure against a downside
the floor already covers.

## DIII-D cannot be used as a proxy for MAST — 2026-08-19

The idea was to solve DIII-D mathematically, parametrise how reality departs from the mathematics,
cross-validate that parametrisation on DIII-D where shots are plentiful, and carry the winner to
MAST. The physics does transfer; the *fit* does not, for two separable reasons.

1. **The equilibrium slides vertically.** On DIII-D the solved axis wanders a metre in Z. Pinning
   it fixes the solve — and the pin is a DIII-D construct (`z_pin=-0.0172`, `z_band=0.15`) that
   has no MAST counterpart, so the thing that makes DIII-D work is exactly the thing that cannot
   be carried.
2. **The real blocker is the per-frame flux gauge**, not the shape. Removing a single per-frame
   constant takes held-out R²ψ from **0.207 to 0.711**. The offset is ±0.147 Wb/rad, **10.5% of
   the range**, against MAST's 1.6% — because ECOILA is degenerate with a constant on DIII-D.

Held-out R²ψ by calibration size is negative at **every** size tried: −6.58 ± 8.99 (1 shot),
−0.601 ± 0.120 (3), −0.253 (6), −0.235 (12), **−0.148 (24)**. It is improving with data and has
not crossed zero by 24 shots. Task #32 is what would have to be closed first, and the flux gauge
is the thing to close.

## The public MAST archive: what was taken, and what was deliberately not

The STFC Echo zarr archive is public and the competition terms permit external public datasets
**with disclosure in the methods report**. The organisers were asked directly.

**It is the withheld ground truth, bit for bit.** `efm/psirz` against the competition's
`efit_psirz` differs by exactly **0.000e+00**. That makes it the one external source where the
rule against de-anonymising or memorising the hidden truth has teeth, and the line taken is
narrow: **only the machine's first-wall geometry was read** — 37 vertices, byte-identical across
shots M5 through M9, a fixed property of the tokamak that changes nothing about any discharge.
Worth **+0.00252** on the board. The disclosure is recorded in three places, and
`solver/mast_archive.py` carries it in its docstring beside the reader.

(Reader detail worth not re-deriving: the archive's A record black-holes; `mast_archive.py` pins
DNS to 130.246.179.110. It reads zarr v2 over HTTP with no `zarr` dependency.)

## The scalar models: ~90 candidates, and the gate that did the selecting — 2026-08-19

The shipped affines read one contour integral through two free numbers. Measured honestly — refit
on two demo shots, scored on the third — they are worth **+0.027 (q95) and +0.047 (βN)** held out,
against +0.384 and +0.165 in sample. The board agrees they are weak: R²qb 0.197 at entry 4.

`mast/bench.py` was built to make this cheap: solve the 115 demo frames once (25 s), cache the
per-frame quantities, and evaluate a candidate scalar model in milliseconds against a
leave-one-shot-out gate. About ninety candidates went through it.

**The held-out gate alone was not enough, and that is the transferable lesson.** A twelve-feature
ridge scored **+0.972 held out** for q95 — better than anything shipped — and was thrown away,
because a second test caught it: refit on each pair of shots, do the coefficients hold still? Its
coefficient on the TF current moved 18 → 30 and its coefficient on kappa tripled between folds. A
model reading noise can still cross-validate cleanly on three sibling shots. Coefficient stability
is the cheap second opinion, and it should be applied to every fit this fork makes on three shots.

What survived both tests:

| | model | held out | per fold | worst coefficient spread |
|---|---|---|---|---|
| q95 | `exp(c + w · log[ \|I_TF\|·q_shape(0.99), volume, Ip ])` | **+0.967** | .96 .97 .97 | 31% (8% on the two carrying the weight) |
| βN | `c + w · [volume, 1/Ip, minor radius]` | **+0.921** | .92 .92 .92 | 16% |

Pooled, **R²qb goes +0.037 → +0.944 held out**. Through the real scorer on the demo shots, S goes
**0.6358 → 0.7200** with R²qb 0.8845 (in-sample on the scalars; the honest estimate is the 0.944).

Two findings inside that are worth more than the numbers:

- **q95 comes out as a^1.15 · Ip^−0.41 × the contour integral.** The cylindrical estimate is
  a² Ip⁻¹, so the solve recovers the *shape* of the scaling and the fit is correcting the
  amplitude of our ψ — it is not inventing the relation.
- **βN is better off without the Thomson term it was built on.** `200 μ0 <p> a / (B0 Ip)` makes
  the model worse (+0.786 with it, +0.864 without), and plasma volume alone beats it four times
  over. Volume correlates −0.90 with βN and does so at −0.91 / −0.89 / −0.90 *within each shot
  separately*, so it is not an artefact of pooling a single campaign. Our Thomson pressure is too
  noisy to carry βN; the geometry the solve gets right carries it instead.

**The bug this shipped with, for one run.** The bench feature is `|I_TF| · q_shape`, and the first
wiring passed the bare contour integral. In a log-space model that is a constant offset — the
model's `centre` described a quantity 85× larger than the one being fed in — and it scaled every
q95 down by 2.6, giving **R²qb = −3.43** end to end where the bench said +0.94. βN, being linear
and not using that feature, was unaffected and matched the bench exactly at +0.925, which is what
localised it. The lesson is narrow and repeatable: **when a bench and the shipped path disagree,
compare the features, not the predictions.**

A related trap in the same change: the fallback for a failed contour was set to the model's
log-space `const`, which is a q95 of 2.0 on a machine that runs at 8.

## The flux span is frozen, and it is the root of the Consistency failure — 2026-08-19

`mast/diagnose_cons.py` asks of each of the seven derived scalars whether our error is an offset,
a scale, or the wrong sign. **Five of the seven are the wrong sign**, and it holds inside every
demo shot separately, so it is not a per-shot offset:

| scalar | pooled corr | within-shot | our sd / truth sd | R² if a per-scalar affine were allowed |
|---|---|---|---|---|
| R_axis | **−0.746** | −0.82 −0.71 −0.73 | 2.58 | 0.556 |
| Z_axis | −0.463 | −0.41 −0.39 −0.50 | 31.9 | 0.215 |
| kappa | −0.378 | −0.34 −0.44 −0.35 | 0.72 | 0.136 |
| tri_top | **−0.741** | −0.78 −0.80 −0.82 | 1.63 | 0.565 |
| tri_bot | −0.652 | −0.70 −0.73 −0.65 | 1.49 | 0.484 |
| volume | +0.292 | +0.29 +0.34 +0.24 | **5.59** | 0.089 |
| li | **+0.824** | +0.81 +0.83 +0.85 | 0.47 | 0.647 |

Against the discharge current the split is explicit: the truth's R_axis rises with Ip at **+0.79**
while ours **falls at −0.47**. Only li moves the right way — and li is the one quantity of the
seven that reads the current profile rather than the boundary.

**Where it comes from.** Our boundary FLUX is nearly perfect: `corr(psi_bnd, truth) = +0.997`
on all three shots. What is wrong is the span.

| | frame 0 | mid | last |
|---|---|---|---|
| truth `psi_axis − psi_bnd` | 0.0364 | 0.0626 | 0.0887 |
| ours | 0.0526 | 0.0526 | 0.0518 |

**The truth's flux span grows 2.4× through a discharge whose Ip is flat; ours does not move.**
`corr(span, truth span)` is **−0.75 / −0.66 / −0.55** — the same anti-correlation, at the same
strength, as R_axis. The cause is structural: the profile shape is one constant for every frame
(`alpha = gamma = 1`) and the source normalisation is pinned to Ip, so at flat-top our span cannot
change. The real discharge peaks its current profile — li rises — and the span rises with it.

**The lever exists.** Sweeping `gamma` on one shot moves the mean span 0.039 → 0.215 (5.5×), and
the sign of the span correlation flips somewhere between gamma 0.5 (**+0.70**) and gamma 1.0
(−0.75). No single gamma can be right, because one number cannot track a quantity that moves.

**And the target is observable.** The truth's span is predictable from raw inputs, leave-one-shot-out:

| inputs | held-out R² |
|---|---|
| solenoid current alone | **0.9645** |
| + Ip | 0.9698 |
| + P4, P5 | 0.9815 |
| all coils | 0.9843 |

`corr(truth span, solenoid current) = −0.983`, and −0.98 / −0.99 / −0.98 within each shot. That is
not a fit, it is a restatement of what a solenoid does: it drives the flux swing, and the plasma's
poloidal flux span is the flux it has delivered. So the per-frame profile can be driven by an
input we always have, with no circularity and nothing withheld.

### The oracle says the mechanism is right, and prices it

Per-frame `gamma` bisected until the span hits the TRUTH's span — not a model, the ceiling:

| | S | R²ψ | R²qb | D_LCFS | Consistency |
|---|---|---|---|---|---|
| shipped, gamma frozen | 0.72001 | 0.93600 | +0.8845 | 0.4125 | 0.06891 |
| ORACLE, per-frame gamma | 0.71395 | **0.95997** | +0.3889 | **0.3173** | **0.29684** |

Consistency **quadruples**, and the per-scalar detail is the confirmation that this is the right
mechanism rather than a lucky knob:

| | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
|---|---|---|---|---|---|---|---|
| shipped | −10.22 | −4894 | −1.52 | −8.05 | −7.38 | −26.71 | +0.48 |
| oracle | **+0.67** | −8139 | **+0.48** | −2.84 | −7.27 | **−2.27** | **+0.92** |

R_axis goes from −10 to POSITIVE; kappa crosses zero; volume improves twelvefold; li reaches
+0.92. Three of the seven now pay. R²ψ rises too (0.936 → 0.960) and D_LCFS falls — every term
that reads the map improves at once, which is what a real physics fix looks like and what a
metric-chasing knob does not.

**The composite still goes DOWN, and the reason is bookkeeping, not physics.** R²qb collapses
0.8845 → 0.3889 because `scalar_model.py`'s q95 and betaN were fitted against the frozen-span
solve — their features (volume, minor radius, the contour integral) all move when the span moves.
Refitting them on the new equilibria is required, not optional. With R²qb merely restored to its
current 0.8845 the composite would read

    0.55(0.95997) + 0.15(0.8845) + 0.10(1 − 0.3173) + 0.20(0.29684) = **0.788**

against 0.720 — **+0.068 of S**, an order of magnitude larger than anything else measured on this
challenge. Z_axis remains hopeless and remains worth nothing (see the note in
`diagnose_cons.py`'s docstring).

### The shipped version, and what it cost to make it work

`mast/span.py` predicts the target span from the solenoid current and Ip; `predict._solve_frame`
bisects `gamma` in log space until the solve lands on it. Held-out span R2 **+0.9698** against
+0.9720 in sample, and the solenoid coefficient holds to **0.3%** across the three fits.

| | S | R²ψ | R²qb | D_LCFS | Consistency |
|---|---|---|---|---|---|
| before | 0.72001 | 0.93600 | +0.8845 | 0.4125 | 0.06891 |
| per-frame profile, scalars NOT refitted | 0.71260 | 0.96017 | +0.4050 | 0.3156 | 0.27657 |
| **per-frame profile, scalars refitted** | **0.78043** | 0.96017 | +0.8572 | 0.3156 | 0.27657 |

**+0.0604 of S**, and the middle row is the warning worth keeping: a solver change invalidates
every scalar model downstream of it, because `q95` and `betaN` read the equilibrium's own volume,
minor radius and contour integral. Refitting is part of the change, not a follow-up — skipping it
made a +0.06 improvement look like a −0.007 regression.

**The refit re-ran the same double gate, and the stability half did the work again.** Every q95
candidate that used `kappa` scored higher held out (0.951 against 0.864) and every one of them was
thrown away: coefficient spread across folds past **1200%** with a sign flip. The survivor,
`log(|I_TF|·q_shape(0.95), a, Ip)`, reads the surface q95 is physically defined on and holds its
coefficients to 15.5% — the physical choice and the stable one turned out to be the same. betaN
became `volume + 1/Ip + kappa` at +0.866 held out, 19.4% spread.

Cost: seven solves a frame instead of one, so ~63 ms becomes ~440 ms and the 1206-shot split goes
from 12 minutes to about 80 on 28 processes.

### How far the span model extrapolates on the real fold

Measured on 60 random `mast_public_test` shots (3524 frames) through the real inference path:

| | demo shots | fold | frames outside |
|---|---|---|---|
| solenoid current | −39.6 .. 18.7 kA | −46.9 .. 37.5 kA | 16.8% |
| Ip | 0.465 .. 0.647 MA | 0.110 .. 1.012 MA | **61.4%** |
| resulting span target | truth 0.034 .. 0.100 | 0.022 .. 0.121 | 11.1% |

The Ip extrapolation is large in frames but small in effect, because Ip carries the weaker of the
two coefficients (0.3% spread on the solenoid against 12% on Ip). The predicted spans stay
physically plausible everywhere and **the [0.010, 0.250] clip never binds on a single frame** — so
the clip is doing its job as a net rather than as the model.

(Read the earlier version of this table with suspicion if it appears anywhere: a first attempt at
it re-scaled the raw current column that `shot.read` already delivers in kA, and reported the whole
fold as zeros. The pipeline was right and the diagnostic was wrong.)

## `updown` was never swept, and it is worth more than the ensemble — 2026-08-19

One parameter, already plumbed since the gain fit ties each up/down coil pair to one number and so
structurally cannot represent a vertical field error. Swept against the real composite with
`mast/sweep.py`:

| updown | 0.000 | −0.005 | −0.010 | −0.015 | −0.020 | −0.030 | −0.060 |
|---|---|---|---|---|---|---|---|
| S (pooled) | 0.72001 | 0.72177 | **0.72440** | 0.72279 | 0.71815 | 0.71841 | 0.66707 |

A clean unimodal curve peaking at −0.010, worth **+0.0044** — most of it through D_LCFS, which
falls 0.4125 → 0.3411. Positive values are much worse (+0.060 costs −0.121), which is itself
evidence of a real optimum rather than a noise plateau.

**The per-shot gate splits it, and that is the part worth keeping.** Scored one demo shot at a
time:

| updown | 28348 | 28350 | 28351 |
|---|---|---|---|
| −0.005 | **+0.00244** | **+0.00094** | **+0.00223** |
| −0.010 | −0.00386 | +0.01094 | +0.00466 |
| −0.015 | −0.00638 | +0.01087 | +0.00223 |

The DIRECTION is unanimous — every shot wants a negative tilt — but the magnitude is not: 28348
is hurt at −0.010 while 28350 gains ten times as much there. **−0.005 is the only value all three
agree on.** Pooled S ranks −0.010 first because 28350's gain dominates the pooled sums, which is
exactly the failure mode the per-shot view exists to expose.

## Measurement facts worth not re-deriving

- `AXIS_SIGN` is **+1** on MAST (the axis is a flux maximum) and **−1** on DIII-D.
- MAST's inner limiter radius is **0.19524440169334412 m**; the first wall is 37 vertices.
- `βN = 200 μ0 <p> a / (B0 Ip)`, with `<p>` volume-averaged from the midplane Thomson.
- The MAST solve is **pure CPU** — numpy plus one sparse LU factorised once. No torch, no CUDA
  anywhere in `mast/` or `solver/`. It is ~63 ms a frame on one core, so the 1206-shot split is
  12 minutes across 28 processes against 1.1 hours serially, and `mast/predict.py` pins each
  worker to a single BLAS thread, which is what makes a pool of 28 an actual 28×.
- The coil gains check themselves: P4, P5 and P6 — the families whose conductor geometry the
  dataset ships in full — fit to **1.010, 0.961, 0.994**, confirming the shipped column current is
  the coil's total ampere-turns. P3 (8.69) and the solenoid (169.8) are the two the dataset
  under-describes, and are what the calibration is actually for. P2 at 0.564 is the one number
  that does not fit the story and is absorbing a discretisation error: its pair sits where the
  grid is 0.030 m coarse against a 0.06 m coil.
- 0.12% of frames have no closed surface (4 of 3243 over 60 shots), all in the current ramp-down
  below 245 kA. The coil field alone is not a fallback there, it IS the answer.
