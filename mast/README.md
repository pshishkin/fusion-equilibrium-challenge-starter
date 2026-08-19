# Challenge 2 — MAST, zero-shot

**Where this stands: `S_MAST = 0.5566` on the leaderboard, `G_ratio = 0.560`, from a standing
start of zero** (submitted 2026-08-18, all 1206 `mast_public_test` shots). The best Challenge 2
entry on the board is 0.7249 / 0.729, so this is second-order right and not yet competitive:

| term | weight | demo-local | **leaderboard** | leader | points to the leader |
|---|---|---|---|---|---|
| R²ψ | 0.55 | 0.9165 | **0.8895** | 0.9732 | +0.0460 |
| R²{q95, βN} | 0.15 | 0.1239 | **0.0574** | 0.5079 | **+0.0676** |
| 1 − D_LCFS | 0.10 | 0.6664 | **0.5879** | 0.7891 | +0.0201 |
| Consistency | 0.20 | 0.0185 | **0.0000** | 0.1727 | +0.0345 |
| **S** | | 0.5941 | **0.5566** | 0.7249 | +0.1682 |

### Leave-one-out, and what it licenses

Every MAST number here is measured on the three demo shots the calibration was fitted on, so the
size of the in-sample bias was unknown until `mast/loo.py`: refit everything on two shots, score the
third, rotate.

| | S | R²ψ | R²qb | 1 − D_LCFS | Consistency |
|---|---|---|---|---|---|
| **held out** | 0.6166 | **0.9328 ± 0.0020** | 0.1663 | 0.5838 | 0.0695 |
| in sample | 0.6275 | 0.9316 ± 0.0014 | 0.2771 | 0.6051 | 0.0652 |

**The flux map does not overfit three shots at all** — an in-sample bias of **−0.0012** on R²ψ, with
the gains barely moving between folds (P4 1.020 / 1.028 / 1.022). DIII-D, asked the same question
where shots are plentiful, gives **−0.601 ± 0.120**. The difference is what each dataset ships: 812
conductor rows give MAST's solenoid a shape the fit can separate from a constant, where DIII-D's
single ECOILA rectangle cannot be told from an offset.

The composite's bias is **−0.011** and almost all of it is one term, R²qb — four numbers on 115
correlated frames. Shrinking those affines toward the constant was tried and **refuted**: held-out
R²qb peaks at keep = 0.75 (0.1756) while held-out S falls monotonically (0.6166 → 0.5853), because
the scorer floors `max(0, R²qb)` before weighting it and one fold is negative either way.

So the three demo shots are a sound calibration set for the map and a poor one for the two value
scalars — and the reason three MAST decisions were refuted by the board after looking good locally
is **not** in-sample bias. It is that these shots are a narrow sample of discharge *types*, which
leave-one-out on siblings cannot see.

**The three leave-one-out fits are also an ensemble**, and not the near-duplicates their overlap
suggests: the spread across the badly conditioned coils reaches **23.6%** of the mean. Averaged as
decoded flux maps they score **S 0.6310 against 0.6296** for the single fit (R²ψ 0.9317 → 0.9348).
Read that as optimistic — on each demo shot two of the three members had seen it — but on the
1206-shot fold none of them has seen anything, which is the case it is actually for.

### Six probes are built and unsubmitted

The board is a **noiseless instrument** here — the solver is deterministic and the fold is fixed, so
any difference between two entries is exact signal, unlike DIII-D where the seed sigma is 0.0009.
Four of the 100 submissions have been spent; these are the next four, in the order worth sending:

| zip in `submissions/` | what changes | local S |
|---|---|---|
| `..._20260819T091150Z.zip` | + MAST's real first wall, calibrations refitted under it | **0.6359** |
| `..._20260819T065915Z.zip` | the three leave-one-out calibrations averaged | 0.6310 |
| `..._20260818T224648Z.zip` | `edge = 0.03`, gains and affines both refitted for it | 0.6296 |
| `..._20260818T220944Z.zip` | `edge = 0.02`, gains refitted for it | 0.6148 |
| `..._20260818T220214Z.zip` | `edge = 0.02`, gains held at the 0.5566 entry's | 0.6105 |
| `..._20260818T194742Z.zip` | `beta = 0.15` (shipped is 0.30) | ~0.594 |
| `..._20260818T195614Z.zip` | `beta = 0.45` | ~0.594 |

The shipped configuration scored **0.5941** locally and **0.5566** on the board; `mast/` now holds
the top row. The first three are one increment each — `edge`, then the gain refit, then `edge` again
at its moved optimum — so the board can attribute what it sees. The last two put three points on the
β axis, which three demo shots cannot resolve at all: locally β 0.15 and 0.30 differ by 0.001.

The DIII-D file in each is byte-identical, so Challenge 1 does not move.

**The demo-local column is trustworthy as of 2026-08-18 and was not before.** `local_score.py`
hardcoded `MACHINE = "DIII-D"` and loaded `d3d_envelope.npz` unconditionally, so every MAST frame
this fork scored had its LCFS extracted on a grid running R = 0.84–2.54 instead of 0.06–2.00,
against DIII-D's vessel envelope, under `AXIS_SIGN = −1` instead of +1. The same three shots read
S 0.6289 with D_LCFS 0.0996 before the fix and S 0.5930 with D_LCFS 0.3336 after — against 0.4121
on the fold. R²ψ was never affected, being a pooled sum over pixels that touches no mask, sign or
contour.

Three things that table says, and they set the whole order of work below.

**Every term now reads local slightly better than fold**, which is what three shots against 1206
is supposed to look like: R²ψ 0.9165 → 0.8895, D_LCFS 0.3336 → 0.4121, Consistency 0.0185 → 0.0000.
Prediction-side LCFS failures are 0.48% and derivations 0.34%, so none of this is crashes — it is
ordinary imprecision, everywhere. What the corrected metric also settles is that the boundary is
genuinely poor and was never merely mis-scored.

**And what the boundary error IS, decomposed** (`mast/diagnose_boundary.py`, Hausdorff on the
correct grid): raw **0.3317**, with the plasma's position removed **0.2401**, with position and
size both removed **0.1126**. So it is roughly a third position, a third size, a third shape — our
contour is **0.929×** the true one on average and only **49.5%** of it lies within 0.10·rgeo of the
truth. The worst point sits at the outboard midplane (−45° to +45° on 81 of 115 frames), not in the
divertor legs, which is where a size error shows up first.

**The largest gap is the cheapest term.** R²{q95, βN} is 40% of the 0.168 deficit, and neither
scalar needs a flux map to compute — matching the leader there alone would give S = 0.624.

**The hardest gap is already measured as out of reach for this solver.** The leader's R²ψ of
0.9732 is *above* the 0.954 this profile family reaches with the TRUE vacuum field substituted in
(see below). So `(1 − ψ_N^α)^γ` with three global numbers is the binding constraint, and no amount
of coil calibration gets past it.

Two readings from the leader's own numbers that cost nothing. **R_axis and Z_axis are unreachable
for everyone**: theirs are −0.69 and −0.33, and the true Z_axis has a spread of 0.0012 m against a
grid step of 0.0625 — 2 of the 7 Consistency slots are a write-off, so the realistic ceiling on
that term is 5/7. And **their D_LCFS histogram is bimodal** — a peak at 0.10 plus a bump at 0.7–0.8
on about 6% of shots — where ours is a single peak at 0.30. That is the difference between being
right and breaking on one class of discharge, and being uniformly imprecise.

## What this is

There is no `mast_train` config — Challenge 2 is zero-shot by construction — so nothing here is
fitted to shots. What replaces the training set is the Grad-Shafranov equation:

```
Delta* psi = -mu0 R J_phi,     J_phi = R p'(psi) + F F'(psi) / (mu0 R)
```

solved as a free-boundary problem, per frame, with the coil field computed exactly from the
conductor rectangles the dataset ships on every row. Four files do the work:

| | |
|---|---|
| `greens.py` | the vacuum field: one flux map per current column, from elliptic integrals |
| `scalars.py` | q95 from a contour integral, βN from Thomson's midplane pressure |
| `gs.py` | the Picard solve, the boundary search, and the sparse `Delta*` factorisation |
| `shot.py` | one MAST row onto the EFIT frame clock |
| `predict.py` | what the submission calls: `predict_row(row) -> {psirz, q95, betaN}` |
| `calibration.py` | the eleven coil gains and three profile numbers, as literals |
| `calibrate.py` | how those numbers were obtained |

Nothing here imports `my_experiments/`, and nothing there imports this. The two meet only in
`machines.py` at the repository root, which dispatches on the row's own `source` column.

## The three demo shots are the whole local metric

`parquet_data/` ships three MAST shots that — unlike the 1206 in `mast_public_test` — carry
`efit_psirz` and every EFIT scalar. **115 frames of ground truth**, and they are the only MAST
truth in the released dataset. Everything below was measured on them, which is both why Challenge 2
could be developed at all without spending leaderboard submissions, and the main thing to distrust:
three shots of one machine on one day, against 1206.

**Read the local Consistency as a floor, not an estimate.** R² is pooled about the fold mean, and
three near-identical shots give the seven shape scalars almost no spread to be scored against —
`kappa` scores −46 here on an error of 0.13 against a spread of 0.02. On a 1206-shot fold the
denominator is a different quantity entirely. R²ψ and D_LCFS do not have that problem, because ψ
itself varies enormously within a shot.

## What was measured on the way, and what it settled

**The units and the sign convention, settled by one test.** Integrating `-Delta* psi / (mu0 R)`
over the grid cells inside the shipped LCFS returns **532 / 633 / 660 / 635 kA** on four frames
whose shipped `Ip` is **496 / 609 / 637 / 618 kA**. Agreement to 5–7% fixes at once that ψ is in
Wb/rad, that every MAST current column is in kA, and that `AXIS_SIGN["MAST"] = +1` means the stored
sign and the physical one agree. Doing this before writing the solver would have saved an hour: the
first attempt assumed the shipped ψ span was too small for a 600 kA plasma and went looking for a
missing factor that was not there.

**The elliptic solve reproduces free space to 1%.** A 600 kA Gaussian blob solved on the interior
with Green's-function Dirichlet values on the rim, read at five points away from the source against
the direct superposition: ratios 0.989 / 0.996 / 1.010 / 1.003 / 1.015. The rim treatment is the
part worth checking — truncating the domain with ψ = 0 at the grid edge would put an image current
just outside it.

**The boundary is at one of a handful of levels, not on a continuum**, and finding it needs
connectivity. `psi_bnd` is either the flux where the plasma reaches MAST's centre column or the
flux at an X-point, both readable off the grid — so the search is over candidates, with no
bisection and nothing lost between bisection steps. The centre column's radius is not a guess:
`efit_lcfs_r` bottoms out at **0.1963 m on all 115 demo frames**, to four decimals.

The connectivity test is what makes it work, and skipping it is the error worth recording. Taking
the maximum of the candidates without asking which are attached to the plasma puts the boundary
**19% of the axis-to-boundary span too high on the median frame** — because the centre column
carries the solenoid's own flux hill, which on a diverted frame sits above the true boundary while
belonging to no plasma at all. With the test: **median 0.045% of the span, mean 0.97%, worst
8.5%**, against the shipped `efit_psi_boundary` over all 115 frames.

**The coil gains confirm their own physics.** MAST ships one row per conductor turn — 812 of them,
656 being the solenoid — and the question is whether a column's flux map is the sum over its rows
or their average. Fitted against the truth, the three coil families whose geometry the dataset
ships in full come out at **P4 1.010, P5 0.961, P6 0.994** on the AVERAGED basis: the shipped
current is already the coil's total ampere-turns. That is a check the parameter count cannot fake,
and it is what licenses the two gains that are genuinely fitted — P3 at 8.69 and the solenoid at
169.8, both coils the dataset under-describes.

`P2` is the one number that does not fit the story, at 0.564. Its pair sits closest to the centre
column, where the grid is 0.030 m coarse against a 0.06 m conductor, and it drifted 1.03 → 0.68 →
0.56 across the rounds while every other family settled by round 3 — so it is absorbing a
discretisation error rather than reporting a turn count. Recorded rather than fixed.

**Alternating a gain fit with the solve diverges unless it is damped and symmetric.** The first
version fitted eleven independent gains against the truth minus the previous round's plasma flux,
and went R²ψ **0.639 → 0.291 → −0.428** with the well-posed gains swinging 0.65 → 1.44 → −0.51.
Two changes fixed it, and both are physics rather than numerics: an up/down coil pair shares one
turn count, so it shares one gain; and the plasma flux being subtracted came from a solve at the
PREVIOUS gains, so the update is damped to 0.4. It then converges monotonically —
0.634 → 0.873 → 0.928 → 0.916, settling at **0.9158**.

**The profile model, not the coil field, is what caps R²ψ.** Substituting the TRUE vacuum field —
recovered by taking `Delta*` of the truth, keeping the current inside the LCFS, and solving forward
— and sweeping the three profile numbers, the best any of them reaches is **0.954**. The
calibrated pipeline reaches 0.916, so the coil calibration costs 0.04 of R²ψ and the three-parameter
source shape costs 0.05. Whichever is worked on next, that is the split.

## Disclosure: one external dataset, used for geometry only

`solver/machine.py` carries MAST's first wall — a closed 37-vertex polygon — taken from the
**public MAST archive at STFC Echo** (`https://s3.echo.stfc.ac.uk/mast/level1/shots/{shot}.zarr`,
CC BY 4.0; index at `https://mastapp.site/json/shots`). `solver/mast_archive.py` is the reader.
**This must appear in any methods report**, as the competition terms require for external data.

Nothing else is taken from it, and the reason is that the archive's `efm/` group **is** the
competition's withheld MAST ground truth. Verified on demo shot 28348, whose truth the competition
does ship: frames match to 0.0000 ms and `efm/psirz` against `efit_psirz` differs by **0.000e+00**,
as do `li`, `betan`, `elongation`, `magnetic_axis_r`, `psi_axis`. The 1206-shot test fold is drawn
from these same campaigns. The terms allow external public datasets with disclosure and separately
forbid any attempt to "de-anonymize, memorize, or otherwise leak the hidden ground truth" — training
on the archive would be the latter even unintentionally, and excluding the test fold first requires
identifying it, which is also the latter.

The wall is neither: **byte-identical on shots sampled from all five campaigns M5–M9**, the same
kind of object as the coil rectangles the competition itself ships on every row, and an answer to no
shot. It corrects a guess — `mast/` had inferred an inboard wall at R = 0.1963 from where
`efit_lcfs_r` bottoms out on the demo shots, and the machine says **0.19524** — and, more usefully,
gives the whole shape, so the boundary search asks where the plasma touches the machine instead of
where it crosses one vertical line at the centre column. Worth **S 0.6310 → 0.6336** with the
calibrations untouched, and **0.6359** once they are refitted under the new geometry.

The organisers have been asked whether the wider archive may be used.

## The boundary is too small, and `edge` is the number that moves it

`mast/diagnose_boundary.py` decomposes D_LCFS — which is a **Hausdorff** distance, the worst point
on the contour, not an average — by removing one kind of error at a time. On the correct grid:

| | |
|---|---|
| as the scorer takes it | **0.3317** |
| with the plasma's position removed | 0.2401 |
| with position and size both removed | 0.1126 |

Roughly a third position, a third size, a third shape. Our contour is **0.929×** the true one on
average, only **49.5%** of it lies within 0.10·rgeo of the truth, and the worst point sits at the
outboard midplane on 81 of 115 frames — not in the divertor legs, which is where a size error shows
up first.

`Profile.edge` lets the source run past the last closed surface, as a fraction of the
axis-to-boundary flux span. Swept against the full composite it is worth **S 0.5737 → 0.5823** on a
broad plateau (0.5791 / 0.5823 / 0.5823 / 0.5795 at 0.01 / 0.02 / 0.03 / 0.04, against 0.5563 at
−0.02), raising R²ψ and Consistency together. It is physical rather than a fudge — there is real
current outside the separatrix — and it is aimed at a measured deficit rather than fitted blind.

The same sweep re-confirmed α = γ = 1 and β = 0.3 **against the composite**, which no earlier sweep
could do: until the scorer was fixed, D_LCFS and Consistency were 30% of the weight and meaningless.

**It was checked on DIII-D and DIII-D could not tell.** Over 2245 held-out DIII-D frames the same
knob moves R²ψ by 0.007 with no structure — 0.35321 / 0.35631 / 0.35572 / 0.35462 / 0.35356 /
0.35702 / 0.35041 at edge −0.02 to 0.08 — because DIII-D's own solve sits at 0.356 against MAST's
0.926 and a 2% shift in where the source switches off is below its own noise. That is neither
agreement nor disagreement, and it is the honest limit of the transfer plan today: **DIII-D becomes
a judge of MAST's choices only once its own solve is worth cross-validating.** So `edge` ships as
two probes and the board decides.

## The source term, driven by Thomson

The parametric source — one shape, two exponents, one mix, all global — is what capped R²ψ at
**0.954** even with the TRUE vacuum field substituted in, against a leaderboard best of 0.9732.
It is gone. MAST's Thomson is a **midplane** laser with a per-channel major radius, so `p(R)` is
measured; the flux map maps R to ψ_N inside the Picard loop, differentiating gives `p'(ψ)`, and

    J_phi  =  p_scale · R p'(psi)  +  lam (1 − psi_N^alpha)^gamma / R

with `lam` still fixed by the measured `Ip`. The pressure term carries whatever it carries and the
poloidal-current term takes the remainder. Nothing about the pressure is fitted per frame.

**R²ψ 0.9158 → 0.9370** on the demo frames, after re-alternating the coil gains against the new
source. And `p_scale = 2.0` is physics, not a fudge: on top of `2 n_e k T_e` it means an effective
`4 n_e k T_e`, and these are NBI-heated discharges where the fast-ion pressure is comparable to the
thermal one. The sweep has a genuine optimum rather than a preference for more — 0.875 at 0.25,
0.889 at 0.5, 0.908 at 1.0, **0.934 at 2.0**, 0.668 at 4.0.

A second-hand confirmation arrived with it: **P2's gain moved 0.564 → 0.726**, toward 1.0, so part
of what it used to absorb was the source rather than the coarse grid near the centre column.

The refinement grid's argmax (p_scale 1.5 / α 0.5, 0.93778) is deliberately **not** what ships.
The plateau spans 0.921–0.938 with neighbours at 0.778, and the board has already priced what a
three-shot argmax is worth — local D_LCFS 0.0996 became 0.4121 there. The middle of the plateau
ships.

## The two value scalars, and the constant that checks itself

They were constants worth R² = 0; they are now computed, at **q95 +0.5877 and βN +0.2948** pooled
over the 115 demo frames, i.e. **R²qb 0.4413** against the leader's 0.5079 and the 0.0574 the first
submission scored.

**q95 is a contour integral, an affine, and one physical constant — three things, kept separate.**
`q = (F / 2π) ∮ dl / (R |grad psi|)`, with a SINGLE power of R, because the textbook form
`q = (1/2π) ∮ dl B_phi / (R B_p)` carries `B_p = |grad psi| / R` and one R cancels. Writing `R²`
there was the first version's error, and the way it surfaced is the useful part: `F` fitted through
the origin came out fourfold below what MAST's actual half-tesla field requires.

- **`f_per_ka` is the toroidal field**, read off ψ_N = 0.95 — the surface q95 is defined on —
  through the origin: **0.004153 T·m/kA, i.e. B0 = 0.415 T** at an 85 kA feed and R₀ = 0.85 m,
  against a machine that runs at 0.4–0.55. That is the check, and it only works if the constant is
  read where it means something.
- **The shipped q95 uses ψ_N = 0.80**, a proxy chosen for correlation: with the Thomson-driven
  source the correlation with `efit_q95` is +0.751 / **+0.767** / +0.759 / +0.688 / +0.758 / +0.781
  at ψ_N = 0.70 / 0.80 / 0.85 / 0.90 / 0.95 / 0.99. (0.99 has the best affine R² at 0.611 but
  carries a surface on only **84 of 115** frames.)
- **The affine on top is where the proxy's offset belongs.** On the TRUE flux maps the same
  integral scores **0.551 through the origin and 0.809 with an intercept**. Fitting both `f_per_ka`
  and the level at 0.80 sent the constant to 0.0096 — a 0.96 T toroidal field — and it stopped
  being checkable at all.

**βN gets its shape from the same pressure profile** the solver now uses: `βN = 200 μ0 <p> a /
(B0 Ip)` with `<p>` volume-averaged over the plasma rather than along the chord. It ships through
an affine too, and that affine is a **safety** device as much as a fit: the composite averages
R²_q95 and R²_βN *before* flooring at zero, so a βN scoring −0.5 would throw away a q95 scoring
+0.6, and an affine with a small slope degenerates to a constant rather than to something worse.

## What DIII-D said, and why the transfer plan is not finished

`solver/` runs on DIII-D — 7041 labelled shots against MAST's three — precisely because three demo
shots have now refuted three decisions taken on them. What it has established:

**The physics transfers.** Integrating `−Δ*ψ / (μ0 R)` inside the shipped LCFS gives **+1274 kA
against a shipped Ip of +1192** on DIII-D, the same 7% as MAST. Sign, units and the operator are
right on both machines.

**The equilibrium does not.** The free-boundary solve puts DIII-D's magnetic axis at **Z = −1.0 m
against a true −0.03** — it slides a whole metre, because an elongated plasma has no stable vertical
equilibrium and EFIT pins it with magnetic measurements this challenge withholds. Two pins were
built and both help, paired on the same shots: bounding where the O-point may be FOUND, and then
shifting the CURRENT so its own centroid lands on the pin — the second matters because the first
leaves the source free to sit elsewhere, and on DIII-D the two part company by 0.17 m.

**And the fix does not come back to MAST.** The search pin moves MAST's R²ψ 0.91579 → 0.91562 and
its axis error from −0.0360 to −0.0347 m, i.e. nothing. The centroid pin is worse: **S 0.6296 →
0.612**, with 1 − D_LCFS rising to **0.7220** — the best boundary number measured here — while q95
falls 0.3768 → 0.1068 with the affines refitted for the pinned map. MAST's centre column already
anchors its plasma, so forcing the current to a fixed height overrides a solution that was right.

**And the honest level is worse than the paired gains suggest.** Every DIII-D score quoted on
2026-08-19 before the correction was measured on shots the coil gains had been fitted on — 8 of 8
in one diagnostic, 11 of 20 in another. `solver/how_many_shots.py` scores the same estimator on a
genuinely disjoint set:

| shots the gains were fitted on | held-out R²ψ |
|---|---|
| 1 | −6.58 ± 8.99 |
| 3 | −0.601 ± 0.120 |
| 6 | −0.253 ± 0.046 |
| 12 | −0.235 ± 0.063 |
| 24 | −0.148 ± 0.087 |

Negative at every size, still climbing at 24, and at one shot a single unlucky draw returns −19.3.
So **DIII-D is much further from being a judge of MAST's choices than the contaminated numbers
suggested**, and the 3-shot cross-validation the transfer plan is built on has nothing to
cross-validate yet.

**And the blocker is now named.** DIII-D's error is the flux LEVEL, not the shape: removing one
constant per frame takes held-out R²ψ from +0.207 to **+0.711**, and the offset is +0.006 ±
**0.147 Wb/rad** on a flux range of 1.4 — mean zero, enormous scatter, and not recoverable from the
inputs (regressed on all 19 currents plus Ip it scores 0.95 in sample and **0.031** held out).
`coil_field.py` already records why: ECOILA's field over this grid is nearly degenerate with a
constant offset, so its gain sets the absolute level, and one number cannot track a level that moves
through a discharge as ohmic flux is consumed. For Challenge 1 that never mattered — whatever is
subtracted is added back unchanged. For a forward solve it is most of the error. **The next piece of
work on DIII-D is the flux gauge, not another knob.**

**MAST does not have that problem**, which is why one machine works and the other does not. Asked
the same question here, the per-frame offset is **−0.00014 ± 0.00418 Wb/rad on a range of 0.261 —
1.6%**, against DIII-D's 10.5%, and removing it moves R²ψ only 0.93170 → 0.94101. The difference is
in what the dataset ships: MAST's 812 conductor rows give the solenoid a *shape* the fit can
separate from a constant, where DIII-D's single ECOILA rectangle cannot be told from an offset. So
**MAST's remaining flux error is shape, not level** — a perfect gauge would be worth +0.005 of S —
which agrees with the boundary decomposition charging it to size and position.

It does not convict MAST's own three shots, and the difference is identifiable: MAST ships 812
conductor rows and its well-posed coil families come out at 1.010 / 0.961 / 0.994, so only two
numbers are really fitted there, while DIII-D's ECOILA is degenerate with a constant offset and
drags eighteen F-coils with it. The evidence that MAST's calibration travels is the leaderboard —
local 0.9158 against a fold 0.8895, a drop of 0.026.

Two calibration lessons were paid for on the way and both generalise: the gain fit must run
**outside the plasma boundary** (inside it one filament cannot represent a distributed current;
fitting over the whole grid returned F-coil gains from −1.4 to +2.8), and it must project out a
**per-frame plasma amplitude**. With both, DIII-D's F-coils come out at a median of **0.965,
10–90% 0.855–1.137**, which is independently what `my_experiments/coil_field.py` reports. A free
19-gain alternation beats the two-stage fit where it is fitted and loses badly where it is not.

## Known gaps, in the order they are worth closing

1. **The Consistency term, at 0.0742.** Read the caveat above before spending anything on it: on
   three shots this number is dominated by a denominator that will not be there on the real fold.
   The honest reading of the geometry is D_LCFS = 0.0996 against DIII-D's 0.0098 — the boundary is
   about ten times as far off as Challenge 1's, which is where any real gain is.
3. **The profile has three numbers for the whole machine.** The per-frame quantity it is standing
   in for is `beta_p`, which sets the Shafranov shift — and the solve puts the magnetic axis at
   0.94–1.04 m against a true 0.79–0.91, i.e. the shift is systematically too large. Thomson's
   pressure would constrain it per frame rather than globally, which is the same measurement item 1
   needs for βN.

## Cost

One shot of 36–42 frames takes **9 s** end to end, dominated by the Picard iteration at ~0.24 s a
frame; the coil basis costs 13.3 s but is built once, because MAST's 812 conductor rows are
byte-identical on every shot. The full 1206-shot test split is therefore about **4 hours on one
core**, or ten minutes on the box.
