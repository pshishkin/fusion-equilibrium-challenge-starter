# What we tested, and what came of it

One line per experiment: what was tried, the number it produced, the verdict. Untested ideas live
in [ideas.md](ideas.md) and move here once there is a number attached — whichever way it went.

Gains are **paired** differences of S within one split salt; "quality" is
`0.05/0.2 0.05/0.2 0.01`, "prod" is `0.45/0.1 0.15/0.1 0.01`. What a run can resolve is measured,
not assumed — see "The noise, measured properly" below: at production a single paired run resolves
**0.0013**, and three seeds on one salt resolve 0.0008.

| date | tested | result | verdict |
|---|---|---|---|
| 08-11 | Thomson explains the failing shots | R² correlates −0.07 with Te, 0.00 with ne, 200 shots | refuted |
| 08-11 | the error is a per-shot flux offset | removing the constant recovers 21% of it | refuted |
| 08-11 | normalize polarity by sign of Ip | 0.68 → 0.63 | rolled back |
| 08-12 | standardize each PCA component | 0.7466 → 0.7155; whitens the loss | refuted |
| 08-12 | ridge as an ensemble member | 0.7466 → 0.5936; dilutes, not diversifies | rolled back |
| 08-12 | early stopping for CatBoost | 0.7435 → 0.7439 across every budget tried | no effect |
| 08-12 | MLP patience | 0.5521 at 20 vs 0.7273 at 100 — the patience IS the hyper-parameter | kept at 100 |
| 08-12 | more shots, fewer frames per shot | 0.8111 → **0.9319** at a fixed row budget | kept |
| 08-12 | `inputs: coil_pca` instead of raw currents | +0.0102, −0.0042, +0.0052 over 3 seeds | no effect; kept as an option for MAST |
| 08-12 | subtract the analytic coil field | quality **+0.0088** (3 seeds) but prod **+0.0007** | see "reversals" |
| 08-13 | `field` loss, hand-set λ | +0.0048 at λ=0.01, +0.0059 at λ=0.1, −0.0054 at 0.3 — a flat plateau | removed; it showed the effect exists and could not locate it |
| 08-13 | `jacobian` loss metric | quality **+0.0117** (3 salts, t=4.71), prod **+0.0097** at n_pca 50 | kept, default — the one change that pays at production |
| 08-13 | refit `jacobian_delta` on the prod model | +0.0029, one run | not measured; direction agrees with theory |
| 08-13 | boundary term in the metric | +0.0021 over 3 salts, and `1−D_LCFS` did not move (1 sign of 3) | refuted |
| 08-13 | `n_pca` above 50 | prod: 75 gives −0.0010, 100 gives 0.0000; the ceiling only rises +0.0003 of S anyway | saturated, do not raise |
| 08-13 | `n_pca` below 50 | quality: 30 beats 50 by **+0.0043** (3 salts, t=4.84). prod: 50 beats 30 by 0.0014, 20 by 0.0055 | see "reversals" |
| 08-13 | training shots 1408 → 3168 | 0.9769 → **0.9809**, every term up | kept |
| 08-13 | is the Jacobian's linearisation valid at the model's real error? | over/under-estimate and correlation per scalar, 318 held-out frames — see below | measured; the metric is miscalibrated per scalar |
| 08-13 | per-scalar calibration from random full-vector probes | ratios 1.08–1.28 for all seven, against li 1.65 / tri_bot 1.51 / R_axis 0.98 from the model's real error | refuted as built — the ratio is a property of the error DIRECTION, not of the functionals |
| 08-13 | **the noise, on 3 salts x 3 seeds at production** | seed sigma **0.0009**, salt spread 0.0018 — see below | measured; the 0.0060 we had been using was five times too big |
| 08-13 | are frames where extraction fails a material share of the loss? | `lcfs_fail_frac` and `cons_fail_each` are both **0.0%** at production | refuted, for free — there is nothing there to repair |
| 08-13 | where the 9 minutes go | MLP fit 352 s of 467 s training; reading shots 82 s; PCA 10 s; Jacobian 19 s | measured — the fit dominates, not the I/O |
| 08-13 | what holds the memory | peak 21.6 GiB against 1.5 GiB of data; ~19 GiB is idle pool workers | measured; fixed by `parallel.release()` and the shot cache |
| 08-13 | is the DataLoader worth its overhead? | 0.269 s/epoch against 0.130 s for the same batches by indexing | measured — 52% of every epoch is DataLoader bookkeeping |
| 08-13 | decoded-shot cache + pool release + sampler-driven batches | prod run **8:45 → 4:54**, peak **21.6 → 8.5 GiB**, S unchanged at 0.9809 and the same best epoch | kept — pure speed, verified bit-identical |
| 08-13 | `pca_frame_share: 0.2` | PCA fit 9.1 s → 4.6 s of a 295 s run, memory 7.79 → 7.81 GiB, S 0.9809 → 0.9816 (+0.0007, sd 0.0013) | not measured on score, negligible on cost — knob kept at 1.0 for when shots grow |

## The two reversals, and what they cost us

Two results measured cleanly at quality scale did not survive at production, and both for the same
reason: **the bigger model can do for itself what we were doing for it.**

- `n_pca = 30` beat 50 at quality on all three salts with t = 4.84 — and at production 50 won. With
  352 shots the tail PCA components are noise the model cannot predict and the metric over-weights
  (condition number 5266 → 2030 → 364 as n falls); with 3168 shots they are signal, and truncating
  throws it away.
- Subtracting the coil field was worth +0.0088 at quality and **+0.0007** at production. At 3168
  shots the model learns the coil field on its own, so computing it analytically saves almost
  nothing. It stays because MAST has no DIII-D coils to learn from — not because it pays here. Half
  a day of Green's functions, a lost 1/π and a sign convention, for nothing measurable on DIII-D.

The production ablations, all at salt 0 against S = 0.9809 (n_pca 50, jacobian, subtraction on):
`parseval` −0.0097, no subtraction −0.0007, n_pca 75 −0.0010, n_pca 100 0.0000. Only the first
clears the noise floor.

Hence the rule in AGENTS: quality screens, production decides.

## How good the linearisation actually is

The whole Jacobian metric assumes the seven functionals respond linearly at the error the model
actually makes. Checked by taking the model's real error direction `dc = c_pred − c_true` on 318
held-out frames, computing the true change of each functional, and comparing with `J·dc`:

| scalar | linear / actual | correlation | bias share of the miss |
|---|---|---|---|
| R_axis | 0.98 | 0.99 | 0.25 |
| Z_axis | 0.91 | 0.91 | 0.02 |
| volume | 0.96 | 0.90 | 0.09 |
| li | **1.65** | 0.92 | 0.11 |
| kappa | 1.21 | 0.77 | 0.02 |
| tri_top | 1.14 | 0.46 | 0.05 |
| tri_bot | **1.51** | 0.45 | 0.01 |

Three readings:

- **`li` is OVER-weighted, not under-weighted.** The prediction that its quadratic nature would
  show up as a systematic positive bias was wrong in emphasis: the bias exists and has the right
  sign but is 11% of the miss. What actually happens is that the Jacobian thinks `li` moves 1.65×
  more than it does, so `M ∝ JᵀJ/var` hands it too much of the loss budget. That is a better
  explanation of why `li` was the one scalar to get worse under this metric.
- **The triangularities barely linearise at all** — correlation 0.45. They read extreme points of
  the contour, which move discontinuously.
- **`R_axis` linearises almost perfectly** (0.98, corr 0.99), and it was the thing we most needed
  to fix. `Z_axis` and `volume` are fine too.

The metric pays +0.0097 at production despite being a mediocre point-predictor for three of the
seven. What matters is evidently the relative weighting of directions in aggregate, not per-frame
fidelity.

## The noise, measured properly

Three split salts times three MLP seeds, nine production runs, 2026-08-13. Seeds vary only the
optimisation, salts vary which shots are in the fold, so **the noise of a paired A/B test is the
spread across seeds inside one salt** — never the spread across salts, which measures a different
fold rather than a different answer.

| salt | seed 0 | seed 1 | seed 2 | mean | sd |
|---|---|---|---|---|---|
| 0 | 0.9809 | 0.9788 | 0.9787 | 0.9795 | 0.0012 |
| 1 | 0.9780 | 0.9774 | 0.9781 | 0.9778 | 0.0004 |
| 2 | 0.9767 | 0.9748 | 0.9762 | 0.9759 | 0.0010 |

**Pooled seed sigma = 0.0009** (6 dof). Salt-to-salt spread of the salt means: 0.0018 — twice as
large, as it should be, since the salt changes the data. Grand mean 0.9777, which is also the
correction to a headline: **0.9809 was the best of three seeds, not the typical one.**

| how a difference is measured | sd of the difference |
|---|---|
| one salt, one seed, paired | **0.0013** |
| one salt, mean of three seeds, paired | **0.0008** |
| three salts x three seeds, paired | **0.0004** |

The noise is not spread evenly. Per term: R²ψ 0.0001, `1 − D_LCFS` 0.0005, R²qb 0.0012, and
**Consistency 0.0045** — which, at weight 0.20, is essentially all of it. Per scalar:

| scalar | mean | sigma over seeds |
|---|---|---|
| Z_axis | 0.911 | **0.0234** |
| R_axis | 0.888 | **0.0177** |
| volume | 0.890 | 0.0083 |
| li | 0.896 | 0.0083 |
| kappa | 0.878 | 0.0080 |
| tri_top | 0.979 | 0.0033 |
| tri_bot | 0.976 | 0.0017 |

So a 0.01 gain on `R_axis` is noise and the same 0.01 on `tri_bot` is a clear signal. The two axis
positions are the least reproducible quantities in the whole metric — which is what you would
expect of a point defined by ∇ψ = 0, and an independent argument for anything that removes
frame-to-frame jitter.

## Where the nine minutes and the twenty gigabytes go

One production run, instrumented by `my_experiments/progress.py` (every printed line carries
elapsed time, the gap since the previous line, and the resident memory of the whole process tree).

| stage | time | share |
|---|---|---|
| read 3168 training shots | 61.9 s | 12% |
| read 1056 validation shots | 19.7 s | 4% |
| coil field: fit gains and subtract | 4.5 s | 1% |
| PCA fit | 9.8 s | 2% |
| PCA transform + Jacobian probe | 18.6 s | 4% |
| **MLP fit** (1148 epochs, stopped at 1047) | **352.1 s** | **67%** |
| scoring 70 shots, three models | 58 s | 11% |

Peak memory 21.6 GiB against **1.5 GiB of data**. The flux arrays are 1.47 GiB; nearly everything
else is the reader pool's 18 workers, which never return their parquet decode buffers to the OS
and then sit idle through the six-minute fit. That is the swap we saw, and it was invisible until
the memory went into the log.

Two more numbers from the same run: the MLP's `DataLoader` costs 0.269 s per epoch where the same
batches taken by indexing cost 0.130 s, so **52% of every epoch is bookkeeping**; shuffling itself
is only 5%.

### What that bought, once fixed

| | before | after |
|---|---|---|
| whole production run | 8:45 | **4:54** |
| reading 4224 shots | 81.6 s | **4.1 s** |
| MLP fit | 352.1 s | **197.0 s** |
| peak resident memory | 21.6 GiB | **8.5 GiB** |
| S | 0.9809 | 0.9809 |

Three changes, all verified bit-identical rather than merely close: the decoded-shot cache
(`shot_cache.py`), `parallel.release()` before the fit, and taking the MLP's batches from the
`DataLoader`'s own sampler while gathering the rows in one indexing operation.

The batch change deserves its own note, because the obvious version of it is wrong. A hand-rolled
`randperm` loop agrees with `DataLoader` on the first epoch and diverges from the second on:
`DataLoader` draws a base seed from the generator every time an iterator is created, so the
permutations fall out of step. Driving `iter(loader)._sampler_iter` consumes the generator exactly
as the real thing does. The proof is not the score — that could coincide — but the **best epoch:
1047 of 1148 both before and after**, so the optimisation walked step for step.

## Measurement facts worth not re-deriving

- The noise floor at production is **0.0009** of S over seeds; see the table above. An older figure
  of 0.0060 came from quality scale (352 shots) and does not apply to the production model — with
  3168 shots the optimisation lands in nearly the same place every time.
- **Salts, not seeds.** Ridge is deterministic given a split and moves 0.7117 → 0.6219 → 0.5067
  across salts 0/1/2, against 0.0 for any seed change.
- **Ceilings must be measured on the term that loses the score.** "50 components reconstruct ψ to
  R² = 1.000000" was true and measured on R²ψ, which saturates at 30 components; Consistency does
  not saturate until 50. The conclusion survived by luck.
- **The instantaneous currents nearly determine the targets**: nearest-neighbour across shots gives
  a ceiling of R² 0.97 (li) to 0.997 (betaN), while the model reaches 0.90–0.93. Features are not
  the binding constraint.
- **`jacobian_delta` is scale-dependent**: it must sit at √(1 − R²ψ) of the run it is used in —
  0.05 at quality, 0.033 at production.
- **The coil-field calibration** needs a plasma filament with a per-frame amplitude, or the coil
  gains absorb the plasma's own field. Pooled amplitude reads 0.997 on 10 shots and 0.911 on 40;
  per frame it is 0.87–0.89 at any size. `ECOILA` is not identifiable at all — its gain moves
  between +142 and −10 depending on the fit — so the pipeline fits its own gains instead.
