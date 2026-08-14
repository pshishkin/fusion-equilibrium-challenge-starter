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
| 08-13 | **it-1** temporal smoothing of the predictions (A1) | the prediction is **half as rough** as the truth: 0.021 against 0.039. Smoothing raises coefficient MSE at every window past 60 ms | refuted, no production run spent |
| 08-13 | **it-2** smoothing only the tail components, in time not frames | best cell −2.7% of the error in the loss metric, which is ΔS ≈ +0.0004 against a resolution of 0.0013 | refuted — below what any run could read |
| 08-13 | **it-3** post-hoc de-biasing of the seven scalars (A7) | biases are ≤ 0.22 of the RMS miss, slopes 0.92–1.01; a perfect affine correction buys **+0.0004**, and that is fitted in-sample | refuted |
| 08-13 | **it-4** are reversed-Ip shots harder for the MLP? (A12) | metric R² 0.9898 on 5 reversed shots against 0.9208 on 65 normal ones, t = −3.85 | refuted with the **opposite** sign — reversed shots are easier |
| 08-13 | **it-5** where the residual actually is | **one shot of 70 carries 23% of the squared error**, two carry 33%, ten carry 54% | measured — the tail, not the average shot, is the target |
| 08-13 | it-5b are the failing shots out of distribution? | \|z\| max 2.8 and 1.8 against a fold median of 2.3; the most extrapolated shots are not the worst. Error spread evenly over frames | refuted — ordinary inputs, uniformly wrong |
| 08-13 | it-5c how much of the error is a per-shot constant? | **45.9% of the fold**, 85.4% of the worst shot; removing it takes the fold's metric R² 0.9594 → 0.9780 | measured — the dominant failure is a shot-level bias |
| 08-13 | it-5d is that constant a flux gauge? | **98.3% is a shaped map**, 1.7% a uniform offset | refuted as a gauge — it is a real field, and one that lasts the whole shot |
| 08-13 | **it-6** is that per-shot constant predictable from the current history? | 252 history features, ridge fitted on 400 validation shots: **0.008** explained on the held-out fold against 0.003 for predicting the mean | refuted — the history carries no signal about it |
| 08-13 | **it-7a** the `li` probe ratio printed `nan` at production | one full-vector probe in a thousand leaves `li` undefined, and one nan poisoned the sum over 300 frames | fixed — finite probes counted, the drop rate printed |
| 08-13 | **it-7** `calibrate_scalars: true` at production, 3 salts | +0.0007 / −0.0003 / −0.0006, mean **−0.0001**, 1 sign of 3 | refuted — it moves budget between terms, not into the score |
| 08-13 | **it-8** MLP 256x256 → **512x512**, 3 salts | +0.0032 / +0.0021 / +0.0013, mean **+0.0022**, 3 signs of 3, every term up and Consistency +0.0131 on salt 0 | **KEPT** — the first accepted change of the loop |
| 08-13 | **it-9** derivative features of `ECOILA` and `Ip`, salt 0 | `raw` −0.0011, `interp` −0.0001, `both` +0.0007 against 0.9841 | only `both` positive, and below what picking the best of three arms produces by chance |
| 08-13 | **it-9** `both`, confirmed on 3 salts | +0.0007 / +0.0003 / +0.0011, mean **+0.0007**, 3 signs of 3 | **not measured** — 0.93σ, below the pre-declared +0.0015; reverted |
| 08-13 | **it-10** Thomson scattering as 11 feature columns, 3 salts | +0.0027 / +0.0033 / +0.0056, mean **+0.0039**, 3 signs of 3; R²qb +0.0064 and Consistency +0.0086 on salt 0, and ridge 0.7022 → 0.7562 | **KEPT** — the largest single gain measured in this fork |
| 08-13 | **it-11** MLP 512x512 → **1024x1024**, salt 0 | 0.9866 against 0.9868, and the fit costs 706 s instead of 230 | refuted — the capacity line is exhausted, the constraint is now the data |
| 08-13 | **it-12** which Thomson groups matter, salt 0 | `te` 0.9840, `te,p` 0.9861, `te,ne,p` **0.9874**, `te,ne,p,shape` 0.9874 | measured — temperature alone is worth nothing, pressure carries it, density adds +0.0013, `shape` adds exactly zero |
| 08-13 | **it-13** `te,ne,p` against the it-10 set, 3 salts | +0.0006 / −0.0025 / +0.0022, mean **+0.0001** | not measured — the two sets are indistinguishable; kept the group-expressible one on grounds other than score |
| 08-13 | **it-14** drop the four columns permutation importance called dead | 0.9862 against 0.9874, **−0.0012** | reverted — see "importance is not ablation" below |
| 08-13 | permutation importance over every input column | pressure integrals top the Thomson block at 253%/244%; the four position columns 0.1–2.3%; `bcoil` 91% | measured — see "importance is not ablation" |
| 08-14 | **it-15** ensemble of 4 MLP seeds, 3 salts | +0.0034 / +0.0041 / +0.0050 over the seed mean, mean **+0.0042**, 3 signs of 3 | **KEPT** — the largest gain of the loop |
| 08-14 | seed sigma, re-measured on the current recipe | 0.00085 / 0.00132 / 0.00155 by salt, pooled **0.0013** | the morning's 0.0009 landed on the quietest salt; thresholds updated |
| 08-14 | **it-16** MLP patience 100 → 300, salt 0 | ensemble 0.9904 against 0.9902; the one undertrained seed gained +0.0012 (best epoch 121 → 645), the other three did not move at all | not measured — the ensemble already absorbs a weak member, at nearly double the fit time; reverted |
| 08-14 | **it-17** training shots 3168 → **4225** (0.45 → 0.60), 3 salts | +0.0012 / +0.0029 / +0.0013, mean **+0.0018**, 3 signs of 3; the seed spread also narrowed from 0.0018 to 0.0010 | **KEPT** — and it is what iteration 11 predicted: the constraint was the data |
| 08-14 | **it-18** the whole accepted stack, on salts **3 and 4** which no iteration ever saw | +0.0077 and +0.0057 against +0.0115 on the three salts used for selection | measured — about 40% of the claimed gain was selection bias; the real gain is **+0.0067** |
| 08-14 | **it-19** CatBoost at production, as a fifth ensemble member | ensemble 0.9916 against 0.9914; CatBoost alone 0.9838 against 0.9875–0.9885 for the MLPs, and it took **3829 s against 296 s**, hitting the iteration ceiling undertrained | refuted on both counts — the gain is noise and the cost would have made the iteration loop impossible |
| 08-14 | **it-20** raw Thomson PRESSURE profile (16 points per system) instead of / beside the summaries | summaries 0.9914, summaries+raw 0.9901, raw alone 0.9900 | refuted — raw alone equals raw+summaries, so the summaries add nothing to it, and yet three numbers per system BEAT sixteen points by 0.0014 |
| 08-14 | **submitted** the four-change stack | leaderboard **0.9896**, 4th place, against 0.9764 for the previous submission — **+0.0132** where the local salt-0 gain was +0.0145 | 91% of the local gain transferred; the fresh-fold estimate of 58% was too pessimistic |
| 08-13 | is the EFIT frame step really 20 ms? | 98.0% of intervals are, but **31% of shots carry a gap over 100 ms**, up to 900 ms | measured — any derivative or time window must use the real step |
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

## Why the jitter idea was wrong, which is worth more than the idea

It was ranked first: frames are predicted independently, they are 20 ms apart, the equilibrium
moves on hundreds of ms, therefore the prediction must carry high-frequency error that physics
forbids. The reasoning is sound and the fact is the opposite.

Measured on 30 held-out shots, variance-weighted over the 50 components, as
`var(lag-1 difference) / var(series)`:

| | roughness |
|---|---|
| predicted | 0.0212 |
| truth | 0.0394 |
| ratio | **0.54** |

The prediction is **twice as smooth as the ground truth**. Predicting a conditional mean from
instantaneous currents produces a smooth series by construction; it is EFIT's own reconstruction
that jitters. Independence between frames does not cause noise here, it causes over-smoothing.

The ratio is not uniform — pc0 (83% of the variance) sits at 0.49 while pc2 is 1.71 and pc6 is
3.02, so the tail really is under-smooth — but smoothing only the tail, with a Gaussian kernel in
milliseconds rather than a window in frames, buys at most 2.7% of the error measured in the loss
metric. That is ΔS ≈ +0.0004 against a resolution of 0.0013: not worth a run.

Two methodological points fell out of it:

- **Measure the error in the metric the loss uses.** Plain coefficient SSE said only one cell of
  the sweep helped; the same numbers under `M = L Lᵀ` said most of the tail cells did. 95% of plain
  SSE is pc0 and pc1, which are not where the score is lost.
- **Windows in frames are not windows in time.** 98.0% of EFIT intervals are 20 ms, but 31% of
  shots carry a gap over 100 ms and the largest is 900 ms, so a five-frame window sometimes spans
  half a second. The first version of this diagnostic had that bug; the conclusion survived it,
  because the roughness ratio compares the same index pairs on both sides.

## The one thing that worked was the one nobody had touched

Eight iterations of the improvement loop on 2026-08-13. Seven refuted, six of them for free; the
eighth was **widening the MLP from 256x256 to 512x512**, and it gained +0.0022 of S over three
salts with all three signs agreeing — more than every physics change of the day put together.

| salt | 256x256 | 512x512 |
|---|---|---|
| 0 | 0.9809 | **0.9841** |
| 1 | 0.9780 | **0.9801** |
| 2 | 0.9767 | **0.9780** |

Every term moved, and Consistency carried it: 0.9283 → 0.9414 on salt 0. The fit did not even cost
proportionally more — 262.9 s against 197.0 — because the wider net stopped at epoch 548 instead
of 1047.

Two things worth keeping from that. **21 → 256 → 256 → 52 had stood since the starter kit and had
never been swept once**, while a day went into Green's functions and quadratic forms. And the
"bigger model does it for itself" trap that killed two earlier results does not apply here,
because this *is* the bigger model — which is exactly why it was the safest capacity bet on the
list.

## Thomson, and the argument that nearly buried it

Iteration 10 added eleven columns per frame from the Thomson profiles and gained **+0.0039** of S
over three salts, every sign agreeing — the largest single change this fork has measured. It
landed where the physics said it would: `R2_qb` moved +0.0064, and betaN is a pressure ratio by
definition. `ridge` moved 0.7022 → 0.7562, which matters because a linear model cannot invent a
correlation: the information is there and it is linearly available.

**The argument that ranked it low was wrong, and the mistake is reusable.** Thomson sat near the
bottom of the list because a nearest-neighbour test had shown that the coil currents alone
determine betaN to R² 0.997 while the model reached 0.981 — "features are not the binding
constraint, the regression is". That inference does not follow. A nearest-neighbour ceiling says
what a PERFECT learner could extract from the currents; it says nothing about how HARD the
extraction is for a real one. Supplying a quantity that is formally redundant still helps when it
makes the function easier to represent — and betaN is nearly linear in pressure, while recovering
pressure from twenty-one coil currents is not.

Read the ceiling as "these features are in principle sufficient", never as "new features cannot
help". The same faulty step was applied to every feature idea in the list.

Three data facts the implementation had to be built around, none of them visible from the column
names:

- **The core system is a VERTICAL chord.** All 44 core channels sit at R = 1.940 and differ in Z.
  The radial derivative that Grad-Shafranov asks for exists only on the EDGE system — 10 channels
  at Z ≈ −0.06 spanning R = 1.68..2.06 — which is also exactly the pedestal region.
- **Missing samples are exact ZEROS, not nan.** Inside the EFIT window 14.1% of core values are
  zero; outside it, where there is no plasma, nearly all are. Every reduction runs on a mask.
- **Channels are not stored in positional order**, so any difference along the chord has to sort
  by coordinate first.

And one whole shot — `d3d_shot_5a79f2123a` — has Thomson that never fired: 1884 core samples and
423 edge samples with zero live channels in every one. It was invisible in samples of 40 and 120
shots and appeared at 3168. Such shots contribute zeros plus a **validity flag column**, so the
model is told the measurement is absent rather than handed a fabricated one, and the rate is
printed: 0.7% of production training frames.

## The summaries are a good compression, not a lossy one

The group sweep showed temperature summaries worth nothing and pressure carrying everything, which
suggested the obvious next step: the summaries throw away the SHAPE, and Grad-Shafranov reads
p(psi) rather than its peak. So the raw pressure profile went in, resampled onto 16 fixed points
along each chord — fixed points rather than channels, because channels die and revive and a column
tied to one would mean a different place in different frames.

| what the model saw | S | R²qb | Consistency |
|---|---|---|---|
| three summaries per system | **0.9914** | 0.9937 | **0.9685** |
| summaries + 16-point profile | 0.9901 | 0.9922 | 0.9638 |
| the 16-point profile alone | 0.9900 | **0.9939** | 0.9615 |

Raw alone equals raw+summaries, so the summaries carry nothing the profile does not. And the
summaries still WIN by 0.0014. The compression is not lossy in any way that matters — it is
better than the thing it compresses, because an integral averages the noise of a 44-channel chord
while a single profile point carries all of it.

One asymmetry worth keeping: the raw profile is the best of the three on `R2_qb` (0.9939) and the
worst on Consistency (0.9615). Pointwise pressure helps the pressure scalars; integrated pressure
helps the geometry.

## The leaderboard, and how the two estimates of selection bias compared

Submitted 2026-08-14: **0.9896, 4th place**, against 0.9764 for the previous submission.

| | local, salt 0 | leaderboard |
|---|---|---|
| previous submission | 0.9769 | 0.9764 |
| this one | 0.9914 | **0.9896** |
| gain | +0.0145 | **+0.0132** |

**91% of the local gain transferred.** The two-fresh-salt estimate below said 58% would — it was
too pessimistic, and by a lot. Both numbers are worth keeping, because the disagreement is the
lesson:

- Salts 3 and 4 are **70 scored shots each**. A paired difference between two configurations that
  differ in four ways carries fold-specific variance far beyond the 0.0009 seed sigma, and two
  such folds cannot pin a number to better than a few thousandths.
- The leaderboard is **874 shots** — twelve times the fold — and it is the only estimate here with
  no selection history at all.

So the honest reading is that the held-out-salt check correctly showed selection bias EXISTS and
correctly refused to endorse the headline, but its magnitude (+0.0067) was itself a noisy estimate.
The gap between local and leaderboard is 0.0018 this time against 0.0005 for the previous
submission — small, and consistent with a modest amount of fold-tuning rather than a large one.

## What seventeen rounds of choosing on the same folds actually cost

The loop selected against salts 0, 1 and 2 — the same 210 held-out shots — seventeen times. Salts
3 and 4 were kept untouched from the start for exactly one purpose: to price that.

| salt | this morning's recipe | everything the loop accepted | gain |
|---|---|---|---|
| 0 | 0.9809 | 0.9914 | +0.0105 |
| 1 | 0.9780 | 0.9889 | +0.0109 |
| 2 | 0.9767 | 0.9898 | +0.0131 |
| **3** | **0.9789** | **0.9866** | **+0.0077** |
| **4** | **0.9843** | **0.9900** | **+0.0057** |

**Selected folds +0.0115. Fresh folds +0.0067.** The difference, +0.0048, is what seventeen rounds
of looking at the same tail bought us in self-deception — about 40% of the headline.

Both halves of that matter. The gain is REAL: +0.0067 on folds that never took part in a decision
is roughly seven sigma against a paired sd of ~0.0009. And the headline was INFLATED: any
expectation for the leaderboard should be built from +0.0067 over the last submission, not from
the local 0.9914.

Two practical notes. Salt 4 is an easy fold — the old recipe scores 0.9843 there against
0.9767–0.9809 on salts 0–2 — which is why absolute scores are never compared across salts, only
paired differences within one. And the old recipe could not be run at all until three keys
(`derivatives`, `derivative_signals`, `thomson`) were added to it in their off positions: the
config validator refuses a file with missing keys rather than assuming defaults, so a
configuration from before those knobs existed is not silently runnable.

## Averaging seeds is worth far more than the seed spread suggests

Four MLPs differing only in their seed, averaged: **+0.0042** of S over the mean of those same four,
on three salts with every sign agreeing. It is the largest single gain in this fork.

| salt | seed mean | ensemble | seeds ranged over |
|---|---|---|---|
| 0 | 0.9868 | **0.9902** | 0.9856–0.9874 |
| 1 | 0.9819 | **0.9860** | 0.9808–0.9836 |
| 2 | 0.9836 | **0.9885** | 0.9823–0.9858 |

**The obvious estimate is wrong by about sevenfold, and the reason is worth keeping.** Reasoning
from the spread of the SCORE across seeds — sigma 0.0013, so averaging four should buy about half
a sigma — predicts +0.0005. What averaging actually reduces is the prediction ERROR, and the score
is a nonlinear function of that: four independent nets cut the independent component of the
squared error by up to 4x, and R² follows directly. Consistency shows it plainly — 0.9640 for the
ensemble against 0.9456–0.9546 for its members, a jump ten times larger than averaging their
scores would give.

Two consequences for how everything after this is measured:

- **Sigma is 0.0013, not 0.0009.** Re-measured on the current recipe it is 0.00085, 0.00132 and
  0.00155 on salts 0, 1 and 2; the morning's figure happened to land on the quietest one.
- **But the production model is now itself an average of four**, whose own seed-to-seed sigma is
  half that, ~0.00065. Comparing configurations by the ensemble rather than by a single seed makes
  every future measurement twice as precise — at four times the cost per run.

## Importance is not ablation, measured the hard way

Permutation importance over every input column, on held-out shots, in the loss metric. The coil
currents dominate at 1200–4300%. Within the Thomson block:

| column | error increase when shuffled |
|---|---|
| `th_edge_int_p` | **253%** |
| `th_core_int_p` | **244%** |
| `th_core_slope_p` | 164% |
| … | |
| `th_core_at_p` | 2.3% |
| `th_edge_at_te` | 1.8% |
| `th_core_at_te` | **0.1%** |
| `th_valid` | **0.0%** |

It agreed with the group ablation on the big thing — pressure carries the Thomson gain, and `p'`
ranks third — and then **it was wrong about what to remove**. Dropping the four near-zero position
columns cost −0.0012. Permutation importance measures what a FITTED model leans on, not how much
information a column holds: correlated columns mask each other and all of them look free.

`th_valid` measures exactly zero and is kept deliberately. It insures the 0.7% of shots whose
Thomson never fired, and the held-out fold evidently contains none of them — a rare-event feature
is invisible to this method by construction.

One surprise worth keeping: **`bcoil` scores 91%**. It is toroidal and drives no poloidal flux,
which is why it has no rectangle in the coil basis and was excluded from the derivative features.
The model leans on it anyway, most plausibly because it is nearly constant within a discharge and
so identifies the scenario — one of the very few per-shot signals available, which is interesting
next to the per-shot constant below.

## The error is a per-shot bias, and that changes what is worth trying

Five iterations of the improvement loop on 2026-08-13 spent no production runs and refuted four
ideas, but the fifth found the shape of the problem, which is worth more than any of them.

**It is concentrated.** Of the squared error over the 70 scored shots, in the loss metric:

| | share of the error |
|---|---|
| worst 1 shot | **23.2%** |
| worst 2 | 32.8% |
| worst 10 (14% of the fold) | 54.1% |

**It is not extrapolation.** The two worst shots have `|z|` maxima of 2.8 and 1.8 against a fold
median of 2.3, and the four most extrapolated shots (`|z| = 4`) are not among the worst. Their
inputs are ordinary.

**It is not a moment in the discharge.** The worst 5% of frames carry 8% of a bad shot's error —
near-uniform. The model is not failing at a transition; it is wrong evenly, all shot long.

**It is a constant.** Splitting each shot's coefficient error into its mean over frames and the
rest: **45.9% of the fold's error is the per-shot mean**, and 85.4% of the worst shot's. Remove
every shot's constant and the fold's metric R² goes 0.9594 → 0.9780. The worst shot goes from
0.7435 to 0.9626 — it is not a hard shot, it is a displaced one.

**And that constant is a shaped map, not a gauge.** Only 1.7% of it lies along "add a constant to
every pixel"; 98.3% is spatially structured. So it is not `ψ`'s additive freedom — it is a real
poloidal field, present for the whole discharge, that the instantaneous currents do not determine.

That description matches vessel eddy currents: they are driven by the *history* of the coil
currents, they persist on the vessel's L/R time, and they are structurally absent from the shipped
signals — `coil_field` cannot compute them because the vessel geometry is not in the dataset.

**And that explanation was then tested, and failed.** 252 per-shot summaries of the current history
— means, spreads, endpoints, mean `|dI/dt|`, and one-pole high passes at τ = 5, 20 and 50 ms —
regressed on the constant, fitted on 400 validation shots and evaluated on the scored fold, explain
**0.008** of it against 0.003 for predicting the fitting shots' mean. The ridge path is monotone
towards that constant model, so this is not a regression that lacked capacity: there is no signal
in the history to find. Whatever the shaped per-shot field is, the coil currents' past does not
say.

The honest limits of that test: 252 features on 400 shots overfits badly at small α (−1.7), and a
real eddy model would need the vessel geometry the dataset does not ship. But it is enough to stop
the derivative features from being built on the strength of a story. What remains possible is that
the constant belongs to the *labels* — EFIT is itself a reconstruction with its own priors and
per-shot settings — in which case it is irreducible from these inputs by anyone.

It also corrects an entry from 08-11. "The error is a per-shot flux offset" was recorded as refuted
at 21%, but that measured one additive scalar on ψ; this measures a constant vector in the
50-dimensional coefficient space, a strictly larger family, and at production it is 46%.

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
