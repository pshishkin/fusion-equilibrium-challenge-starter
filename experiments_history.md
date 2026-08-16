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
| 08-14 | where the MLP's epoch actually goes, on a V100 | 44→512→512→52 at batch 512 is ~50 MFLOP against 15.7 TFLOP/s; four seeds batched cost 0.186 s/epoch against 0.200 s for one | measured — the loop is bound by kernel LAUNCHES, so the GPU was idle and the fourth seed is nearly free |
| 08-14 | does batching the seeds compute the same thing? | forward 6.9e-7, every gradient ≤ 8.5e-7 against four separate nets at initialisation | measured — float32 round-off and no more; see "Bit-identity is the wrong test" below |
| 08-14 | the four ensemble seeds as ONE vectorized model, on CUDA | quality S **0.9639 → 0.9639** with ridge identical to four decimals, fit **803.5 s → 61.3 s** (13.1x); on CPU it still pays 1.36x at production shapes | **built, measured, then removed** — see below |
| 08-14 | `device: cuda` for the MLP | paired on `make quality` (0.60/0.1 0.15/0.1 0.01, one net, salt 0, seed 0): S **0.9873 GPU against 0.9878 CPU**, ΔS −0.0005 = **0.4σ**, and `ridge` identical to four decimals. Fit **1037.6 s → 189.0 s** (5.5x), whole run 19:42 → 5:33 | **kept** — same score within noise, and the deterministic control says nothing else moved |
| 08-14 | do the two devices stop at the same epoch? | GPU 626 epochs keeping 525, CPU 704 keeping 603, same data and seed | measured — they do NOT, and should not be expected to: Adam turns float round-off into a different realisation of the same fit within a few dozen steps |
| 08-14 | is README's production table still the configuration it describes? | its `ridge` row says **0.7022**, which it-10 records as the PRE-Thomson value (Thomson took ridge to 0.7562); measured now, ridge is **0.7615** on the same command | the table was stale — ridge is deterministic, so it cannot disagree by chance, and the row had never been re-measured after a change it-10 records as kept |
| 08-14 | does batch size move the epoch time? | 183 batches 0.307 s, 91 → 0.152, 45 → 0.080, 22 → 0.033, 11 → 0.020 — time tracks the NUMBER of batches, not their size | measured; a candidate in ideas.md, NOT a free speed-up — it changes the optimisation |
| 08-14 | **grid E** more data: 0.60/0.1 → 0.80/0.1 → 0.80/0.2 → 0.80/1.0, with patience and the epoch ceiling rescaled per arm to hold the step budget at 18400 | against a measured `fast` control of 0.9885 on salt 0: **+0.0000 / +0.0020 / +0.0025**. The frames axis flattens hard — 0.1→0.2 is +0.0020 and 0.2→1.0 is only +0.0005 for 2.3x the wall clock and 51.9 GiB of RSS | **0.80/0.2 is the knee.** Five times more frames beyond it buys nothing measurable |
| 08-14 | **grid F, the confirmation**: 0.80/0.2 against 0.60/0.1 on salts 3 and 4 | **+0.0048 and +0.0025, mean +0.0037**, signs agree, for one extra minute of wall clock | **ACCEPTED** — and note the gain is LARGER on the salts nothing selected against (+0.0037) than on the selection salt (+0.0020), which is the opposite of selection bias. The largest confirmed gain in this fork |
| 08-14 | **grid G**: is the frames axis really flat past 0.2? 0.80/1.0 on salts 3 and 4 | the 0.2→1.0 increment is **+0.0005 / −0.0002 / +0.0025** on salts 0 / 3 / 4 — mean **+0.0009**, signs disagree, 1.2 sigma on the standard error of three | **not accepted, and not refuted either.** The effect is below what three runs resolve; settling it at 2 sigma needs ~8 salts, about 90 min. Left at 0.2 because the possible +0.0009 costs 2.2x the wall clock and 2x the memory, where the step TO 0.2 bought +0.0037 for one extra minute |
| 08-14 | did the frames-beat-shots prediction hold? | predicted shots would buy more, from the 08-12 result at a fixed ROW budget. Measured: shots 0.60→0.80 gave +0.0000, frames 0.1→0.2 gave +0.0020 | **prediction wrong, and recorded as wrong.** The 08-12 measurement traded frames AGAINST shots at a fixed budget; this one adds both, which is a different question |
| 08-14 | **grid D, the confirmation**: the two quality candidates re-measured on salts 3 and 4, which nothing selected against | `cos600` +0.0013 on salt 0 → **+0.0001 / −0.0014**, mean −0.0007. `deep3` +0.0012 → **−0.0014 / +0.0008**, mean −0.0003. Both have OPPOSITE signs on the two salts | **both refuted.** Not "fell short" — they went negative. Opposite signs across folds is the signature of noise, and the pre-registered arithmetic said this would happen: the observed best (+0.0013) was already below what selection alone produces over thirty arms (+0.0029) |
| 08-14 | **grid D**: batch 4096 with a sqrt-scaled rate AND gelu, composed | S **+0.0007 / −0.0005**, mean **+0.0001**; fit **2.29x and 2.31x** faster on the two salts | **kept** — parity to within a tenth of a sigma, and the two accelerators compose as their mechanisms predict: 1.89x from the batch, 1.61x from gelu's faster convergence, 2.30x together |
| 08-14 | **grid B** batch size, with patience AND the epoch ceiling rescaled so every arm sees the same 18400 steps of leash and 368000 of budget | 256 → +0.0009, 512 → baseline, 1024 → −0.0011, 2048 → +0.0001, **4096 → +0.0002 at 1.89x the speed**. All five inside the 0.0013 sigma | **score is insensitive to batch size over 256-4096**, and the time falls almost twofold. The speed result of the day |
| 08-14 | sqrt vs linear rate scaling with the batch | batch 4096 at 2.8e-3 (sqrt) gives +0.0002; the same batch at 8e-3 (linear) gives **−0.0047** | sqrt is right and linear is wrong here, by four sigma with an unambiguous sign |
| 08-14 | **grid C** weight_decay 1e-5 / 1e-4 / 1e-3 | +0.0009, −0.0009, **−0.0123**; monotone in strength | refuted — only the near-off setting is non-negative |
| 08-14 | **grid C** dropout 0.1 / 0.2, at patience 100 and 300 | −0.0015, −0.0050, −0.0008, −0.0061; monotone in strength at both patiences | refuted, and the longer patience did not rescue it |
| 08-14 | **the pre-registered prediction, tested**: do dropout and weight decay close the gap without moving validation? | train/val gap goes 3.39x (baseline) → 1.90x (BN) → 1.19x (wd 1e-3) → **0.56x** (dropout 0.2), and validation goes 0.0650 → 0.0662 → 0.1511 → 0.0965 | **confirmed, and worse than predicted.** Three independent regularisers, one answer, monotone in strength: the model is limited by BIAS, not variance, and the 3.4x gap is not overfitting worth removing |
| 08-14 | **grid C** gelu / silu instead of relu — is the dying-ReLU story real? | **+0.0000 and +0.0002**, the agreeing sign that was declared the criterion in advance | refuted. Both smooth activations land on exactly zero — but they converge in 392 and 425 epochs against 626, which is **1.6x of free speed** |
| 08-14 | **grid C** tanh in the hidden layers | −0.0031 | refuted, as written down before the run: it saturates on BOTH sides, so it cures dying units less well than relu, not better |
| 08-14 | **grid C** a third hidden layer, [512, 512, 512] | **+0.0012** — the best thing in grid C and second-best of all thirty runs | at the threshold, not past it. Note it-11 refuted extra WIDTH; depth behaves differently, which is what the bias-limited diagnosis predicts |
| 08-14 | **what thirty screened configurations are worth, honestly** | best result over grids A+B+C is **+0.0013**, against an expected maximum from selection alone of σ·sqrt(2 ln n) = **+0.0029** over the ~12 plausibly-neutral arms and +0.0034 over all thirty | **nothing here has been shown to help.** The observed best is SMALLER than pure selection would be expected to produce, which is the strongest available evidence that these knobs do nothing |
| 08-14 | **grid A** batch norm x learning rate, jointly, on `make quality` (one net, salt 0) | BN peaks at lr 3e-3 with **0.9873 against the baseline's 0.9873**, and falls away either side (1e-3 0.9866, 1e-2 0.9861, 3e-2 0.9816); it costs 30-38% more per epoch | **refuted at its own best rate** — the strong form. Measuring it at the baseline's rate alone would have called it harmful, which is a different and wrong claim |
| 08-14 | is the learning rate itself already right? | 3e-4 gives 0.9858, **1e-3 gives 0.9873**, 3e-3 gives 0.9830 — unimodal, peak on the value the fork has always used | measured; the rate was never tuned and is nonetheless optimal. Axis closed |
| 08-14 | does BN buy learning-rate robustness? | lr 3e-3 costs −0.0043 without BN and 0.0000 with it | measured — BN delivers exactly what it promises, and here that is worth nothing, because 1e-3 was already the optimum |
| 08-14 | **cosine** LR decay to 0.01x over the epoch ceiling | **0.9882 (+0.0009)** and the fit drops 191.2 s → **161.1 s**, 626 → 533 epochs | +0.0009 is BELOW the pre-registered 0.0013, so the score is not confirmed. Carried to the confirmation phase on the SPEED argument instead, which is a different criterion and is labelled as one |
| 08-14 | cosine on top of batch norm | 0.9879 against 0.9882 for cosine alone, at 228.3 s against 161.1 | BN adds nothing over the schedule either, and costs 42% more time |
| 08-14 | did anything else drift across the nine grid-A runs? | `ridge` is **0.7615 in all nine**, to four decimals | the deterministic control held — between configurations, only what was set changed |
| 08-14 | **A16**, the training loop recounted in STEPS: `max_steps` / `patience_steps` / `eval_every_steps` / `lr_t_max_steps`, no epoch boundary left in it | acceptance on two salts against the recorded controls: salt 0 **0.9906 against 0.9905**, salt 3 **0.9886 against 0.9890** — +0.0001 and −0.0004, both a third of a sigma | **accepted.** Not a result, a unit change: the shares→patience table in params.yaml is deleted and a grid can now move the data without silently moving the leash with it |
| 08-14 | **grid H**, the frames axis settled properly: 0.80/1.0 against 0.80/0.2 on EIGHT salts, with the step budget identical by construction and the PCA's sample held fixed | **+0.0001 / +0.0012 / +0.0032 / +0.0024 / +0.0009 / +0.0017 / +0.0024 / +0.0022** — mean **+0.0018**, every sign positive, standard error 0.0004, **t = +5.02**. Costs 7.6 min against 4.7 and 178 s of fit against 102 | **ACCEPTED, and it overturns grid G.** G read the same axis as flat at 1.2 sigma — but G gave the 1.0 arm `pca_frame_share: 0.1` and the 0.2 arm 1.0, so its PCA was estimated from half the sample on one side. Holding that fixed turns "unresolved" into five sigma. A confound in the CONTROL, not in the treatment |
| 08-15 | **submitted** the stack: every frame, gradient clipping, poloidal derivatives, four seeds and the sequence model | leaderboard **0.9932, first place**, against 0.9896 and 4th for the previous submission. Local salt 0 was 0.9945 | **+0.0036 on the board against +0.0031 locally — 116% transferred**, where the previous submission managed 91%, and the local-to-board gap narrowed 0.0018 → 0.0013. Over 100% is noise around 100%; the result is that a day of heavy selection cost nothing, which is what confirming on unseen salts is for |
| 08-15 | **the pair confirmed on salts nothing selected against** — mlp+seq at 0.80/1.0, gaps off | salt 3 **+0.0002**, salt 4 **+0.0012** against +0.0011 on the selection salt; both signs positive, mean of the unseen **+0.0007** | **accepted.** Smaller on the unseen folds than on the selected one, which is what selection bias looks like and is why the rule exists — but positive on both, so it survives its own test |
| 08-15 | how salt-sensitive the sequence model is, on its own | alone it scores 0.9922 / 0.9874 / 0.9900 on salts 0 / 3 / 4 — a spread of **0.0048** where the MLP's is 0.0016 | it trains on 5633 SEQUENCES, not 1.25M rows, so which shots it draws matters three times as much. That is the cost of the unit being the shot, and it is why its contribution to the pair swings from +0.0002 to +0.0012 |
| 08-15 | **splitting the output vector between models** — the map from one, q95 and betaN from another | `psi=mlp+seq,qb=mlp` gives **+0.0010 / +0.0005 / +0.0010** against the single MLP, mean of the unseen **+0.0008**. The REVERSE, `psi=mlp,qb=mlp+seq`, gives +0.0001 / −0.0003 / +0.0002 — **flat** | the control did its job: the sequence model's whole contribution is in the FLUX MAP and none of it is in the two directly regressed scalars. So taking the scalars from the MLP is mechanism, not tuning — and it costs a rescore, since both members are already fitted |
| 08-15 | is the arithmetic on the printed tables trustworthy? | the composite is linear in its four terms and each term reads one block, so a split is predictable from two tables. Rescored exactly on salt 4: **0.9927 / 0.9925 / 0.9917 / 0.9915** against the predicted same four | it is — checked rather than assumed, because the rounding at four decimals is the same size as the effect being read |
| 08-15 | **every combination of three families**, from ONE fit — mlp, catboost, seq and their averages at 0.80/1.0, salt 0 | **mlp+seq 0.9934** and mlp+catboost+seq 0.9934; mlp+catboost 0.9922, catboost+seq 0.9920; alone: mlp 0.9919, seq 0.9914, catboost 0.9869 | **the pair is the answer, and CatBoost is not in it** — all three score exactly what two do. The model that LOSES alone adds the most: seq is 0.0005 below the MLP by itself and worth +0.0015 in the average, all of it through Consistency (0.9697 → 0.9762). Decorrelated errors are what an ensemble is for, and a structurally different model is where they come from |
| 08-15 | CatBoost on the GPU, the cost that got it dropped | **1053 s** against 3829 s on the CPU, and `best iteration 7999 of 8000` again — still undertrained at its ceiling | the cost argument is gone and the score argument stands: +0.0003 on top of the MLP, nothing on top of the pair. Its `iterations` comment still calls 8000 "far above anything the fit should need", which is false at this data scale |
| 08-15 | **`frame_gaps` costs the per-frame MLP** — measured for free, as the same configuration with and without | 0.9919 with the two gap columns against **0.9931** without, salt 0, everything else identical and the pipeline deterministic | **−0.0012, where the comment predicted zero.** 97.86% of the steps are 20 ms, so the column is nearly constant — and a nearly constant input is not free: it is two more directions for the first layer to fit noise in. The pair was measured WITH it, so the pair's own number is a lower bound |
| 08-15 | **the sequence model, trained to convergence** — 200000-step ceiling, patience ended it at 55500 with the best at 45500, 41 minutes | **S 0.9914 against the MLP's 0.9931** on the same salt, shares and features — and worse on EVERY term: R2_qb 0.9914 against 0.9954, Consistency 0.9706 against 0.9755, 1−D_LCFS 0.9872 against 0.9885. But its validation **loss is LOWER**, 0.033245 against 0.0335, and the shot moves its prediction by 31.6% | **refuted as a replacement for the MLP, and diagnosed rather than merely lost.** It optimises the objective it was given better and scores worse, which is the same gap A13 is about — and the mechanism is one this fork already measured: the prediction is HALF AS ROUGH as the truth (0.021 against 0.039), so smoothing was refuted in 08-13. A bidirectional recurrence smooths further, and the functionals that read the map's geometry charge for it |
| 08-15 | **the sequence model, first arm** — bidirectional GRU over the shot, residual on the per-frame MLP | 0.9911 against the MLP's 0.9931 on the same salt, shares and features. But **best step 40000 OF 40000**, still improving, and the correction branch had grown to **32.1%** of the per-frame prediction | **not a result, a budget.** The screen was stopped after one arm rather than spending 2.4 hours on five more numbers about the ceiling. The budget was calibrated to match the MLP in PASSES over the data, and that was the wrong analogy: a pass here is 176 steps against 305, and a gradient averaged over a whole sequence moves less per step |
| 08-15 | **grid K**, the derivatives confirmed on the two salts nothing selected against | salt 3 **+0.0023**, salt 4 **+0.0013**, mean **+0.0018** — against +0.0010 on the selection salt | **ACCEPTED, default.** Larger on the unseen salts than on the selected one, which is the opposite of selection bias, and the second time that has happened in this fork (the first was the frames axis) |
| 08-15 | **C2, the representation floor** — ground truth pushed through the artifact's own 50 components and scored (`evaluate.py --mode basis`), same 70 shots as production | ceiling **S = 0.9990** against the model's 0.9945. Per term: R2_psi **1.0000** vs 0.9998, 1−D_LCFS **0.9950** vs 0.9902, Consistency **0.9974** vs 0.9802 | **of the 0.0055 left, 0.0045 is reachable by fitting and 0.0010 is the basis.** The bottleneck is not the bottleneck: 50 PCA directions reproduce the geometry the scorer reads to within 0.0010 of S, where B9 was written on an assumed 0.0175 of headroom there |
| 08-16 | **does the feature-diversity axis keep paying?** — four more members, each different along one axis nothing else varies (coil-flux directions alongside the currents, a 40-direction basis, the raw derivative block alone, a Huber loss), plus A4, added to the confirmed three | **it saturates at three.** prod+driving+poloidal is 0.9950; adding a fourth gives 0.9950–0.9951 whichever fourth, and adding ALL FIVE gives **0.9951**. Alone the new members score inpboth 0.9937, pca40 0.9940, huber 0.9941, draw 0.9942, **a4 0.9944** against a 0.9940 control | **the gain is from having two or three genuinely different views, not from member count.** +0.0001 for four more fits is nothing, and it says the confirmed submission needs no rebuilding. Also the cheapest possible reading of A4: **+0.0004 alone**, so absorbing the validation window does help, below what one paired run resolves, and it adds nothing the other members were not already contributing |
| 08-16 | **the stack on salt 4** — the same four fits and two scorings | control **0.9931**; weight average alone **0.9932 (+0.0001)**; two feature sets **0.9946 (+0.0015)**; three **0.9949 (+0.0018)** | **ACCEPTED.** Averaging across feature sets is +0.0005 / +0.0009 / +0.0018 on salts 0 / 3 / 4 — positive everywhere, **mean +0.0014 on the two salts nothing was selected against**, and larger on the unseen folds than on the selection one for the fourth time in this fork. Above the 0.0013 a single paired run resolves. The weight average is +0.0003 / −0.0002 / +0.0001, mean **+0.0000**, and is dropped |
| 08-16 | **the stack on salt 3, which nothing was selected against** — control against the weight average alone, then two and three feature sets averaged as decoded maps, four seeds each | control **0.9934**; weight average alone **0.9932 (−0.0002)**; two feature sets **0.9940 (+0.0006)**; three **0.9943 (+0.0009)** | **the stack splits, and cleanly.** Averaging across FEATURE SETS confirms and comes back LARGER than on the fold it was found on (+0.0009 against +0.0005), which is the opposite of selection bias and the third time that has happened in this fork. The weight average does not confirm: +0.0003 on salt 0, −0.0002 here, i.e. zero. Since the treatment arms all carry it, the feature-set effect alone is about +0.0011 |
| 08-15 | **A20, a running average of the WEIGHTS** — kept alongside the live ones, evaluated beside them, the fit keeps whichever scores better. Production, salt 0, four seeds, decay 0.999 | ensemble **0.9943 against 0.9940**, but the individual models move far more: **0.9930 / 0.9927 / 0.9927 / 0.9929 against 0.9920 / 0.9916 / 0.9920 / 0.9927**, a mean of **+0.0008 each**. The averaged weights won **1537 / 1838 / 1766 / 1261** of roughly 1630 evaluations | **+0.0008 to one net and +0.0003 to four, and the gap IS the result.** The average removes the noise of one trajectory rattling in its basin; averaging four seeds removes much of the same noise a second time, so most of what it buys has already been bought. Winning 94% of the evaluations it was measured at means a fit keeping the live weights is keeping the worse of two things it already holds |
| 08-15 | **where the vessel integrals actually act** — the A19 driving arm decomposed onto the same 14830 frames as the control, four seeds each | total geometry cost **0.00529 → 0.00504**, and it is not spread evenly: the **last decile falls 10.0%** (0.001314 → 0.001183) and carries **52% of the whole gain**, while the first decile gets 1.5% worse. Per scalar the RMS residual moves **li −8.7%**, kappa −4.1%, volume −3.1%, against tri_bot +2.9% and R_axis +3.8% | **the mechanism, not just the number.** C1 found the last decile carrying 24% of the cost; C7 said it was coefficient error and not sensitivity; C6 measured that the sequence model built 1.8–2.4 s integrators and that its output reads them; A19 hands the same timescales over as eight columns and the gain lands in that decile and on `li` — the current-profile quantity whose evolution IS diffusion on that timescale. Four measurements, one story, and the scalars that improve are the expensive ones |
| 08-15 | **an ensemble across FEATURE SETS, not across seeds** — the production five averaged with the two A19 arms as decoded flux maps, all three fitted on the same split, salt 0, nothing retrained | **0.9950** for prod+driving+poloidal, 0.9949 for prod+driving, 0.9948 for prod+poloidal, against **0.9945** for production alone. Every term up: R2_psi 0.9999, R2_qb 0.9974 (+0.0007), 1−D_LCFS 0.9908 (+0.0006), Consistency 0.9819 (+0.0017) | **+0.0005 from three feature sets whose individual scores are 0.9945, 0.9944 and 0.9940** — the two arms that were flat or marginal ALONE are worth something TOGETHER, which is what decorrelated error looks like and is the same lesson as the sequence model (worst alone, most valuable in the average). A deterministic rescore, so there is no optimisation noise in it; the uncertainty is the 70-shot sample and the fact that three combinations were tried. Goes to two unseen salts |
| 08-15 | **A19 arm 2, the same integrals on the two signals the physics names** — ECOILA and plasma_current only, 8 new columns on 84, production, salt 0 | ensemble **0.9944 against the control's 0.9940, +0.0004**, and **all four seeds improve**: 0.9924 / 0.9922 / 0.9927 / 0.9925 against 0.9920 / 0.9916 / 0.9920 / 0.9927. Per term the gain is Consistency **+0.0012** and R2_qb **+0.0009**, D_LCFS flat. Validation loss 0.0295 against ~0.0315, the lowest of any arm run today | **below the 0.0013 a single paired run resolves, so not accepted — and the shape is exactly what was pre-registered.** The broad set was predicted to dilute and it did (arm 1, flat, 80 columns); the narrow set carries the same physics in 8 columns and moves every seed the same way. Four of four positive plus a lower loss is more than one number's worth of evidence, so this goes to two unseen salts rather than being dropped |
| 08-15 | **A19 arm 1, the vessel-state features on every poloidal signal** — four leaky integrals of dI/dt per signal at 20 / 100 / 500 / 2500 ms, 80 new columns on 84, production, salt 0 | ensemble **0.9940 against the control's 0.9940**. Per term 0.9998 / 0.9966 / 0.9901 / 0.9779 against 0.9998 / 0.9962 / 0.9901 / 0.9785 — R2_qb up 0.0004, Consistency down 0.0006, net zero. **But `ridge` moved 0.7651 → 0.7867, +0.0216** | **flat for the MLP and large for the linear model, which is the whole result.** The integrals carry real information — a linear map gains more from them than from anything else tried — and the MLP had already extracted it. That is not as surprising as it first looks: a coil's CURRENT is the integral of its own dI/dt, so the levels the net already sees are themselves accumulated history, and two hidden layers over 21 of them can build what an explicit EMA hands over. The missing state A22 is chasing is either already there in a form the net can use, or it is not of this form |
| 08-15 | **A13 proper — the composite decides, but only once the validation loss has been flat for 10000 steps.** Four seeds, production, salt 0, 400 monitor frames, against the same four seeds loss-stopped | **S 0.9940 against 0.9940.** Per term the treatment is 0.9998 / 0.9965 / 0.9899 / 0.9780 against the control's 0.9998 / 0.9962 / 0.9901 / 0.9785 — inside noise everywhere and marginally BEHIND on both geometry terms. And it is not for want of training: best step 84000 / 94000 / 98000 / 108000 against 58900 / 39150 / 80250 / 88550, over runs 35–100% longer | **refuted, and it answers its own confound.** The rule necessarily trains ~28400 steps past the loss peak where the control stops at 18400, so a third length-matched arm was pre-registered in case it won. It did not, so training longer bought nothing either — which is what the composite trace already said, flat from step 42000 on. The oldest open entry in the file, built, controlled and closed |
| 08-15 | **A13 with the composite deciding from step one** — production, salt 0, four MLPs stopped on the score itself over 150 validation frames | it stops EARLIER every time, and sometimes absurdly so. Best step against the loss-based control: **48000 / 46000 / 36000 / 16000** against **59250 / 50700 / 99650 / 61500**. `mlp2` recorded its best composite at step **2000**, the first evaluation it ever took, and needed 18000 more steps to match it | **killed before it finished, and the mechanism recorded instead of the score.** A 150-frame sample can flatter a barely-started net and the validation loss cannot — 0.0597 at step 2000 against an eventual 0.0324 is not a trained fit under any sampling. Selecting the argmax over ~34 evaluations of a FIXED small sample is ordinary selection bias, running inside a training loop where nothing was watching for it. Replaced by the gated rule, and the seq fit that would have cost 49 more minutes is identical in both arms |
| 08-15 | **the sequence model's weight in the ensemble, swept as a rescore** — ten weightings on a 70-shot SELECTION set of validation shots, then the best confirmed on the untouched tail | the selection curve is smooth and unimodal: 0.9925 at weight 0, 0.9929 at 0.11, **0.9930 at 0.20 (production)**, 0.9932 at 0.33 and 0.43, 0.9931 at 0.50, 0.9929 at 0.60, 0.9924 at 0.71, 0.9918 at 0.83, 0.9908 alone. On the TAIL: weight 0.20, 0.33 and 0.43 all score **0.9945** | **the flat optimum was already occupied.** +0.0002 on the selection set did not transfer at all, which is what +0.0002 against a 0.0013 resolution should do. The shape is worth keeping — a unimodal curve over ten points is real even where its peak is not resolvable — and it closes the global half of A23: one weight per member is already right |
| 08-15 | **A2, `calibrate_scalars: true`** at production, salt 0, against the recorded control | **S 0.9945 against 0.9945.** Per term +0.0005 on R2_qb, +0.0001 on 1−D_LCFS, −0.0001 on Consistency. Per scalar the seven move by ±0.0012 with no pattern: R_axis +0.0009, li +0.0006, tri_bot +0.0008, but **kappa −0.0010** and volume −0.0012 | **refuted, and the entry's own "open problem" paragraph called it.** The probe's ratios are 1.11–1.39 — near enough uniform that dividing by them is close to a uniform rescale of the Consistency block, which the target scaler then largely absorbs. The per-scalar moves are 1/20th of the per-scalar seed noise (0.018 on R_axis, 0.023 on Z_axis) and mean nothing individually; the composite is what resolves, and it moved 0.0000 against a resolution of 0.0013 |
| 08-15 | **C7, is an expensive frame under-weighted or simply harder?** — the per-frame Jacobian on 2998 frames, same probe step as the training loss, joined to C1's costs | sensitivity spreads **10.1× p99/median**, so B6's clause is not met — but Spearman(sensitivity, cost) is only **+0.209** and the quintile means are 0.74 / 0.79 / 0.65 / **0.56** / 2.25. By decile: ramp-up is **3.05× sensitivity and 0.51× implied error**, ramp-down is **0.98× sensitivity and 2.22× error** | **B6 was aimed the wrong way.** A Jacobian weight up-weights the first decile, which the model already predicts twice as well as average, and leaves the last decile — 24% of the geometry cost — exactly where it is. The two ends of a shot are expensive for opposite reasons and only one of them is a mis-weighting |
| 08-15 | **the last decile is not a representation problem** — the ceiling decomposed onto the same frames | the ceiling is FLAT across the shot: 0.000086 to 0.000120 per decile against a mean of 0.000100, and the last decile's basis share is **10.1%**, the LOWEST of the ten, against 21–33% in the middle | **the tail is not harder to represent**, so 27.2% of everything reachable sits in 10% of the frames. **Corrected the same day:** this was first written as also refuting EFIT label noise, and it does not — a noisy label is reconstructed by the basis exactly as well as a clean one, so a flat ceiling is what label noise would produce too. That branch is open |
| 08-15 | **what the sequence model actually buys, by region** — C1 run for `mlp` alone against the ensemble | the MLP loses 0.00603 of geometry cost, the ensemble 0.00490, a saving of **0.00113 — 18.3% overall, 24.9% of the first decile and 15.2% of the last** | **confounded, and superseded by the row below.** The ensemble is four seeds AND the sequence model, so this prices the ENSEMBLE by region and not the recurrence |
| 08-15 | **the same question with the seeds controlled** — `mlp+mlp1+mlp2+mlp3` against the five-member ensemble, same frames | the four seeds alone average 0.00700, averaging them gives 0.00529 (**saves 0.00171**), adding `seq` gives 0.00490 (**saves 0.00039**). By decile the recurrence's saving climbs monotonically: +2.4% at the start, 5–8% through the middle, **+14.5% at 0.8–0.9 and +10.1% at 0.9–1.0** — **34% of everything it buys is in the last decile, 52% in the last two** | **the earlier reading REVERSES.** Seed averaging does four fifths of the work and the recurrence one fifth, but that fifth is concentrated exactly where the memoryless models are worst. The one member carrying state helps most during ramp-down, which is what "the vessel current is largest and most transient there" predicts — evidence FOR the missing-state branch of A22, and it makes B5's unrun screen a live question rather than a debt |
| 08-15 | **C6, the memory the sequence model actually learned** — the update gate recomputed from the stored weights over 20 held-out shots, 256 channels per direction, no training and no scoring | **median tau 90.7 frames forward (1.81 s) and 122.0 backward (2.44 s)**; only 2.3% / 1.6% of channels sit under 2 frames, while **60.5% / 69.1% are over 50 frames (1 s+)**. Weighted by each channel's influence on the output (`delta`'s column norms) it is LONGER still — the top ten forward channels have a plain median tau of 212 frames — and Spearman(influence, tau) is +0.045 forward, +0.254 backward | **it is not a smoother.** The 08-13 refutation of temporal smoothing does not describe what this model built: the majority of its channels integrate over more than a second, the channels the output actually reads are the longest-memory ones, and a tau above ~373 frames simply means z stays near 1 for the whole shot — an integrator. That is the vessel-and-profile state A19 and A22 are chasing, learned rather than given |
| 08-15 | **C8, does the model know where it is wrong?** — the four seeds' per-frame disagreement against the ensemble's own per-frame cost, 14830 frames, nothing retrained | **Spearman +0.565**, against +0.209 for the Jacobian sensitivity. Monotone, unlike sensitivity: mean cost by quintile of disagreement is 0.20 / 0.32 / 0.52 / 0.99 / **2.97×**, the top fifth carrying **59.3%** of the cost. But by decile the disagreement is 1.55× at the start against a cost of 1.54×, and only **1.19× at the end against a cost of 2.61×** | **a free confidence signal that locates the loss 2.5x better than the metric's own linearisation** — and blind exactly where the money is. Seed residuals correlate +0.550 in the tail, so a majority of the error there is SHARED, and four seeds wrong the same way look confident. Disagreement measures variance and cannot see bias. Opens A23 (weights that vary, shrinkage), both without retraining |
| 08-15 | **is the tail bias or variance?** — pairwise correlation of the four seeds' per-frame residuals over the seven derived scalars | mean pairwise r = **+0.550 in the last decile** against **+0.615 in the middle** of the shot | **a majority of the tail's error is SHARED across seeds**, so it is bias of this model-and-feature class and more seeds cannot remove it — averaging is bounded to the ~45% that is independent. The tail is if anything slightly MORE variance-driven than the middle, not less. What this still cannot separate: a missing input from an unlearnable EFIT label |
| 08-15 | **C1, where the score is lost** — the same two geometry terms decomposed onto all 14830 frames on one additive scale (`diagnose_frames.py`) | the worst **1% of frames carry 22.4%** of the cost, the worst 5% carry 45.8%, the worst 10% carry 57.9%. By position: the LAST decile **24.1%** and the first 15.1%, the middle eight flat at 6–9%. By scalar: kappa **23.3%**, li **18.7%**, LCFS 18.4%, R_axis 11.3%, volume 10.8%, Z_axis 6.8%, the two triangularities 9.9% together | **the loss is heavy-tailed and the loss function does not know it.** A pooled R2 weights every frame the same; the ends of the shot cost 2.4x their share and one frame in a hundred costs 22x. That is B6's hypothesis measured directly, and its own refutation clause — "refuted if the per-frame variation is under ~2x" — is answered the other way |
| 08-15 | **grid J**, the five config-only arms of the pre-registered ten, at the stable default | control 0.9921. **derivatives=both/poloidal +0.0010**; n_pca 75 −0.0005; patience 6000 −0.0008; pyramid [1024,256] −0.0012; Thomson raw profiles −0.0012 | one positive of five, and it is a FEATURES arm — which is where the capacity result said to look. A10 screened the same configuration at 0.60/0.1 and got +0.0007 at best, below its floor: a feature that needs data to pay is exactly what a small screen refutes wrongly. Confirmation running |
| 08-15 | is the leash still wasteful now the fit is stable? | cutting `patience_steps` 18400 → 6000 costs **−0.0008** | it was wasteful at the unstable setting (60-70% of the fit ran past the best step) and it is not now: clipping lets the fit train 93000 steps and the best one arrives late. The measurement that A16 queued, answered — and the answer changed because the thing underneath it changed |
| 08-15 | **the fits are unstable at the production shares**, found while chasing a control that moved 0.0021 between two runs of the identical configuration | five repeats of one configuration give **0.9889 five times over**, best step 50000 in each — so there is no run-to-run noise at all. What differed: at step 6000 one run's training loss is **13.6** against unit-variance targets. Across every log at these shares, **15 fits of 28 blow up**, worst 67.9 | the learning rate — sqrt-scaled for batch 4096 and tuned at 0.60/0.1, a twenty-fifth of the frames — is past the edge of stability at this scale. Not noise in the measurement: a lottery inside it |
| 08-15 | **grid T**: what production should train at, paired on three salts | `grad_clip: 1.0` at the current rate **+0.0015 / +0.0007 / +0.0008, mean +0.0010, t = 3.97**, 0 blow-ups of 3, 93000 steps run. A quarter of the rate: **+0.0005 / +0.0013 / +0.0019, mean +0.0012, t = 3.04**, 0 of 3, but 170000 steps. The default blew up in 2 of 3 | **clipping accepted, default.** Same gain as the slower rate, tighter spread, half the wall clock — and it removes the instability that had been corrupting every comparison of the evening |
| 08-15 | **grid A15**, capacity and light dropout crossed: width 512/700 x depth 2/3 x dropout 0/0.02/0.1 at 0.80/1.0 | best cell +0.0012 against a selection floor of +0.0029 — but the MARGINAL means use all twelve runs: width 700-512 is **−0.0006 and −0.0011** at the two dropout levels where the design is complete, depth 3-2 is **−0.0001 and −0.0011**, dropout 0→0.02 is −0.0001 and 0.02→0.1 is **−0.0025** | **capacity refuted on both axes, and past 0.02 dropout hurts.** Reading the best of twelve would have said "+0.0012, promising"; reading the design says every axis is flat or negative |
| 08-15 | what that does to the bias-limited diagnosis | three regularisers had closed the train/val gap with validation flat, which is the signature of a model limited by bias — and MORE CAPACITY DOES NOT FIX IT | so the bias is not a capacity bias. The net can already represent more than the inputs determine, and the constraint is what a single frame's 46 numbers CONTAIN. That is an argument for the sequence model and for features, and against every further width |
| 08-15 | 700x700x700 at the default 2.8e-3 | train loss **1.44** at step 2000 — worse than predicting the mean of unit-variance targets — best at step 11650, then rising again to 0.152 by 30000. With dropout 0.02 the same architecture trains fine and scores 0.9911 | not a capacity result but a **divergence**: the rate was tuned for 512x512. Recorded as such and re-run at lower rates rather than counted as depth hurting by 0.0074 |
| 08-14 | why more frames pay, visibly | the best step moves from 21631 to 42518 on average (13350..30250 against 18800..75600) at an unchanged step budget | five times the data delays overfitting, so twice as many of the same steps stay useful. Not more optimisation — more of it that is worth doing |
| 08-14 | the EFIT frame clock, measured over 202 shots and 44729 steps | 97.86% of steps are 20 ms and every step is a multiple of it, but **77% of shots contain at least one dropped frame** (largest gap 2580 ms), and a shot is 2 to 373 frames long | settles the sequence model's design before it is written: train it at frames 1.0 only, and feed Δt in units of the base step, because 1.0 is still not a regular grid |

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

## The `make quality` sweep, pre-registered 2026-08-14

Written down BEFORE the first run, because the alternative is not a measurement. The screen is
`make quality` — 4225 shots, one net, salt 0 — at 5.5 min a run.

**Why pre-registration and not just a bigger sweep.** The single-net sigma is **0.0013** (08-14,
pooled over three salts). The expected maximum of N draws from that noise is about 2.25 sigma at
N = 50, i.e. **+0.0029** of pure selection — larger than any change this fork has ever accepted
(+0.0022, +0.0039, +0.0042, +0.0018). It-18 measured exactly this failure on this project: the
accepted stack showed +0.0115 on the three salts it was selected on and +0.0077 / +0.0057 on two
salts it had never seen, so about 40% was selection bias.

**Thresholds, fixed in advance:**

- a configuration becomes a *finalist* if it beats the salt-0 baseline by more than **0.0013** (1
  sigma) on the screen;
- a finalist is *accepted* only if, re-measured on salts **3 and 4** which nothing selected
  against, the mean gain exceeds **0.0013** AND both salts agree in sign;
- whatever survives gets one four-net production run before it goes near a submission;
- the screen number and the confirmation number are always reported together. The screen number
  alone is not a result.

**Grid A — batch norm and the learning rate, JOINTLY.** Batch norm changes the convergence enough
that measuring it at the rate tuned without it would test the rate, not the normalisation. Nine
runs: `batch_norm` off/on crossed with `learning_rate` 3e-4 / 1e-3 / 3e-3 / 1e-2 / 3e-2, plus the
cosine schedule on the best of each arm.

**Grid B — batch size and the rate**, for both goals at once: the loop is launch-bound, so epoch
time tracks the number of batches (183 -> 0.307 s, 45 -> 0.080, 11 -> 0.020). `patience` is
converted to a constant number of STEPS rather than left at 100 epochs, or the comparison measures
the leash: at batch 4096 an epoch is 22 steps against 183 now.

**Grid C — the cheap regularisers.** `weight_decay` is still 0.0 and the measured train/val gap on
this screen is **3.4x** (train 0.0192, val 0.0650 at epoch 625), so there is something to
regularise. `dropout`, and `gelu`/`silu` for the hidden layers.

**Declared in advance as a likely disappointment**, so it is not explained away later: the screen
fits ONE net and production averages FOUR. Dropout and batch norm work partly by reducing variance,
which is what seed-averaging already does, so a gain here can shrink in the production ensemble.

## Why batch norm gave nothing, and what it predicts — 2026-08-14

Batch norm was measured at four learning rates spanning 30x (1e-3, 3e-3, 1e-2, 3e-2) because it
changes convergence enough that testing it at the rate tuned without it would test the rate. The
curve is unimodal with its peak at 3e-3, and that peak is **exactly the baseline**: 0.9873 against
0.9873, for 30-38% more time per epoch.

The train/val curves say why, and it is not "batch norm did not work":

| at epoch 600 | train | val | gap |
|---|---|---|---|
| baseline | 0.0192 | 0.0650 | **3.4x** |
| BN, lr 1e-3 | 0.0375 | 0.0693 | **1.9x** |
| BN, lr 3e-3 | 0.0343 | 0.0662 | **1.9x** |

**It regularised hard and that bought nothing.** The gap closed from the wrong side: validation did
not move (0.0650 -> 0.0662) while the training loss got 1.8x worse. So the 3.4x gap was not the
constraint — the model is limited by bias, not by variance, which agrees with it-11 (capacity
exhausted, the constraint is the data) and with the nearest-neighbour ceiling of R² 0.97-0.997
against the 0.90-0.93 the model reaches.

Batch norm's other textbook effects DID show up and also failed to matter: the epoch-0 loss fell
1.83 -> 0.79, so it conditions the start; and it did not converge in fewer epochs (626 baseline
against 604 and 652). On two hidden layers the gradient path is short enough to need neither, and
the pipeline already standardises the features and scales the targets to unit variance, so both
ends of the net were well conditioned before batch norm saw them.

**Pre-registered prediction, written before grid C ran:** dropout and `weight_decay` should do the
same thing — close the train/val gap without moving validation. If either moves validation, this
explanation is wrong and has to be replaced rather than patched.

## What the thirty-eight-run architecture sweep settled — 2026-08-14

Grids A, B and C screened thirty configurations on salt 0; grid D re-measured the two survivors
plus the speed composition on salts 3 and 4, which the selection never saw.

**Nothing improves S.** Batch norm, dropout, weight decay, the learning rate, its schedule, four
activations, depth, and batch size from 256 to 4096 — the best screen result was +0.0013, and it
did not survive contact with an unseen fold. The single most useful number in the whole sweep is
that +0.0013 was already SMALLER than the +0.0029 that selection alone produces over that many
arms, so the screen was telling the truth before the confirmation ran.

**What did come out of it is 2.30x of speed at parity**, confirmed on two unseen salts to within
0.02x of each other: batch 4096 with a sqrt-scaled rate, plus gelu. And a diagnosis, from three
independent regularisers agreeing: the model is limited by **bias, not variance**. Every one of
them closed the train/val gap — 3.39x to 1.90x to 1.19x to 0.56x — and validation moved the wrong
way or not at all.

**Where that points.** Not at the architecture, which is now well covered: at the inputs, where the
nearest-neighbour ceiling is R² 0.97-0.997 against the 0.90-0.93 the model reaches, or at what the
loss asks for (A13). The one loose thread inside the architecture is A15's empty cell — capacity
and light regularisation together, which neither it-11 nor grid C tested.

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
