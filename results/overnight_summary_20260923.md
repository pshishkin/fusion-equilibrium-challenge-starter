# DIII-D overnight experiment summary — 2026-09-22 UTC

All S values below are the vendored scorer's native six-decimal results on the **same 70 unseen
salt-0 shots** (14,830 frames) unless a row explicitly says otherwise. The ordered filename-list
SHA-256 is `6e2486e24cb5b31b1e3883f70b2ca514045963fb26cd27cf7acb3990f953cd4f`;
the original compact scorer JSON SHA-256 is
`4da58a6fed12c3f07051dfcee853b5b928240fd563f2c890fd59c163b1927630`.
Production fits used 0.80/1.0 train, 0.19/1.0 validation, 0.01 score, split salt 0 and one
seed per candidate. The separate fixed-step refits used 0.99/1.0 train and zero validation.
`D_LCFS` in this report is the **distance**, so lower is better.

## Best observed local diagnostic, with every map member

The recovered August five-member artifact scored **0.994478**. Adding the previously screened
`cons_gain=2` MLP, `env_weight=1` MLP, cosine sequence, constant-rate two-model refit, and cosine
sequence refit gives eleven equal-weight **map** members. The full eleven-member blend scored
**0.995046**. Taking q95/βN bit for bit from the prior eight-member blend while retaining the
eleven maps scored **0.995080**, the largest observed local S in this session. This scalar-block
choice and all additions were explored on an already reused 70-shot tail, so **0.995080 is a
development diagnostic, not independent confirmation or a leaderboard result**.

| Map member | Individual S | Map weight | q95/βN weight in the best observed diagnostic |
|---|---:|---:|---:|
| Recovered old `mlp` | 0.993140 | 1/11 | 1/8 |
| Recovered old `mlp1` | 0.991943 | 1/11 | 1/8 |
| Recovered old `mlp2` | 0.990809 | 1/11 | 1/8 |
| Recovered old `mlp3` | 0.991653 | 1/11 | 1/8 |
| Recovered old `seq` | 0.992175 | 1/11 | 1/8 |
| `cons_gain=2` MLP | 0.992381 | 1/11 | 1/8 |
| `env_weight=1` MLP | 0.992519 | 1/11 | 1/8 |
| 80%-train cosine sequence | 0.992205 | 1/11 | 1/8 |
| 99%-train constant-rate refit MLP | 0.992006 | 1/11 | excluded |
| 99%-train constant-rate refit sequence | 0.990179 | 1/11 | excluded |
| 99%-train cosine sequence refit | 0.992808 | 1/11 | excluded |

The five old members were predicted from the artifact's original inference source and rescored
individually once; no old member was retrained. Their complete scorer fields, files, source hashes
and timing are in [the member record](historical_five_members_20260922.json). The recovered old
artifact also contains a fitted ridge control, which has zero weight in this five-member ensemble.
The assembled blend's fixed weights and exact map/scalar identity audit are in
[the eleven-member record](historical_eleven_blocks_20260922.json).

| Local collection | S | R²ψ | R² q95/βN | D_LCFS ↓ | Consistency |
|---|---:|---:|---:|---:|---:|
| Compact MLP + GRU control | 0.993385 | 0.999753 | 0.995598 | 0.011106 | 0.976461 |
| Recovered old five | 0.994478 | 0.999836 | 0.996681 | 0.009756 | 0.980209 |
| Old five + three selected candidates (eight maps) | 0.994872 | 0.999845 | 0.997165 | 0.009452 | 0.981641 |
| Full eleven-member blend | 0.995046 | 0.999855 | 0.996934 | 0.009378 | 0.982616 |
| Eleven maps + prior-eight q95/βN block | **0.995080** | **0.999855** | **0.997165** | **0.009378** | **0.982616** |

The recovered August artifact is **not** a verified reconstruction of the September submitted
six-salt collection: its original submission artifacts, weights and hashes are unavailable.
The two original compact members scored 0.992149 (MLP) and 0.990514 (GRU) alone on these same
shots; their 50/50 native ensemble is the first row above.

## What each new screen changed

For compact replacements, the matched decoded MLP+GRU control is **0.993386**. It differs by
+0.000001 from the original native compact ensemble because decoded-map averaging changes the
floating-point operation order; [the identity check](geometry_identity_20260922.json) reproduces
each route exactly. All candidate/mixture comparisons used the same 70 filenames and explicit
0.5/0.5 weights.

| New member or treatment | Individual S | Compact replacement S | ΔS vs matched control | Evidence |
|---|---:|---:|---:|---|
| `cons_gain=2` MLP | 0.992381 | 0.993439 | +0.000053 | [geometry screen](geometry_loss_screen_20260922.json) |
| `cons_gain=4` MLP | 0.991828 | 0.993123 | −0.000263 | [geometry screen](geometry_loss_screen_20260922.json) |
| `env_weight=1` MLP | 0.992519 | 0.993530 | +0.000144 | [geometry screen](geometry_loss_screen_20260922.json) |
| `field_weight=0.3` MLP | 0.991210 | 0.993204 | −0.000182 | [geometry screen](geometry_loss_screen_20260922.json) |
| `ECOILA` ±5 ms MLP | 0.991120 | 0.992938 | −0.000448 | [fast-input screen](fast_signal_screen_20260922.json) |
| `ECOILA` ±5 ms + corrected Ip ±20 ms MLP | 0.991114 | 0.993323 | −0.000063 | [fast-input screen](fast_signal_screen_20260922.json) |
| 80%-train cosine sequence | 0.992205 | 0.993692 | +0.000306 | [cosine screen](sequence_cosine_screen_20260922.json) |
| Temporal-convolution sequence | 0.990636 | 0.993279 | −0.000107 | [convolution screen](seq_conv_screen_20260922.json) |

The fresh fast-input control reproduced **every** original compact MLP scorer field at S
0.992149. B11's new pooled form is a different MLP control, so its valid treatment contrast is
**conditioned minus pooled**, with the original sequence held fixed: MLP alone 0.992075 versus
0.991412 (**+0.000663**), but compact mixtures 0.993294 versus 0.993584 (**−0.000290**).
Fixed old-five additions were 0.994523 versus 0.994586 (**−0.000063**). The shared probes,
global target scaler, PCA and input transforms were identical across B11 arms; only training-loss
mode differed. [B11 native record](position_metric_screen_20260922.json),
[fixed old-five additions](historical_five_position_metric_20260922.json).

The constant-rate two-model 99%-train refit scored **0.992860** alone against compact 0.993385,
but a 50/50 blend of the *complete* two-model artifacts scored **0.994032**. Its 5:2 addition
to old five scored 0.994707. The separately preregistered cosine-sequence fixed-step refit
scored 0.992808 alone and 0.994824 as a fixed 5:1 old-five addition. These are different
replacement and addition questions, not extra weights selected for the best eleven-map recipe.
[Constant-rate refit](refit_production_20260922.json),
[cosine refit](seq_cosine_refit_20260922.json).

Every fixed old-five addition was also measured separately. These are descriptive same-tail
diagnostics; a member's solo score, compact replacement effect and addition to the old five are
not interchangeable.

| Added to recovered old five | Fixed artifact/member weights | S | ΔS vs old five |
|---|---:|---:|---:|
| `cons_gain=2` MLP | 5:1 | 0.994690 | +0.000212 |
| `env_weight=1` MLP | 5:1 | 0.994585 | +0.000107 |
| 80%-train cosine sequence | 5:1 | 0.994688 | +0.000210 |
| Temporal-convolution sequence | 5:1 | 0.994615 | +0.000137 |
| Constant-rate two-model refit | 5:2 | 0.994707 | +0.000229 |
| Cosine sequence fixed-step refit | 5:1 | 0.994824 | +0.000346 |
| B11 pooled-control MLP | 5:1 | 0.994586 | +0.000108 |
| B11 position-conditioned MLP | 5:1 | 0.994523 | +0.000045 |

Native fields and frozen prediction hashes are in the
[geometry](historical_five_geometry_20260922.json),
[cosine](historical_five_cosine_20260922.json),
[convolution](historical_five_conv_20260922.json),
[two-model refit](historical_five_refit_20260922.json),
[cosine refit](historical_five_cosine_refit_20260922.json), and
[B11](historical_five_position_metric_20260922.json) records.

Input-only and mechanism screens are distinct from score gains. The
[fast-magnetics audit](../experiments/fast-signal-diagnostic.md) established available raw
windows, but the two scored additions above did not improve the compact mixture. The
[D8 whole-shot linear aggregate gate](d8_whole_shot_20260922.json) failed. The
[B11 shared-denominator Jacobian direction gate](b11_shared_variance_20260922.json) passed,
but its subsequent conditioned training loss reduced both tested ensemble comparisons.

## Observed time by stage

`Train preparation` below is the native train timer minus the fit-only durations printed by each
model, so it is approximate. `Native score` is the exact scorer timer in each campaign; `child`
is the end-to-end campaign worker wall, including small startup gaps. Separate matched comparisons
are additional scorer passes. CPU and GPU work overlapped in several rows, so **do not sum rows to
infer a wall-clock total**.

| Experiment | Train preparation | Model fits only | Native score | Child wall | Extra matched comparison |
|---|---:|---:|---:|---:|---:|
| Four geometry MLP fits | ~8m39s | 19m18s | 3m31s | 31m44s | 7m02s total |
| Cosine sequence | ~2m02s | 22m55s | 0m59s | 26m00s | 1m51s |
| Constant-rate two-model refit | ~2m45s | 14m37s | 1m34s | 19m01s | 1m33s |
| Cosine sequence refit | ~2m36s | 33m21s | 0m59s | 37m00s | old-five addition scored separately |
| Fast control + two MLPs | ~8m43s | 11m26s | 2m41s | 23m02s | 3m45s total |
| Temporal-convolution sequence | ~2m14s | 6m28s | 0m55s | 9m40s | 1m46s |
| B11 pooled + conditioned MLPs | ~6m08s | 15m00s | 1m52s | 23m11s | 3m35s total |

The fourteen new model fits in this table consumed **2h03m04s of fit-only time**. The fast
campaign also needed a fresh auxiliary-target build (**19m31s**) because its scorer producer
changed; A/B full-shot sidecars took **33.70/39.90 s**. The constant-rate and cosine refits
reused a verified 6971-shot auxiliary cache composed in **4.50 s**, not a fresh target
extraction. [Fast cache hashes and timings](fast_cache_prewarm_20260922.json),
[refit cache composition](refit_cache_composition_20260922.json).
The two B11 fixed old-five additions each needed about 26 s of candidate prediction, 8.7 s
of decoded blending and 100 s of native CPU4 scoring; their exact concurrent walls are in
[the addition record](historical_five_position_metric_20260922.json).

The old-five individual table involved **no training**: prediction wall **1m22s** and five
separate CPU4 native scorer walls **8m12s** total. The DIII-D public candidate took **22m35s**
predicting six sources, **2m39s** assembling the final NPZ and **0m57s** validating it, total
**26m20s**. Its CPU4 work overlapped B11 GPU fitting, and the cosine sequence refit GPU fit
overlapped the 24-worker fast-cache preparation; those fit and build walls are not isolated
throughput benchmarks. The original compact control's **58m31s cold run**, including its first
target/cache construction, is a historical measurement from
[the compact control record](compact_production_20260922.json). Its roughly 19.8-minute warm
repeat was an estimate, not a second complete fit. A newly frozen CUDA smoke in
[the final CI record](final_ci_20260923.json) passed 39 tests and the unchanged 14-shot
correctness score; its worker wall was **1m29s**.
After the public builder/tests were added, Ruff, mypy on 83 source files and **41** tests passed.
All 114 frozen CI source files were still byte-identical to the working tree, so no duplicate
CUDA smoke fit was needed.
From the first new geometry fit's log start at 18:15:10 UTC to the last old-member score's
completion at 22:12:24 UTC, the observed execution window was about **3h57m**. That window
excludes earlier preparation, and multiple CPU and GPU stages overlapped within it.

## Float32 export and public candidate

The previous submission builder cast `psirz` to float16. Quantizing only that field on the
unchanged 70-shot predictions moved old five **0.994478→0.994100** (−0.000378) and prior eight
**0.994872→0.994475** (−0.000397), mainly through LCFS and Consistency. These measurements do
**not** include a float16 test of the eleven-map block choice. The corrected local/stream writer,
MAST/per-salt cache versioning and one-cast float32 blender passed a 210-array bit-exact old-five
round-trip, restoring the original native scorer JSON. [Precision audit](float16_export_audit_20260922.json),
[writer parity](float32_export_parity_20260922.json).

The best observed eleven-map/prior-eight-scalar recipe has been built **locally for DIII-D public
inputs only** at [diii_d_public_test.npz](../runs/public-diii-d-eleven-eight-20260923/submission/diii_d_public_test.npz).
It holds 874 shots, 193,359 frames and 2,622 float32 arrays; file size is 2,967,287,097 bytes,
SHA-256 `7b39cdc94ec8b95b2c89dfbee4c0fcd3db4898e2671ac3e0ac136efed5b5d704`.
The existing local validator passed all 874 shots. A 70-shot dry run of the same streaming
assembler reproduced every already scored map and scalar array bit for bit.
[Build/input/validation provenance](public_diii_d_candidate_20260923.json).

There is **no public-test score**, MAST public-test output, pointer ZIP, upload or push in this
work. The September six-salt submitted collection still cannot be reproduced from the recovered
August artifact without its original weights/artifacts/hashes. None of these same-tail screens
provides an independent estimate of top-three leaderboard performance.
