# DIII-D: current experiment status

Updated 2026-09-26 (Moscow). This is the current index, not another experiment history.
Historical explanations and negative results remain in `experiments_history.md`, `backlog.md`,
`ideas.md`, and `single_model_results.md`. Their old "running" labels are historical.

## Public submissions and iteration control

- **Final phase, 2026-10-19 (pages re-read): the package is NOT attached to Final submissions any
  more** ("Nothing extra is attached to a Final submission"): only the top five per challenge are
  asked for a reproduction package within 7 days after the close; a pipeline that "genuinely cannot
  fit" 24 A100 GPU-hours says so and gets a case-by-case check; a rebuild passes within 0.01 S.
  Best score across a team's (max 3) Final submissions counts. Private-test inputs:
  `ganatma/fusion-equilibrium-challenge-final` (the organizers' Final repo, the starter kit's
  FINAL_REPO_ID): 879 DIII-D shots (7.5 GB), 1205 MAST shots; shot_i in sorted (streaming) order as
  in Development; `data_fixes.py` (upstream, later) fixes the Ip axis (#6) — our own
  `align_ip_times` does the same and is what the models were trained with, so it is kept.
  Development closed 2026-10-18 without a compactA submission (no approval given in time).
  Running: full39e on the private DIII-D fold (`runs/final_predict.sh`), not published.
  **Bug caught before submitting (2026-10-19):** the first private prediction of full39e had GBM
  corrections 4.6x (coefficients) and 9x (q95) the public ones. Inputs were not the cause (Ip axis
  identical to the organizers' fix on 100% of shots, feature shift |z| 0.042 vs public 0.040, members
  agree better on private). Cause: full39e's stage was fitted on 242-wide inputs (5036 stage
  features); `thomson_chord` later made them 377 wide and CatBoost silently read the leading 5036 of
  5171 columns. Fix: `CatBoostStage2.predict` now raises on a width mismatch; `stack_predict
  --pre-chord` drops the chord block for stages fitted before it. Check: --pre-chord on 30 public
  shots reproduces the submitted full39e npz bit-exactly (max |diff| 0). full45e / compactA were
  fitted on 377-wide inputs and are unaffected. Private full39e re-running.
- **Final-phase rule 6 (read 2026-10-13 from the competition pages):** every Final submission
  must carry, AT SUBMISSION TIME, a public reproduction package (git repo at a pinned SHA on a
  public forge, or an archive with SHA256): source, pinned environment, the trained weights used,
  a from-scratch training entrypoint, an inference entrypoint, a per-machine data-provenance
  statement, and a declared training budget; **training must be reproducible within 24 GPU-hours
  on one 80 GB A100**. Organizers rebuild the top five per challenge. Measured from queue.log,
  full39e's 39 members cost ~115 wall hours on the RTX 3090 (CV folds for the stack 51 h, five
  full-data seeds per member 64 h) — about 4-5x over the cap as built. Per member (5 CV folds + one full seed, RTX 3090
  wall incl. data loading and the OOF dump): cheap mlps 0.6 h, residual mlps 0.9-2.3 h, seq
  members 1.1-3.8 h; all 45 with one seed ~75 h. Compact rosters under evaluation
  (`runs/compact24.sh`): A, 17 members, ~21.6 h (six edge/chord mlps, attn + conv edge, three vessel
  mlps, four cheap old mlps, conv p100, LSTM p100); B = A + res14 edge, 23.8 h (reference only).
  Plan agreed with the user 2026-10-15: a compact candidate to Development first; package prepared
  privately; after 2026-10-19 the private fold predicted, the package pinned and made public, and
  submitted — publication only on the user's explicit yes.
- **compactA built 2026-10-17 07:01** (`runs/submit_compactA_stack`, validated, not published): roster
  A (17 members, ~21.6 GPU-h with one full seed), trees on its 9 non-edge members, predicted from
  seed 0 only; stage + prediction 45 min, peak 71 GB. vs full39e mean |psi| difference 3.5e-4,
  scalars 0.0107 (one seed instead of five shows here). Linear CV 0.995151 (full39e's set 0.995234). Weights are small (3 GB
  full members + 51 MB stage). The private test inputs are not out yet (Final opens 2026-10-19).
- **2026-10-12 16:33: 980602 `submit_full45e_stack` S 0.996016, -0.000009 vs full39e (neutral)**
  (HF d8a439ff; approved by the user). Linear CV said +0.000054. volume +0.00035, R_axis +0.00011,
  D_LCFS 0.008268 -> 0.008209, but **li -0.00071** (0.969226 -> 0.968512), Z_axis / kappa -0.00007;
  0621 0.9885 -> 0.9877. CV had li +0.0006 for the same members: li again moves the other way on the
  public test (as in `li-wide-basis`). The leaderboard shows the latest submission (11th, 0.9960);
  full39e stays the best public candidate.
- **Current best public, 2026-10-12 08:30: 979841 `submit_full39e_stack` S 0.996025** (11th of the
  board; submitted by the agent on the user's approval; HF revision a087b770). full34 + five
  `thomson_edge` members, trees on the first 22 members + raw inputs that now carry the edge columns.
  vs full34 (967520): **+0.000046** (linear CV said +0.000141) — Consistency +0.00015 (kappa +0.00031,
  volume +0.00032, li +0.00019, Z_axis +0.00012, tri_bot +0.00010, R_axis -0.00002), R2_qb +0.00006,
  D_LCFS 0.008327 -> 0.008268, R2_psi +0.000003; worst shot 0621 0.9860 -> 0.9885. The board moved:
  argonclouds 0.9967, heavy4evy / kimsunjoong / yangyiliu 0.9965, five at 0.9964, gwangsan / anulum
  fortis 0.9963; their Consistency 0.9872-0.9881 vs our 0.9861, D_LCFS 0.0068-0.0084 vs 0.0083.
- **Deadlines (Codabench API, 2026-10-08):** Development phase 29580 (public test) ends 2026-10-18,
  5 submissions/day, 100 total; **Final phase 29581 (private test) 2026-10-19..26, 3 submissions per
  person in total.** `submit_full34blk_stack` (full34 + vessel-group GBM block) was submitted by the
  user by hand as 969954 and, the same hour, by the agent's new tool as 969973 (a duplicate of the
  same file; one earlier tool attempt broke off after creating its dataset — whether it also created
  a submission is unverified). **969954: S 0.995987, +0.000008 over full34 (967520)** — R_axis +0.00015,
  Z_axis +0.00012, kappa +0.00009, tri_bot +0.00008, li -0.00030; neutral overall. The vessel-group
  block does not move the GBM stage.
- **Current best public, 2026-10-07 21:17: 967520 `submit_full34b100_stack` S 0.995979** (10th; board
  moved: seantang / kimsunjoong / heavy4evy 0.9964, gwangsan / dmitryzhevn / dgaytanv 0.9963,
  argonclouds / anulum fortis 0.9962, shansiming 0.9960). +0.000203 over full22 (CV +0.000312):
  R2_psi 0.999890, R2_qb 0.997878, D_LCFS 0.008327, Consistency 0.985950 (R_axis 0.983058, Z_axis
  0.992360, kappa 0.978462, tri 0.994566 / 0.993263, volume 0.990913, li 0.969032). li is now at the
  leader's level (967345: 0.969075); the gap moved to Z_axis -0.0025, kappa -0.0023, volume -0.0017,
  tri_bot -0.0014, D_LCFS +0.0008. Worst shots unchanged as CV predicted: 0621 0.9860 (long ramp-down),
  0718 0.9923 (7-frame fragment), 0046 0.9935; the leader has no shot below 0.997.
- **Published to private HF 2026-10-07 (revision 6b2eeae3), pointer zip ready:** `submit_full34b100_stack`
  — full27's 27 + seven vessel-feature members (res6 vessel poloidal / driving, res10 / PCA-70 res6 /
  res6 seed 2 / res 8x768 vessel poloidal, attn t4resln vessel driving); linear stage on 34, CatBoost
  features from the first 22 (as shipped), rows every 3rd. Linear CV 0.995092: +0.000312 over the
  shipped 22's set (q95/betaN +0.00065, kappa +0.0017, li +0.0017, volume +0.0014). Expected public
  0.99600-0.99608.
- **Built, not published (2026-10-06):** `submit_full27b100_stack` — the shipped 22 + five residual+LN
  mlps (res 8x768, res 14x512, PCA-50 res6, res10 seed 2, res6 + dropout 0.05); linear stage on 27,
  CatBoost features from the first 22 (`--gbm-members 22`, the shipped stage's 4900 columns, rows
  every 3rd). Linear CV 0.994834 vs 0.994780 (+0.000054, 4/5). Expected public ~0.99580 (0.99578-0.99583).
- **Current best public, 2026-10-06 13:14: 964529 `submit_full22b100_stack` S 0.995776** (9th):
  R2_psi 0.999872, R2_qb 0.997537, D_LCFS 0.008585, Consistency 0.985369 (li 0.969307, kappa
  0.977586, volume 0.989786, R_axis 0.982369, Z_axis 0.991268, tri 0.994378 / 0.992887). +0.000198
  over full15 (CV said +0.000234). Its linear-only twin 964525 scored 0.995523 (li 0.965703): **the
  GBM stage is worth +0.000253 on the test** — li +0.0036, D_LCFS -0.00026 — while CV on folds 0-2
  credited it +0.000035 and a li loss on fold 0. CV under-reads the stage (it corrects full-refit
  members it never saw); the GBM variant stays the shipped one. Board: seantang 0.9964, heavy4evy
  0.9964, kimsunjoong / dmitryzhevn / dgaytanv 0.9963, anulum fortis 0.9960, argonclouds 0.9959,
  shansiming 0.9958, **pshishkin 0.9958**.
- **Built, not published (2026-10-05):** `submit_full22b100_stack` — strong 15 + six residual+LN
  mlps (PCA-100 res6lnwd / res10ln / res6x1024lnwd / res10lnwd / res6lnwd seed 2, PCA-70 res6lnwd) +
  attn t4x1024 res+LN+wd; stack basis PCA-100, ridgecv_block + CatBoost 1000 iterations on every 3rd
  OOF row (4900 features). Linear CV 0.994780: +0.000234 over full15's set (5/5), li +0.0004.
  Expected public 0.99567-0.99581 (CV-to-public ratio 0.4-1.0; full15 transferred at ~0.4).
- **Public 2026-10-04 12:33: 960052 `submit_full15b100_stack` S 0.995578, 8th** (expected ~0.9958): +0.000084 over full18. R2_psi 0.99987, R2_qb 0.997267, D_LCFS 0.008690, Consistency 0.984644 (kappa 0.977017, tri 0.994264 / 0.992834, R_axis 0.982309, Z_axis 0.99075, volume 0.988789, **li 0.966546**). Against full18 the PCA-100 basis paid as CV said (D_LCFS -0.00047, kappa +0.0013, tri +0.0004) but li fell -0.0022 (CV: -0.0009), about -0.00006 of S. On CV li falls with the basis (24 b70 -> 26 b100 -0.0003) and more with compaction (31 members -0.0005, 15 -0.0009 vs 18 b70): dropped members held li. Board 2026-10-04: kimsunjoong 0.9963, heavy4evy 0.9963, seantang 0.9962, dmitryzhevn 0.9961, dgaytanv 0.9961, anulum fortis 0.9960, shansiming 0.9957, **pshishkin 0.9956**. `submit_full26b100_stack` published (revision c4ceca4b) as the diagnostic: same basis, all old members. Published to private HF (revision 5bdc43ae): — the compact strong 15
  (capacity members of every class at PCA-50/70/100, no old seeds, no old attn; mlp res6ln is the
  irreplaceable one), stack basis PCA-100, ridgecv_block + CatBoost 1000 iterations on every 2nd OOF
  row. Linear CV 0.994546: +0.000063 over full26b100's set, +0.000200 over full18's. Expected
  public ~0.9958 (0.99570-0.99583).
- **Built, not published (2026-10-02):** `submit_full26b100_stack` — the 24 full-data members +
  mlp 3x512 / attn t1024 trained on PCA-100 (five seeds each), stack basis PCA-100, ridgecv_block +
  CatBoost 1000 iterations on every 4th OOF row (wider rows ran out of host RAM, then GPU memory).
  Linear CV 0.994483 vs 0.994346 for full18's member set (+0.000137, 5/5); expected public
  ~0.9957 (0.99563-0.99577 on the full11->full18 CV-to-public ratio). Publication on request.
- **Current best public, 2026-09-28 16:49:** **950254 `submit_full18_stack` S 0.995494** (R2_psi
  0.999875, R2_qb 0.997131, D_LCFS 0.009161, Consistency 0.984547): 18 full-data members (13 +
  conv/attn PCA-70 + conv trunk1024, mlp 1024, mlp 3x512), basis-70 ridge + CatBoost 1000 iterations;
  +0.000203 over full11. Board 2026-09-28 22:00: heavy4evy 0.9963, kimsunjoong 0.9962, anulum
  fortis 0.9960, dmitryzhevn 0.9959, dgaytanv 0.9957, shansiming 0.9957, seantang 0.9956,
  **pshishkin 0.9955 (8th)**.
- **Earlier, 2026-09-27 17:17:** **948230 `submit_full11_stack` S 0.995291**
  (R2_psi 0.999866): eleven members refitted on 100% (five seeds each), basis-70 ridgecv_block +
  CatBoost stage fitted on their 80% OOF (`runs/stackcv_fc11r70`); +0.00010 over five members.
  Board at 17:30: heavy4evy 0.9963, kimsunjoong 0.9961, anulum fortis 0.9960, shansiming 0.9957,
  dmitryzhevn 0.9956, dgaytanv 0.9954, **pshishkin 0.9953 (7th)**.
- **Earlier, 2026-09-27 morning:** **947586 `submit_full5b70_stack` S 0.995191** — mlp,
  seq, conv, lstm, attn each refitted on 100% of the shots (five seeds), basis-70 ridgecv_block
  linear stage + CatBoost (`runs/stackcv_fc5r70`). Same members: basis-50 stack 947607 0.995131
  (basis-70 +0.00006, CV said +0.00005); basis-70 linear only 947579 0.994984 (GBM +0.00021 on
  public vs about +0.0001 on CV). Two full members 946814 0.994951 → five 0.995131 (+0.00018).
  R2_psi 0.999864, D_LCFS 0.009688, Consistency 0.983497.
- **Previous best, 2026-09-26:** **946828 `submit_fc10_stack` S 0.9951** — ten fixcos
  5-fold members (80% fits), per-column linear weights plus a CatBoost residual stage fitted
  cross-fold (`tools/stack_cv.py`, `runs/stackcv_fc10`). **946814 `submit_full_fc2_stack` 0.9950**
  is mlp+seq refitted on 100% of the shots (five seeds) under the fc2 stage, +0.0003 over the
  same members at 80% (946020, 0.994738). Leaderboard top 3 at that time: 0.9959 / 0.9958 /
  0.9957. The protocol, every CV and public number and what was refuted are in the 2026-09-25/26
  entry of `experiments_history.md`. Everything below this item predates the 5-fold protocol.

- **Latest public submission:** Codabench f16 flux-only probe **941187** by `pshishkin` scored
  DIII-D **S 0.993731** on 874 public shots and ranked **11th of 69** in the 2026-09-23
  11:45 UTC [leaderboard response](https://www.codabench.org/api/phases/29580/get_leaderboard/?page=1&page_size=100).
  The previous float32 [submission 941038](https://www.codabench.org/api/submissions/941038/get_detail_result/)
  scored **0.994082**: the [f16 detailed result](https://www.codabench.org/api/submissions/941187/get_detail_result/)
  is **−0.000351** lower, close to the matched local −0.000376. R²ψ and R²qb were unchanged at
  reported precision (0.999817 and 0.995787); D_LCFS rose 0.010898 → 0.011618 and
  Consistency fell 0.979525 → 0.978130. MAST S stayed 0.568092. The current leaderboard shows
  only the latest row; the earlier detail remains accessible. Full official-source hashes and
  terms are in `results/best11_f16_public_20260923.json`.
- **Previous public float32 control:** Submission **941038** ranked **10th of 69** when retrieved
  on 2026-09-23 10:13 UTC. It improved only **+0.000009** over the prior six-salt-labelled
  [submission 915662](https://www.codabench.org/api/submissions/915662/get_detail_result/)
  at S 0.994073: lower R²qb offset better boundary and Consistency. This public comparison is
  separate from the selected local 70-shot score.
- **Historical public reference:** Submission **915662** was created 2026-09-05 and labelled
  **“6 salts”**. Its previous public row and retrieval timestamp are in
  `results/leaderboard_reference_20260923.json`, with its manifest at
  `baselines/diii_d_20260905_leaderboard.json`. The earlier
  `baselines/diii_d_20260904.json` describes a **three-salt script-inferred historical recipe**;
  it does not identify this verified six-salt row. Original fitted artifacts, exact weights and
  submitted code hashes remain unavailable, so no local config is a certified reproduction.
- **Iteration control:** `baselines/iteration.params.yaml`, frozen from the checkout on 2026-09-22.
  Its full enabled ensemble has no fresh score. It retains the accepted configuration including the static auxiliary
  head and `cons_split`. Its sequence learning-rate schedule is `none`. The historical cosine
  result below is a candidate to verify against this control, not silently reapply.
- **Four new split salts measured, no promotion:** The [fixed protocol](experiments/cross-salt-four-review.json)
  excluded the original salt-0 evaluation tail from all eight new member fits and stopping.
  The new eight-member block and its single 50/50 float32 blend were scored on the same 70 shots
  as the existing best local block. The mixed candidate was effectively tied but slightly lower
  than the bit-identical decoded/native control and missed the declared investment gate; no
  public candidate or Hugging Face update followed. Member, block and per-term native scores,
  exact filenames, cache/source hashes and stage times are in
  [results/cross_salt_four_20260924.json](results/cross_salt_four_20260924.json). This new
  collection cannot reproduce the historical six-salt submission.

## Latest recorded evidence

The [source-origin polar representation gate](experiments/polar-source-origin-review.json) is
[measured and closed](results/polar_source_origin_20260924.json). The first OOF TRAIN sample had
no hard source-origin failures or input-only source fallbacks, so the fixed PCA64 basis was
evaluated on the preselected, disjoint third OOF TRAIN sample. Every resulting polygon was
simple and in-grid, but scorer-native point-set Hausdorff missed both predeclared distance limits.
The uncompressed outer-star-envelope construction already carried nearly all that distance;
increasing PCA rank on these measured shots is not the answer. No contour learner, native S,
gate200, original evaluation tail or public artifact was touched.

The one [physical-map residual U-Net pilot](experiments/map-residual-unet-review.json) is now
[measured and closed at fold0](results/map_residual_unet_fold0_20260924.json). Its genuinely OOF
source/true-map fit and input-only li guard passed identity and numerical checks, but the fixed
native gate improved only negligibly against the same sealed 200 shots and missed the declared
investment margin. No fold1 base, two-source mean or older-collection correction was fitted or
scored. The original salt-0 70-shot evaluation tail remained unopened for this pilot, and no
public artifact changed. The separate
[synthetic CUDA/resource check](results/map_residual_unet_preflight_20260924.json) remains
correctness evidence, not a model-quality result.

The fixed full-coordinate input-only contour MLP passed its truth-assisted Stage-0 mechanism
screen, then failed the predeclared Stage-2 native validation score gate. Its sole frozen fit used
the compact train/validation windows only; valid predicted polygons and zero source fallbacks did
not produce a gain against the unchanged compact source. The original 70-shot evaluation tail
was not read for this learner, and no public artifact changed. Full component scores, file/source
hashes, selected-frame counts, wall times and stop decision are in
`results/direct_full_contour_stage2_20260923.json`; the distinct truth-assisted mechanism result
is in `results/direct_full_contour_stage0_20260923.json`. The validation200 window is selection-only
and was seen by both compact-source and contour-model early stopping.
A separate fivefold out-of-fold normal-residual [protocol](experiments/oof-normal-contour-mlp-review.json)
was frozen and stopped after its first fold. K8 captured most normal-displacement energy, but its
truth-assisted full-map [native pilot](results/oof_normal_k8_pilot_20260923.json) missed the
predeclared score margin; the remaining four base fits and correction learner were not run.
The last 200 original validation shots were excluded from the base fit as the internal native
gate, and the original tail70 remained evaluation-only. No public artifact changed.
The [separate full-normal diagnostic](experiments/oof-full-normal-diagnostic.json) is now
[measured](results/oof_full_normal_diagnostic_20260923.json) on the same TRAIN-only pilot. Keeping
all scorer-contour normal components improved over K8 but still missed the prior investment
margin. Neither result promoted the K8 learner; folds 1–4, internal gate200 and original tail70
were not evaluated for this branch.
A [read-only scorer-native li audit](results/oof_li_discrepancy_20260923.json) on those same
saved TRAIN-shot maps found material final-minus-source li discrepancies concentrated in a few
frames. It used no true labels; the separate native true-li R² decline remains a measured score
fact, not a per-frame attribution from this audit. The audit itself performed no optimizer fit
or native rescoring.
A [separate fixed source-fallback li guard](experiments/oof-li-guard-v2.json) was
[measured](results/oof_li_guard_20260923.json) on the same saved TRAIN-shot maps. Its explicit
input-only fallback nearly restored source li and improved over the unguarded truth-assisted
map, but the predeclared native-score investment margin was not reached. This contour-normal
branch is closed without a threshold sweep, additional fit, original-tail score or public action.
A separate [two-rate raw-waveform information gate](experiments/b10-two-rate-information-gate-v4.json)
tested a different permitted input source. Historical raw-signal screens made its prior weak;
the input-only clock/support [preflight](results/b10_raw_support_20260923.json) passed. Its
[first filename-hash preflight](results/b10_prepare_hash_preflight_20260923.json) caught a copied
digest typo before any raw file was read; the v3 amendment preserves that failure and the unchanged
scientific protocol.
The frozen pilot's raw/low-pass sidecar, genuinely OOF target cache, three fixed matched heads
and aggregate/rate-class native scorer path are implemented under `tools/b10_*.py`, with settings
in `baselines/b10-two-rate-info-gate.params.yaml`. The campaign and
[matched native result](results/b10_matched_information_gate_20260923.json) are complete: raw
subframe windows barely changed S relative to the equal-capacity low-pass control, well within
the measured A/A seed difference. The fixed gate failed; no full two-rate GRU or reserved-tail
evaluation followed.
A distinct [nonlinear whole-shot residual gate](experiments/d8-nonlinear-shot-context-v2.json)
used one fixed ExtraTrees model and a deterministic Ridge control on genuinely OOF shots. Its
[held-shot diagnostic](results/d8_nonlinear_shot_context_20260924.json) failed before any
gate200 candidate map or native score was built; the original evaluation tail stayed unopened.
A focused five-step CPU comparison of the ordinary path before and after the explicit-file change
produced byte-identical artifacts and predictions; see
`results/oof_default_path_parity_20260923.json`. This checks compatibility, not the proposed OOF
learner's score.

The user approved a compact production baseline on 2026-09-22: `mlp` and `seq` only, seed 0 each,
salt 0, equal weights, unchanged production shares. Spec: `experiments/compact-production.json`.
Completed campaign: `runs/compact-production-9301af36275620f7`; `run status` verified completion
hashes. On the same 70 held-out shots, MLP scored 0.992149, sequence 0.990514, and their equal-weight
ensemble **0.993385**. Command: `make experiment SPEC=experiments/compact-production.json`.
The cold run took 3511.48 s; MLP fitting took 151.1 s and sequence fitting 817.4 s. Warm full-frame
loading was separately measured at 29.50 s; the estimated repeated iteration is about 20 minutes,
not a second measured complete fit. Native metrics, scored filenames, effective-config hash,
source identity and timing evidence: `results/compact_production_20260922.json`.
This is a one-split working control, not an uncertainty estimate, a measurement of the full
enabled ensemble, or a reproduction of the historical submission.

An older four-MLP-plus-sequence artifact was recovered and rescored locally on these same 70 shots
using its original inference code and the current native scorer: S 0.994478. Fixed-weight
additions of the screened MLP, cosine sequence and two-model refit artifacts led to a best observed
eight-member stack S 0.994872. This is an adaptive diagnostic on reused shots, not an accepted
production replacement or evidence that the recovered August artifact is the September submission.
All fixed grids, member weights and native terms are in the four `results/historical_five_*.json`
records indexed by `experiments_history.md`.
Every fitted member of that five-member collection has also been scored individually with the old
inference code and the pinned current native scorer. The table and exact same-shot provenance are
in `results/historical_five_members_20260922.json`; the concise cross-experiment score and
timing index is `results/overnight_summary_20260923.md`.

An expanded eleven-member fixed blend and a separate scalar-block control were then measured on
the same repeatedly used 70-shot tail. The best observed diagnostic takes the expanded map and
the prior eight-member q95/βN block at S 0.995080, but its selection and evaluation share shots.
That local score is not independent confirmation; the later public submission is recorded above.
Full fixed weights, block identity checks and native terms:
`results/historical_eleven_blocks_20260922.json`.

The fixed eleven-map/eight-scalar recipe has now been assembled locally for all 874 pinned
DIII-D public-test input shots as a float32 NPZ. The existing local validator passed every shot,
and a held-70 dry run of the builder matched the previously scored recipe bit for bit. It was
later uploaded as DIII-D only to the private Hugging Face dataset while the prior MAST file was
preserved, and submission 941038 received the float32 control result above. A separate Challenge1-only
offline root ZIP was also built and checked against the validated NPZ. Output/source identities
are in `results/public_diii_d_candidate_20260923.json`;
ZIP hash, CRC and access audit are in `results/challenge1_offline_zip_20260923.json`.

An explicit float16-flux-only probe kept that recipe and every q95/βN bit fixed. On the same
70-shot tail, the native score fell from S 0.995080 to 0.994704, entirely through LCFS and
Consistency at reported precision; the unchanged scalar R² was 0.997165. Its separate 874-shot
f16 NPZ passed complete structural validation and was uploaded to the existing private HF dataset
at a new pinned revision, preserving MAST and the earlier float32 pointer. The new pointer is
labelled `f16_probe` and was then scored as public submission 941187, with the negative effect
above. The local/public precision penalties differ by only +0.000025 in S.
The initial spec's public source SHA transcription was corrected in an explicit v2 amendment
before public conversion. Exact evidence: `grids/best11-f16-public-probe.json`,
`grids/best11-f16-public-probe-v2.json`, `results/best11_f16_probe_20260923.json` and
`results/private_hf_f16_probe_20260923.json`, plus
`results/best11_f16_public_20260923.json`. The private HF HEAD has since been restored to the
verified float32 file, preserving MAST and the original pinned f32 pointer; the f16 public and
held-70 NPZs and the local f16 pointer copy were deleted at the user's request after their hashes
were retained. See `results/private_hf_f32_restore_20260923.json` and
`results/f16_local_cleanup_20260923.json`. Historical HF revisions and Codabench results remain.

The first-three-shot truth-assisted geometry correctness smoke used 578 fixed, fit-excluded
training-corpus frames. The fixed interior-li remap produced valid maps and improved its derived
term on that small, unusually difficult subset, but two separately frozen full-70 B attempts
stopped on different numerical validity gates; no full-70 B map or score exists. The contour
warp required a boundary taper and a direct damped-Newton inverse to pass all three shots. Its
v6 and safety-only v7 maps and native JSON were bit-identical on that smoke: shape and LCFS
improved but li deteriorated strongly. The sole root-approved v7 full-70 attempt stopped at a
physical Jacobian condition of 11.619 against its frozen limit 10; no full-70 A map or score
exists. This fixed invertible-warp family is closed without another threshold change. No
deployable input-only predictor, public-label use or combined A+B result was produced. See
`results/geometry_map_smoke_20260923.json`, `results/geometry_B_v3_failure_20260923.json`,
`results/geometry_A_v6_smoke_20260923.json`, `results/geometry_A_v7_parity_20260923.json`,
`results/geometry_A_v7_failure_20260923.json` and the frozen specs under
`experiments/geometry-map-*.json`.

A distinct, fixed first-order **additive** ψ correction localized around the predicted LCFS
completed the exact same full 70-shot tail with no native extraction failures and q95/βN bits
unchanged. Native S was **0.995080 → 0.996257** (**+0.001177**): D_LCFS improved
0.009378 → 0.003863 and kappa/volume improved, while derived li R² fell 0.972388 → 0.939465.
Scorer-native contour-to-true distance decreased on 13,776 of 14,830 frames; the full per-shot
physical-distance table and timing are in `results/geometry_additive_full70_20260923.json`.
This uses **true contours** from the labeled training corpus and the same repeatedly used salt-0
tail, so it is a constrained mechanism diagnostic, not a deployable or independently confirmed
gain or a public-score forecast. One borrowed inverse-warp displacement cap stopped the first
full run; a predeclared v3 amendment removed only that irrelevant preflight and reproduced the
three-shot prediction/scorer bits before the successful full run. Evidence:
`results/geometry_additive_smoke_20260923.json`,
`results/geometry_additive_v2_failure_20260923.json`,
`results/geometry_additive_v3_parity_20260923.json` and the frozen specs under
`experiments/geometry-map-additive-*.json`. No model has learned this correction from inputs.

The first source-li compensation family exposed a support bug: all 4,624 nonzero profile
candidates on the first 578 frames altered vacuum grid-edge ψ bits, so only the bit-exact a=0
control remained. The separately frozen spatially restricted variant uses the scorer's polygon
interior and an exact one-cell segment-distance taper. On the same three shots all nine fixed
profile candidates were valid for every frame, and mean absolute error to the ORIGINAL
prediction's own derived li fell 0.0190 → 0.0094; the true-li score and S fell on this small,
atypical subset, as expected because the additive map had already improved them. This does not
measure full-tail or deployable value: the contour stage still uses true contours. Evidence is
`results/geometry_source_li_v1_null_20260923.json`,
`results/geometry_source_li_v2_smoke_20260923.json` and the frozen source-li specs under
`experiments/geometry-additive-source-li-*.json`. The exact fixed full-70 compensation has now
also completed. It selected a nonzero profile parameter on 3,468/14,830 frames, with 50
explicitly rejected nonzero candidates (39 axis and 11 contour validity failures); a=0 remained
available on every frame. The mean absolute error to the **original predicted map's** derived
li fell **0.016031 → 0.007934**. Native S on the same 70 shots was
**0.996257 → 0.997035** (**+0.000778**) against the truth-assisted additive-map control; li R²
rose **0.939465 → 0.967447**, near the original map's 0.972388, while flux R² fell slightly.
R²qb was unchanged and no LCFS/derived extraction failed. This postprocessor's *target* needs
no true labels, but its additive-map input still used C_true, so the combined score is neither
deployable nor independent confirmation. Exact native fields, per-shot target errors and stage
times: `results/geometry_source_li_full70_20260923.json`.
A [bounded input-only contour learnability design](experiments/contour-displacement-learnability-review.json)
has completed exactly once after the user delegated its split choice. On first35/7,267 training
frames, its fixed K8 normal Fourier representation retained **93.915%** of normal displacement
energy and met the one-cell reconstruction gate. One per-shot-weighted Ridge used 216 permitted
features; evaluation last35/7,563 frames were read through target-stripped input columns before
predictions were sealed. Native S on those same 35 was **0.994876** for the original best11 map,
**0.994813** for the predicted contour-local additive map (**−0.000063**), and **0.994841**
after source-derived li preservation (**−0.000035**). The preregistered **+0.001000** screen gate
failed. Raw D_LCFS increased **0.009526 → 0.009702 → 0.009703**; the submitted scalars were
bit-identical and no native extraction failed. This fixed learner did not predict a useful
correction on this reused tail; it does not refute all input-only contour learners, and the local
fold gives no public-gain estimate. Frozen source, exact file hashes, native fields and stage
timings: `results/contour_learnability_screen_20260923.json`. No hyperparameter retry or public
candidate was made.

For **future** contour or base-model fits, the user has reserved all original salt-0 tail 70
shots/14,830 frames as evaluation-only. The completed 35/35 screen above is preserved as a
historical diagnostic, not reused as a fitting protocol. A direct absolute-contour learner from
permitted inputs is the current *unfitted* candidate: its proposed labels come from scorer-native
true LCFS on the 5,633 train and 1,338 validation shots only, with no base-model prediction in
its training target or features. This avoids treating a base model's in-sample residuals as
independent. A [fixed train-only Fourier representation calibration](experiments/direct-contour-representation-prereg.json)
inspected three fixed frames each from the first 100 eligible train shots. Both predeclared
absolute-contour Fourier targets failed their physical Hausdorff gate: K8 median/p95 were
**1.627/1.940 grid cells**, K16 **0.799/1.070 cells**, despite centered-shape L² energy retention
**99.9927%/99.9990%**. All reconstructed polygons stayed simple and inside the grid. This is a
representation failure for these two low-dimensional absolute Fourier choices, not a model-fit
result or a general contour impossibility. No full target cache, architecture, GPU fit, 70-tail
read/score or public prediction followed. The fixed sample, frozen source/scorer identities and
timings are in `results/direct_contour_representation_20260923.json`.
One [physical-contour PCA representation gate](results/direct_contour_pca_20260923.json) then fit
a basis on the same 100 TRAIN shots/300 probes and checked two fixed ranks on the next 100
disjoint TRAIN shots/300 probes. PCA32 missed its Hausdorff limits: median/p95 **0.257/0.570
grid cells**. PCA64 met them at **0.094/0.233 cells**, but one of 300 reconstructed polygons was
non-simple, violating the predeclared zero-topology-failure gate. Both ranks therefore failed;
no extra rank, threshold or repair was selected on those shots. This closes the fixed
low-dimensional absolute-contour branch before any full target cache or learner fit. The scorer
tail 70 and validation 1,338 were not read. The exact disjoint samples, failing frame and timing
are in the result record.
On the one non-simple held TRAIN frame, the unprojected PCA64 contour crossed itself at
**R 1.371973 m, Z −1.060385 m**. The fitted mean contour was simple, but the first fixed
mean-line point that removed the crossing, **t=0.875**, raised Hausdorff error from
**0.203 to 0.556 grid cells**, above the half-cell gate. Later line points were worse.
This [single-frame topology diagnostic](results/direct_contour_pca_topology_20260923.json)
closes that constrained projection without reading a third sample or changing thresholds.

A [read-only official-source external-data audit](results/external_diii_d_data_feasibility_20260923.json)
found that external archives are permitted with disclosure, and EFIT-AI describes potentially
relevant DIII-D equilibria, but did not verify a downloadable corpus with the matching permitted
currents/Thomson/ψ schema, a clear data license and a shot-disjointness proof against the
competition's hidden identifiers. No external corpus was downloaded or fitted; no gain is claimed.
The [one checked-in EFIT-AI HDF5 sample](results/efitai_sample_feasibility_20260923.json) was
subsequently downloaded at its pinned public GitLab commit and SHA. It contains sparse EFIT and
Thomson slices, including q95/βN labels and a finer ψ grid, but no challenge-compatible named
high-rate actuator traces. The project has only that one HDF5 example on its public main branch;
a DVC branch points bulk data at SSH remotes. Its repository BSD license is not an explicit
data-wide reuse grant, and no disjointness proof exists. No bulk access, public/private shot
fingerprinting, model fit or submission was attempted.
The next bounded contour question is a [train-only centroid-ray viability gate](experiments/direct-contour-star-viability-v2.json)
on the already selected first 100 compact TRAIN shots/300 frames. The half-bin angular amendment
avoided deterministic vertex hits, but the [fixed gate](results/direct_contour_star_20260923.json)
failed: all **300** true polygon centroids were strictly inside, yet **4 of 153,600** rays on
four distinct frames each had three physical boundary intersections. The other 153,596 rays had
exactly one; there were no zero or collinear-ambiguous rays. A read-only check found distinct,
interior segment crossings rather than shared-vertex duplicates. The true centroid is a training
target, never an inference feature; this single-chart log-radius construction is now stopped
before PCA, a third sample, learner fit or use of the reserved tail.
A separately [frozen O-point-origin gate](experiments/direct-contour-opoint-star-review.json)
uses the scorer-native magnetic axis of each true TRAIN map as its one alternative polar origin,
with unchanged rays and strict all-or-none threshold. This is a target construction only: a
future direct model would have to predict that axis from permitted inputs, and the existing
best11 predicted axis is excluded from fitting to avoid in-sample leakage. The
[train-only result](results/direct_contour_opoint_20260923.json) found all **300** true O-points
strictly inside but **9 of 153,600** shifted rays on nine frames crossed the LCFS three times;
the other 153,591 had one intersection, with none missing or collinear. Those radii were at least
**174×** farther apart than the fixed vertex-deduplication tolerance. The strict gate failed,
closing this simple one-origin star-shaped family before log-radius PCA, a third TRAIN sample,
model fitting or any use of the reserved tail.
A distinct [full-resolution contour MLP review protocol](experiments/direct-full-contour-mlp-review.json)
is now frozen for an input-only 512×2 physical polygon target, train5,633/validation1,338 and
evaluation-only tail70. It avoids the failed low-dimensional basis and never uses the existing
best11 map, contour or axis as a learner input. Before any new target cache or GPU fit, its fixed
half-strength truth-contour correction must first pass a scorer-native investment gate on the
first 200 compact validation shots using the compact two-model source. That validation source
was early-stopped on the same window, so it is selection-only. The YAML, stage gates, counted
source fallback and final once-only tail protocol are in the review file; no Stage-0 result or
new model fit is yet claimed.
The Stage-0 runner now has separate source-freeze, isolated compact prediction, truth-assisted
half-contour construction and native-score phases. Its code is correctness-checked but no
independent validation score has been read: its [fixed Stage-0 result](results/direct_full_contour_stage0_20260923.json)
on first200 compact validation shots/44,911 frames passed the preregistered investment gate.
Native S was **0.991623** for the compact source, **0.993561** for the truth-assisted half map,
and **0.993861** after input-only source-li preservation (**+0.002238** versus source).
Raw D_LCFS fell **0.012443 → 0.006948** and Consistency rose **0.970709 → 0.979143**;
ψ and q95/βN were essentially unchanged, with scalar bytes exactly identical. There were zero
source fallbacks and native extraction failures; the corrected map's contour was within median
**0.158**, p95 **0.981** physical grid cells of its intended half-blend. Preparation took
**3.35 s**, compact inference **53.69 s**, map plus li construction **1,414.10 s**, and three
native scorer passes **145.12/146.16/145.92 s**. This uses TRUE validation contours and the
compact source had already early-stopped on those shots: it demonstrates constrained local
mechanism headroom, not a learned correction, independent validation or public gain. The next
work is the single preregistered input-only MLP on train5,633/validation1,338; all original
tail70 remains evaluation-only and no GPU fit has yet begun.
Stage-1 code is prepared in `tools/direct_full_contour_cache.py`,
`tools/direct_full_contour_train.py` and `tools/direct_full_contour_predict.py` with the reviewed
YAML unchanged. Its cache stores selected scorer-native contours and permitted inputs only for
the declared train/validation windows, with atomic per-shot identity checks. The one MLP uses
train-only feature/target scaling, shot-balanced batches and the fixed mean-plus-worst-vertex
loss; invalid inferred polygons trigger an explicit counted source-map fallback. Synthetic
input-isolation, cache-format and CPU forward/gradient checks pass. No full target cache, CUDA
fit, learned validation score or reserved-tail inference has yet been measured.

The submission builder's former float16 flux cast has a measured deterministic score cost on
this same 70-shot tail. The local/stream writer and blend/cache path now preserve float32 and
reject stale half-precision caches; an old-control writer round-trip reproduced the original
native scorer JSON exactly. Evidence is in `results/float16_export_audit_20260922.json` and
`results/float32_export_parity_20260922.json`. This is a local export correction, not a claim that
any current artifact reproduces the historical leaderboard submission.

The four 2026-09-22 geometry-loss screens have completed on this same salt-0 tail. The best
matched replacement was `env_weight=1` at ΔS +0.000144 against the decoded two-member control;
`cons_gain=2` was +0.000053, while `cons_gain=4` and `field_weight=0.3` decreased S. This is
single-salt screening with no measured paired variance, so the compact control remains unchanged.
Full native terms and the distinct candidate-alone scores are in
`results/geometry_loss_screen_20260922.json`; interpretation is in `experiments_history.md`.

The separately registered cosine sequence screen also completed on the same 70-shot tail:
replacement ΔS +0.000306 against the matched decoded control. Its candidate-alone score and
per-term result are in `results/sequence_cosine_screen_20260922.json`. This single-salt result
does not replace the working control; the historical eight-pair cosine evidence below remains a
distinct measurement on its own protocol.

The user-approved fixed-step refit on 6971 shots also completed with zero validation shots and
the same 70-shot tail. It scored S 0.992860 alone versus the compact control's 0.993385; an
equal-weight addition of the full refit artifact scored 0.994032. The former is a negative
replacement result, the latter a positive diversity screen, with no new acceptance decision.
Native fields, exact union file check and timings are in `results/refit_production_20260922.json`.
The follow-up cosine sequence fixed-step refit was preregistered separately in
`experiments/seq-cosine-refit.json`, with step and schedule budgets derived from the validated
cosine fit's best step and exact sampler pass ratio. Its individual native score on the same 70
unseen shots is 0.992808; q95/βN worsened while geometry improved relative to the earlier
80%-train cosine sequence. Its fixed 5:1 addition to the recovered old five-member ensemble
scored 0.994824 against 0.994478, a positive reused-tail diagnostic without independent
confirmation. Exact fields and provenance: `results/seq_cosine_refit_20260922.json` and
`results/historical_five_cosine_refit_20260922.json`.

The two-arm fast magnetics production screen has now completed. A (`ECOILA` ±5 ms) lowered the
matched compact replacement by ΔS −0.000448; B (A plus corrected Ip ±20 ms) by −0.000063. A
fresh MLP control reproduced every native metric field of the original compact MLP exactly. The
input-only audit had established feature availability, not predictive value; neither candidate
is accepted. Native terms, exact files, fitted hashes and timing are in
`results/fast_signal_screen_20260922.json`, with cache provenance in
`results/fast_cache_prewarm_20260922.json`.

D8/S6's separate whole-shot linear aggregate screen has now failed its preregistered 25%
explained-variance gate. The all-input diagnostic returned 0.005938 unweighted and −0.022983 when
weighted by each shot's frame count; adding Thomson did not make the residual predictably
recoverable by this Ridge construction. No native score or context branch was run. This does not
settle nonlinear/global context. The original 1,338 validation shots were already seen through
the old ensemble's early stopping, so this is a rule-out screen only. See the dated entry in
`experiments_history.md` and `results/d8_whole_shot_20260922.json`.

B11's Jacobian-direction premise has now passed a controlled denominator check: with one pooled
scalar variance and the compact artifact's Consistency weights across three frame-index groups,
the trace-normalised form dispersion was 0.5394 against the preregistered permutation-null 95th
percentile of 0.2837 and a 0.20 minimum. The old group-specific denominator on identical probes
gave 0.5653. This supports a position-dependent orientation mechanism, not a competition-score
gain; no per-row loss has been implemented or fitted. See `experiments_history.md` and
`results/b11_shared_variance_20260922.json`.

The follow-up B11 MLP training screen has completed. With identical probes/global scaler and one
fixed λ=0.5, the conditioned MLP improved alone by +0.000663 but lost to its matched pooled
control when blended with the unchanged compact sequence (ΔS −0.000290) or added to old-five at
fixed 5:1 (ΔS −0.000063 between additions). The mechanism gate therefore did not yield an
ensemble gain, and no position-conditioned arm is accepted. Native fields and probe identities:
`results/position_metric_screen_20260922.json` and
`results/historical_five_position_metric_20260922.json`.

The single-configuration temporal-convolution sequence member also completed. Its individual S
was 0.990636, but the matched compact MLP-plus-convolution replacement was ΔS −0.000107 against
the decoded MLP-plus-GRU control on the same 70 shots. It is not promoted; the historical
diversity premise and the measured compact replacement are separate evidence. A separate fixed
5:1 addition to the recovered old five was +0.000137, still a reused-tail diagnostic rather than
confirmation. See `results/seq_conv_screen_20260922.json`,
`results/historical_five_conv_20260922.json` and `experiments_history.md`.

Numbers below are paired within salt, recomputed from the named CSVs; they are historical
experiments, not measurements of the newly frozen control. SE is the sample standard deviation
of paired deltas divided by sqrt(n), not an independent-shot confidence guarantee.

| Experiment | Evidence | Pairs | Mean delta S | SE | Positive | Current reading |
|---|---|---:|---:|---:|---:|---|
| Sequence cosine | `results/x_seq_cos_prod.csv`, `c_cos` minus `c_base` | 8 | +0.00010138 | 0.00002000 | 8/8 | Positive historical production result; current config still uses `none` |
| Consistency reweight | `results/x_cons_reweight_prod.csv`, `r_x3` minus `r_base` | 12 | +0.00004075 | 0.00004481 | 6/12 | Accepted by user; gain unresolved, not refuted |
| PCA 70 as replacement | `results/z_npca.csv`, `z_npca_p70` minus `z_npca_base` | 7 | -0.00000571 | 0.00004100 | 4/7 | No established replacement gain; retained as a blend member |
| Static PCA head | `backlog.md`, `1n-canon`, 2026-08-30 | 12 | +0.000065 | See historical CI | — | Accepted with uncertainty and measured +29.6% runtime cost |

The script `python -m my_experiments.run status runs/<campaign>` reports local execution state;
this document does not claim a queue is running. New runs publish scorer results directly in
`metrics.json`; never reconstruct them from progress text. Legacy CSV column `D_LCFS` actually
contains **1-D_LCFS**. New JSON uses the scorer's `dlcfs`, the distance itself (lower is better).

Recompute the production CSV comparisons (these commands display rounded summaries):

```bash
uv run python my_experiments/prodpair.py results/x_seq_cos_prod.csv
uv run python my_experiments/prodpair.py results/x_cons_reweight_prod.csv
uv run python my_experiments/prodpair.py results/z_npca.csv --base npca_base
```

## Next work, not claims of gains

The compact control, geometry loss, cosine sequence, fixed-step refits, fast magnetics,
temporal-convolution and B11 position-conditioned screens above are complete. The recovered
older five-member collection and its fixed additions have been scored on the same 70-shot tail;
all five individual member scores are in `results/historical_five_members_20260922.json`.
The final Ruff/mypy/workflow checks, 39-test frozen CUDA smoke and subsequent 41-test non-GPU gate
are recorded in `results/final_ci_20260923.json`. The complete local dataset remains pinned and
verified. A concise score/timing index is `results/overnight_summary_20260923.md`.

1. The best observed local eleven-map/eight-scalar recipe is assembled as a **DIII-D-only**
   float32 NPZ and a verified root-level Challenge1-only offline ZIP. It passed full local
   structural validation, archive CRC and extracted-entry SHA checks. The private Hugging Face
   dataset already holds MAST predictions, so the preferred delivery is a single-file DIII-D
   upload plus a small pinned pointer ZIP. That single-file upload is now complete; full pinned
   read-token download matched the validated NPZ, and remote MAST and other non-target files are
   unchanged. A subsequent Codabench submission returned the public result above. The publisher is
   `tools/publish_private_candidate.py`, with remote commit and pointer identity in
   `results/private_hf_candidate_20260923.json`, candidate evidence in
   `results/public_diii_d_candidate_20260923.json`, and offline archive evidence in
   `results/challenge1_offline_zip_20260923.json`.
   An input-only comparison of the prior private-repo DIII-D file with this candidate found
   identical 874-shot keys, 193,359 frame counts and shapes. Prior ψ was float16 while the new
   ψ is float32; q95 and βN changed as float32 predictions too. Thus the two files differ beyond
   export precision, but these unlabeled differences do not explain a score or identify the
   exact artifact submitted as 915662. See `results/old_vs_new_private_hf_20260923.json`.
2. A trustworthy generalization check for the adaptively selected old collection needs shots
   held out **before every member fit and selection decision**. The new four-salt fits excluded the
   original tail70, but old collection selection had already reused those same shots, so their
   mixed score is an investment diagnostic rather than independent confirmation. A fresh
   predeclared holdout for all members or an actual public leaderboard evaluation would be needed.
3. If the original September six-salt submission artifacts and weights become available, verify
   their hashes and inference code against submission 915662. The recovered August five-member
   artifact is a useful local control but is not the verified submitted collection.

EMA/Huber, ordinary scalar auxiliary outputs and blind width increases are not reopened merely
because a new assistant is working on the repository. Read their recorded negative results first.
