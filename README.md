# Fusion Equilibrium Challenge — pshishkin's fork

Working notes for this fork. The organizers' full guide — physics, data dictionary, scoring
detail, submission rules — is [`README_ORIGINAL.md`](README_ORIGINAL.md), kept unmodified.
Coding conventions, starting with the fail-fast rule, are in [`AGENTS.md`](AGENTS.md) — read that
before adding code under `my_experiments/`.

## What this fork adds

| | |
|---|---|
| `my_experiments/train.py`, `evaluate.py` | The two entry points: shares of the shot list, ordered by a hash of the shot id — training takes the head, the validation window sits right behind it, scoring takes the tail. The pipeline lives in `baseline_model.py`, which **saves** what it trains (`baseline.joblib`) — the starter kit's `experiments.py` never persists a model, so nothing it trains can be scored or submitted. |
| `my_experiments/models.py` + `params.yaml` | The model zoo: ridge, CatBoost, a torch MLP, and a weighted average of the last two. All are fitted on the same features and targets and scored side by side. Every hyper-parameter and the ensemble weights live in `params.yaml`. |
| `my_experiments/eda*.py` | One shot printed transposed with a shape column, for the training split and both public test splits. `eda_coil_field.py` is the odd one out — it plots the flux decomposition below. |
| `my_experiments/coil_field.py` | ψ = ψ_coil + ψ_plasma, with the first term computed from the shipped coil rectangles and currents instead of learned. |
| `my_experiments/target_metric.py` | Which errors of the flux map the loss is allowed to care about, measured from the scored functionals rather than assumed. |
| `mast/` | **Challenge 2.** MAST's calibration and predictor, over the solver below. Its own README, and no import of `my_experiments/` in either direction. |
| `solver/` | The free-boundary Grad-Shafranov solve itself, machine-neutral: `machine.Machine` carries the envelope, the wall radius, the flux sign and the column names. It runs on DIII-D too — not to compete with Challenge 1, but because three MAST demo shots have refuted three decisions taken on them and 7041 labelled shots are where a choice can be tested. |
| `machines.py` | The only file that knows both challenges exist: four dispatch functions on a row's `source`. |
| `toolkit/` | The process pool and the progress bar: what both challenges need and neither owns. It used to live under `my_experiments/`, which made the two top-level files that serve BOTH machines reach into Challenge 1's package for it. |
| `my_experiments/diagnose_budget.py` | What every term of S is worth if it were solved outright — the arithmetic that closed li and every other single-scalar idea. |
| `experiments_history.md`, `ideas.md` | DIII-D: every experiment with its number and verdict; and the ideas nobody has measured yet. |
| `experiments_mast.md`, `ideas_mast.md` | MAST: the same pair, split by machine because results do not cross between them. |
| `mast/span.py` | MAST: the per-frame flux-span target, and the fix that unfroze the profile. |
| `Makefile` | `make ci` = ruff + mypy + the standard metric run. |
| `--source local` | Added to `local_score.py`, `submission_skeleton.py`, `validate_submission.py` — read a downloaded copy of the dataset instead of streaming the Hub. `experiments.py` already had it. |
| `--models` | On `local_score.py` and `evaluate.py`, to score several members of the zoo on one pass over the ground truth. |
| `--jobs` | Everything per-shot on one shared process pool: reading shots (22.6 s to 8.5 s for 352) and the per-shot half of scoring (85.7 s to 28.5 s for 14 x 4 models). Results are bit-identical — see `toolkit/parallel.py`. |
| `--configs` | On `submission_skeleton.py`, to build a DIII-D-only submission without downloading MAST. |

## Where this stands

**Challenge 1 — current public submission:** Codabench submission 941187 by `pshishkin` has been
scored. Its controlled flux-precision comparison with the previous float32 submission 941038
and the earlier six-salt-labelled submission 915662
are in [`experiment_status.md`](experiment_status.md), with links to the official result pages.
The earlier time-stamped public API evidence is
[`results/leaderboard_reference_20260923.json`](results/leaderboard_reference_20260923.json).
The historical six-salt public-reference manifest is
[`baselines/diii_d_20260905_leaderboard.json`](baselines/diii_d_20260905_leaderboard.json).
The earlier [`baselines/diii_d_20260904.json`](baselines/diii_d_20260904.json) is a historical
three-salt script inference, **not** a verified recipe for that submission. Its original fitted
artifacts, weights and code revision are unavailable; today's `params.yaml` does not establish
reproduction of the public row.

**The iteration control is frozen separately** in
[`baselines/iteration.params.yaml`](baselines/iteration.params.yaml). It has no newly measured
score yet for its full ensemble. A compact two-model control has now been measured; see the
2026-09-22 compact production entry in `experiments_history.md`. Current decisions, historical
evidence and unresolved candidates are indexed in
[`experiment_status.md`](experiment_status.md); old "running" labels in the historical notes are
not a live queue. See [the iteration workflow](#reproducible-diii-d-iterations) below.
For the measured same-shot candidate/member scores, observed stage times, export precision and
the locally validated DIII-D public artifact, see the
[`overnight experiment summary`](results/overnight_summary_20260923.md).

The D8/S6 whole-shot linear aggregate diagnostic, including Thomson inputs, failed its
predeclared rule-out gate. It did not produce a new competition score or context model; the
scope and measurement are in [`experiments_history.md`](experiments_history.md).
The B11 Jacobian-direction premise survived a controlled shared-denominator diagnostic; the
later position-conditioned loss improved its MLP alone but reduced both matched ensembles.
The measurements and limits are in the same history.
The one-control/one-candidate training screen was preregistered in
[`experiments/position-metric-screen.json`](experiments/position-metric-screen.json): both arms
share global target coordinates and stopping, with only the candidate's training metric varying
by frame position. Its native same-shot results and separate old-five addition effects are in
[`results/position_metric_screen_20260922.json`](results/position_metric_screen_20260922.json)
and [`results/historical_five_position_metric_20260922.json`](results/historical_five_position_metric_20260922.json).
The conditioned member improves alone but reduces both tested ensembles against its matched
pooled control; it is not promoted.
Two constrained truth-assisted map-space geometry mechanisms were checked on labeled DIII-D
training-corpus shots. The [first-three-shot smoke](results/geometry_map_smoke_20260923.json)
showed useful contour and interior-profile directions, but their separately frozen full-70
versions stopped at numerical validity gates before producing a score. The final contour-warp
[smoke and parity](results/geometry_A_v7_parity_20260923.json) and
[full-70 failure](results/geometry_A_v7_failure_20260923.json) are evidence about this fixed
construction, not a deployable predictor or a generalization result.
A separate [contour-local additive correction](results/geometry_additive_full70_20260923.json)
uses the true contour only as a mechanism probe. It produced finite maps for the full labeled
70-shot tail and a scorer-native geometry change with some li damage; it cannot be run on public
inputs without learning the contour correction from inputs alone. Its fixed formula and one
preflight amendment are in `experiments/geometry-map-additive-*.json`.
A separate [source-li preservation probe](results/geometry_source_li_v2_smoke_20260923.json)
uses the original prediction's own scorer-derived li as its target, never a true-li label.
Its [fixed full-70 diagnostic](results/geometry_source_li_full70_20260923.json) is complete;
the contour-corrected input to this postprocessor remains truth-assisted, so the combined
local score is not a deployable or public result.
A [contour-displacement learnability screen](experiments/contour-displacement-learnability-review.json)
has now [completed](results/contour_learnability_screen_20260923.json). Its fixed input-only
Fourier/Ridge learner passed the shape-representation check but did not improve the native score
on the reused evaluation tail. The frozen runner in `tools/run_contour_learnability.py` separates
source/filename preparation, supervised training, target-stripped predictions and native scoring;
this result neither confirms nor rules out other learned contour corrections.
For future learners, the user has reserved the original salt-0 score tail entirely for evaluation.
The earlier 35/35 screen remains a historical diagnostic. A separately
[predeclared train-only contour representation check](results/direct_contour_representation_20260923.json)
found that the tested absolute Fourier targets preserved nearly all smooth contour variation
but lost boundary detail under the scorer's distance measure. That fixed family stopped before any
learner fit and did not inspect the reserved tail.
A separate [train-only physical-contour PCA gate](experiments/direct-contour-pca-representation-review.json)
has now [completed](results/direct_contour_pca_20260923.json). It recovered the held training
contours more accurately than smooth Fourier, but neither fixed rank met both the boundary and
topology gates. This low-dimensional branch stopped before a correction-learner fit or use of the
reserved evaluation tail.
A [single TRAIN-frame topology diagnosis](results/direct_contour_pca_topology_20260923.json)
found that moving the failed PCA contour toward its valid mean removed the crossing only after
the boundary error exceeded the fixed gate; it did not reopen this branch.
An [official-source external-data review](results/external_diii_d_data_feasibility_20260923.json)
found no verified, downloadable labeled DIII-D corpus with matching inputs and a provable
non-overlap with all competition folds. No outside corpus was downloaded or fitted.
A [single public EFIT-AI HDF5 sample audit](results/efitai_sample_feasibility_20260923.json)
confirms that equilibrium and Thomson fields are present, while the required named high-rate
actuator traces and an openly downloadable bulk corpus remain unverified.
One [train-only centroid-ray viability check](experiments/direct-contour-star-viability-v2.json)
has now [completed](results/direct_contour_star_20260923.json). Almost every fixed ray crossed
the true boundary once, but four train-frame rays crossed it three times; a single positive-radius
chart around the true contour centroid therefore failed its strict gate. No log-radius PCA,
correction model, score-tail access or public prediction followed.
A separate [train-only O-point-origin gate](experiments/direct-contour-opoint-star-review.json)
has now [completed](results/direct_contour_opoint_20260923.json). Some true boundaries also
intersect rays from the magnetic O-point more than once, so this single-origin positive-radius
chart stopped before a log-radius PCA or direct model fit. The O-point came from true ψ only as
a TRAIN-target geometry diagnostic; the reserved tail was untouched.
A [full physical-contour MLP experiment](experiments/direct-full-contour-mlp-review.json) tested
all scorer contour points learned from permitted inputs alone. Its [truth-assisted Stage-0
mechanism gate](results/direct_full_contour_stage0_20260923.json) passed, so one frozen
[input-only target cache](tools/direct_full_contour_cache.py), [fixed MLP](tools/direct_full_contour_train.py)
and [counted-fallback prediction path](tools/direct_full_contour_predict.py) were run. The
[native Stage-2 validation result](results/direct_full_contour_stage2_20260923.json) failed its
predeclared score gate despite valid predictions. The reserved evaluation tail was not read for
this model, and no public prediction was changed. This validation window was used for model
selection, so the result is an investment decision rather than independent confirmation.
A separate [fivefold out-of-fold normal-residual proposal](experiments/oof-normal-contour-mlp-review.json)
has a [frozen-list builder](tools/oof_normal_pilot.py), a guarded
[fixed-step fold selector](my_experiments/oof_split.py) and an
[allowed-shot auxiliary-cache preparer](tools/oof_aux_cache.py). The first
[out-of-fold K8 pilot](results/oof_normal_k8_pilot_20260923.json) failed its fixed
truth-assisted native-score margin despite an accurate compressed normal displacement. The
remaining four base fits and correction learner were not run; the original evaluation tail and
internal gate stayed out of all training and this pilot. The pilot code remains reproducible from
its frozen campaign, but no deployable OOF correction was selected.
A separately [preregistered full-normal diagnostic](experiments/oof-full-normal-diagnostic.json)
retained all scorer-contour normal components on the same TRAIN pilot. Its
[native result](results/oof_full_normal_diagnostic_20260923.json) showed some value in the
discarded detail, but still missed the fixed investment margin. No further folds or contour
learner were fitted, and the reserved evaluation tail stayed untouched.
A [read-only li discrepancy audit](results/oof_li_discrepancy_20260923.json) on the same saved
maps found that the input-only profile step did not reliably preserve its own li target. A small
number of frames dominate the discrepancy. No alternative li optimizer or contour learner was
run as a consequence.
A [single fixed input-only li guard](experiments/oof-li-guard-v2.json) on those saved TRAIN-shot
maps [recovered much of the li loss](results/oof_li_guard_20260923.json), but still failed its
predeclared native-score investment margin. No threshold search, new fit or public action followed.
The [two-rate waveform pilot](experiments/b10-two-rate-information-gate-v4.json) passed its
[input-only raw-support audit](results/b10_raw_support_20260923.json), but the matched
[native gate](results/b10_matched_information_gate_20260923.json) found no material advantage
over a low-pass waveform control. Its [sidecar](tools/b10_waveforms.py),
[OOF target producer](tools/b10_targets.py), [matched heads](tools/b10_head_model.py) and
[fixed YAML](baselines/b10-two-rate-info-gate.params.yaml) remain reproducible; no production
two-rate sequence model was fitted or added to the ensemble.
A [fixed nonlinear whole-shot information gate](experiments/d8-nonlinear-shot-context-v2.json)
reused genuinely out-of-fold base predictions. Its
[held-shot result](results/d8_nonlinear_shot_context_20260924.json) failed the prerequisite
explained-variance gate, so no corrected map or native gate score was built.

**Challenge 2 — MAST scores `S_MAST = 0.5566` on the leaderboard, `G_ratio = 0.560`, from zero.**
Not a transferred model: a free-boundary Grad-Shafranov solve, with the coil field computed exactly
from the shipped conductor rectangles. The best entry on that board is 0.7249 / 0.729, and the gap
is 40% in R²{q95, βN}, 27% in R²ψ, 21% in Consistency and 12% in D_LCFS. Everything about it — the
module, the calibration, what transferred and what did not — is in
[`mast/README.md`](mast/README.md).

**Four probes are built and waiting to be uploaded**, each an exact reading because the solver is
deterministic and the fold is fixed: `β = 0.15` and `β = 0.45` against the shipped 0.30, and two at
`edge = 0.02` (local 0.6105 and 0.6148 against the shipped 0.5941) that separate a new boundary knob
from a gain refit. See `submissions/` — newest four.

**The two challenges share no code.** `my_experiments/` is DIII-D's, `mast/` is MAST's, neither
imports the other, and they meet only in `machines.py` at the repository root — four dispatch
functions that read a row's own `source` column and hand it to the right module. `local_score.py`,
`submission_skeleton.py` and the Makefile go through that file and nowhere else.

## Continuing on another machine

For the state of this checkout and the next agent's starting points, read
[the 22 September 2026 handover](handover-22-sep-2026.md).

Use Python 3.12 and the locked environment. The complete public dataset is 103.86 GB in decimal
units at revision `1e280905b85f2a6fdde7e06fca8cf3a1edf447cb`; DIII-D training contains 7041 files.
The pinned DIII-D filename/size/SHA-256 manifest is `my_experiments/diii_d_dataset.json`.
Data and reproducible caches are not committed.

```bash
uv sync --locked --python 3.12
make download_dataset
make doctor               # dataset completeness and the explicitly configured CPU/CUDA device
make warm-cache           # optional; only after the complete corpus is present
make test                 # frozen smoke campaign, not a model-quality measurement
```

CUDA must be accessible to the process running the checks. An agent sandbox may hide GPU device
nodes even when the host driver works; a failing sandbox check alone does not diagnose a broken
driver. There is no automatic CPU fallback. `experiments/smoke-cpu.json` explicitly selects CPU
when that is wanted; its score must not be compared bit-for-bit to CUDA training.

## The MLP fits on the GPU

`device` in each `torch_mlp` block decides where the fit runs — `cpu` or `cuda`, and there is no
`auto`, on purpose (AGENTS.md, "The core count must not change the score. The device does").

The net is 44 → 512 → 512 → 52 on batches of 512, about 50 MFLOP against a V100's 15.7 TFLOP/s, so
every kernel finishes long before the next is launched and the training loop measures **launch
overhead, not arithmetic**. Two things follow, both measured at production shapes (93736 rows):

| | s/epoch |
|---|---|
| CPU, 4 threads | 0.864 |
| CUDA | 0.237 |
| CUDA, no per-batch sync | 0.233 |
| **CUDA, + fused Adam** | **0.200** |

`float(loss)` inside the batch loop was a full device sync every batch; it is accumulated on the
device and read once per epoch instead. `fused=True` on Adam is one kernel per step rather than a
handful per tensor, measured equivalent to the default implementation to 3.5e-7 over 20 steps.

**The batch size is the larger lever, and it is not free.** Because the loop is launch-bound, epoch
time tracks the NUMBER of batches and barely notices their size — 183 batches cost 0.307 s, 45 cost
0.080, 11 cost 0.020. But raising `batch_size` changes the optimisation rather than the arithmetic:
at 8192 an epoch is 11 Adam steps instead of 183, so an epoch-counted patience means an eighteenth
of the leash it meant before, and the learning rate has to be refitted with it. That first half is
now impossible to get wrong — the leash is counted in steps (see "Steps, not epochs") — and the
second half was measured: batch 4096 with the rate scaled as sqrt is worth 2.3x the speed at
parity.

**A GPU run is not bit-identical to a CPU one and cannot be.** What is checked instead: the score,
paired within a salt and read against the 0.0009 seed sigma, with `ridge` — deterministic, and
untouched by any of this — reproducing to four decimals as the control.

## Data

Downloaded copy lives at `../downloaded_huggingface/hf_dataset`, laid out exactly like the Hub
repo (`data/<config>/*.parquet`) — beside the repo rather than inside it, so a `git clean` cannot
delete 98 GB. That path is the default for every `--local-data-dir` flag, so `--source local`
needs no extra argument.

On a fresh machine:

```bash
make download_dataset                                       # all three configs, 98 GB
make download_dataset DATASET_CONFIGS=diii_d_train          # 88 GB, enough to train
```

No token: the dataset is public. Resumable, since `hf download` skips what is already there, so a
killed run only has to be repeated. `diii_d_train` is 7041 shots and 88 GB, `diii_d_public_test`
6.7 GB, `mast_public_test` 3.1 GB. For a subset of shots, call the tool directly:

```bash
# each parquet is one shot, ~13 MB
uv run hf download Sophelio/fusion-equilibrium-challenge --repo-type dataset \
  --local-dir ../downloaded_huggingface/hf_dataset \
  --include "data/diii_d_train/d3d_shot_01*" --max-workers 16
```

### The decoded-shot cache

Training reads each parquet once and keeps the result in `.shot_cache/` (gitignored): the
features, the flux and the two scalar targets, **every frame**, so one build serves any frame
share. Measured at production, reading 4224 shots went from 80 s to 4.5 s and the peak memory of
that stage from 21 GiB to 8 GiB, with the score bit-identical. It costs ~3.7 MB per shot — about
26 GB for the whole training set.

It invalidates itself: the key is the source parquet's size and modification time together with a
hash of the source text of every function that shapes the arrays, so editing any of them rebuilds
what it has to. `make clean-cache` drops it by hand, which is only needed for a change the hash
cannot see, such as a numpy upgrade that decodes a value differently.

**Prefer `--source local` over streaming.** `experiments.py --source hf` fills a shuffle buffer of
up to 500 rows, and a row costs ~420 MB as a Python dict — measured, not estimated. `--n-shots 50`
therefore asks for ~200 GB and gets OOM-killed before the first shot loads.

## Reproducible DIII-D iterations

The compact iteration baseline uses one seed-0 MLP and one seed-0 GRU sequence model, averaged
with equal weights, at split salt 0 and the unchanged production data shares. Run
`make experiment SPEC=experiments/compact-production.json` to prepare and execute it. This is a
separate baseline from the historical submitted collection; see `experiment_status.md` for results.
The completed first run, native scores, frozen provenance and timing breakdown are recorded in
[`results/compact_production_20260922.json`](results/compact_production_20260922.json).
The first full run may build auxiliary-target and decoded-shot caches; distinguish that cost from
model fitting when estimating subsequent iteration time.

The 2026-09-22 geometry loss screen is frozen in
[`experiments/geometry-loss-screen.json`](experiments/geometry-loss-screen.json). It trains four
single-change MLP candidates at the same production shares and salt, then uses
`python -m my_experiments.run compare-replacement` to score each MLP alone and in place of the
original MLP beside the original sequence model. The comparison command records member-view
provenance and verifies the unchanged native baseline and the matched decoded control on the exact
same 70 shot filenames. Decoded averaging has a measured rounding difference, recorded in
[`results/geometry_identity_20260922.json`](results/geometry_identity_20260922.json); compare
replacements against the decoded control.
The completed four-arm screen and its scorer-native results are in
[`results/geometry_loss_screen_20260922.json`](results/geometry_loss_screen_20260922.json),
with the selection rule and verdict in [`experiments_history.md`](experiments_history.md). The
original compact control remains the working baseline while the small single-salt effects are
assessed.
The separately frozen cosine sequence screen is recorded in
[`results/sequence_cosine_screen_20260922.json`](results/sequence_cosine_screen_20260922.json)
and interpreted in `experiments_history.md`; the working control is unchanged pending a broader
ensemble decision.
An older four-MLP-plus-sequence artifact has also been rescored on the same 70 shots with its
original inference code. Its own score and fixed-weight additions are indexed in
[`experiments_history.md`](experiments_history.md), with native evidence in the
`results/historical_five_*.json` records. This recovered August artifact is a useful local
ensemble control; its identity does not establish what was submitted in September.
The expanded fixed-weight ensemble and its scalar-block diagnostic are indexed in
[`experiments_history.md`](experiments_history.md), with full native evidence in
[`results/historical_eleven_blocks_20260922.json`](results/historical_eleven_blocks_20260922.json).
Their local score reuses the same development tail; the later public result is reported below.
The fixed map/scalar recipe has been assembled as a locally validated DIII-D public-input NPZ;
the artifact, full input identity, float32 format, held-70 recipe parity and build timing are in
[`results/public_diii_d_candidate_20260923.json`](results/public_diii_d_candidate_20260923.json).
A verified Challenge1-only root ZIP is also prepared locally; its archive SHA, entry CRC,
extracted NPZ identity, official format reference and access audit are in
[`results/challenge1_offline_zip_20260923.json`](results/challenge1_offline_zip_20260923.json).
That offline ZIP has no MAST component and was not uploaded.
The DIII-D NPZ has now been uploaded alone to the existing private Hugging Face dataset, which
also contains MAST predictions. The pinned file was downloaded through the read token and matched
the local SHA-256; non-target files were unchanged. The small pointer ZIP and redacted receipt are
recorded in [`results/private_hf_candidate_20260923.json`](results/private_hf_candidate_20260923.json).
For a later revision, use `plan` to check the local candidate and `preflight` after placing
distinct fine-grained `HF_WRITE_TOKEN` and `HF_READ_TOKEN` in the gitignored mode-600 `.env`.
The publisher accepts no token on the command line:

```bash
.venv/bin/python tools/publish_private_candidate.py plan
.venv/bin/python tools/publish_private_candidate.py preflight
.venv/bin/python tools/publish_private_candidate.py publish
```

`publish` uploads only `diii_d_public_test.npz` to the existing private
`pshishkin/fusion-eq-predictions` dataset and checks the pinned LFS identity, unchanged MAST file
and a full read-token download before creating `submissions/submission_pointer_*.zip`.
The ZIP contains a root `manifest.json` with the read credential, so keep it out of Git. The
publisher itself does not submit to Codabench. Submission 941038 was subsequently scored there;
see [`experiment_status.md`](experiment_status.md). An input-only comparison with the DIII-D file
at the private repo's previous commit is in
[`results/old_vs_new_private_hf_20260923.json`](results/old_vs_new_private_hf_20260923.json).

An explicit flux-only float16 probe of the same fixed recipe is now recorded in
[`results/best11_f16_probe_20260923.json`](results/best11_f16_probe_20260923.json). The exact
70-shot native comparison, full 874-shot structural validation and strict scalar-bit checks are
separate from the normal float32 export path. Its DIII-D file was uploaded as a separate pinned HF
revision while the original float32 pointer remained valid; the redacted receipt is
[`results/private_hf_f16_probe_20260923.json`](results/private_hf_f16_probe_20260923.json).
The ignored `submissions/submission_pointer_f16_probe_*.zip` was submitted as Codabench row
941187. Its public precision comparison is in
[`results/best11_f16_public_20260923.json`](results/best11_f16_public_20260923.json).
After that controlled probe, the private HF dataset HEAD was restored to the verified original
float32 DIII-D file without touching MAST or creating a new pointer; see
[`results/private_hf_f32_restore_20260923.json`](results/private_hf_f32_restore_20260923.json).
The user's requested local f16 bulk files and f16 pointer copy were deleted after their hashes
were recorded, freeing space while preserving the small evidence files and remote score history:
[`results/f16_local_cleanup_20260923.json`](results/f16_local_cleanup_20260923.json).
The publisher's optional
`--evidence`, `--receipt`, `--expected-parent` and `--variant float16-probe` arguments make that
single-file revision explicit and prevent overwriting the prior receipt.

The completed fast magnetics screen is documented in
[`experiments/fast-signal-diagnostic.md`](experiments/fast-signal-diagnostic.md) and
[`experiments/fast-signal-screen.json`](experiments/fast-signal-screen.json). Its input-only audit
checked 100 train and 20 public-test shots against their full hashes. The two planned additions
are an `ECOILA` centered ±5 ms derivative, then that feature plus a corrected plasma-current
centered ±20 ms derivative aimed at shot edges and gaps. Their scorer-native production results
are in [`results/fast_signal_screen_20260922.json`](results/fast_signal_screen_20260922.json),
with the explicit new-key cache preparation in
[`results/fast_cache_prewarm_20260922.json`](results/fast_cache_prewarm_20260922.json). Neither
candidate changed the working model.
One separate temporal-convolution sequence candidate was preregistered in
[`experiments/seq-conv-production.json`](experiments/seq-conv-production.json). It uses the
existing convolution cell and changes only that setting from the compact sequence control;
its own score and matched contribution beside the unchanged MLP are in
[`results/seq_conv_screen_20260922.json`](results/seq_conv_screen_20260922.json). It did not
improve the compact ensemble on this split.
Its separate fixed-weight addition to the recovered old five is recorded in
[`results/historical_five_conv_20260922.json`](results/historical_five_conv_20260922.json).
The fast feature bank is an artifact-owned YAML setting backed by separate full-shot sidecars;
the 208-column base shot cache remains reusable. The frozen two-feature CPU smoke and exact
unchanged-model scorer identity are recorded in
[`results/fast_signal_smoke_20260922.json`](results/fast_signal_smoke_20260922.json) and
[`results/fast_signal_identity_20260922.json`](results/fast_signal_identity_20260922.json).
They establish pipeline correctness only.

After `run prepare` prints a fast-screen campaign path, prewarm its new auxiliary producer key and
the two explicit sidecar specifications from that campaign's frozen source:

```bash
repo_dir="$(pwd)"
fast_campaign="$repo_dir/runs/fast-signal-screen-<content-hash>"
cd "$fast_campaign/source"
FUSION_AUX_CACHE="$repo_dir/.aux_targets" MPLCONFIGDIR=/tmp/matplotlib-fast-aux \
  "$repo_dir/.venv/bin/python" -m my_experiments.aux_targets \
  --share 0.80/1.0 --val-share 0.19/1.0 --salt 0 --jobs 24 \
  --local-data-dir "$repo_dir/../downloaded_huggingface/hf_dataset"
for arm in ecoila-5ms-ip-20ms ecoila-5ms; do
  FUSION_SHOT_CACHE="$repo_dir/.shot_cache" MPLCONFIGDIR=/tmp/matplotlib-fast-prewarm \
    "$repo_dir/.venv/bin/python" -m my_experiments.fast_magnetics \
    --params "$fast_campaign/${arm}_s0/params.yaml" \
    --train-share 0.80/1.0 --val-share 0.19/1.0 --jobs 24 \
    --local-data-dir "$repo_dir/../downloaded_huggingface/hf_dataset"
done
```

New sidecars verify each pinned parquet before publication. The auxiliary producer changed with
the scoring context, so this campaign computes its own target caches rather than reusing an older
key. These CPU preparation steps have no fitted model or score.

For a candidate fitted on the same production split, freeze the evaluator to the candidate's
campaign and pass the baseline artifact hash and the measured decoded control:

```bash
.venv/bin/python -m my_experiments.run compare-replacement \
  --baseline runs/<compact>/base_s0/baseline.joblib \
  --baseline-sha256 <compact-artifact-sha256> \
  --candidate runs/<candidate>/<arm>_s0/baseline.joblib \
  --reference-member mlp --companion-member seq --candidate-member mlp \
  --source-campaign runs/<candidate> --decoded-control runs/<identity>/comparison.json \
  --reference-weight 0.5 --companion-weight 0.5 --share 0.01 --jobs 24
```

Run once with the original compact artifact as both baseline and candidate, omitting
`--decoded-control`, to establish the matched control. The command stores scorer-native metrics,
all scored filenames, artifact hashes, the frozen evaluator identity and the comparison controller
hash. It checks both fit and validation filenames against every scored shot.

The canonical data protocols live in [`experiments/protocols.json`](experiments/protocols.json):

| Profile | Train shots/frames | Validation shots/frames | Score shots | Purpose |
|---|---|---|---|---|
| `smoke` | 0.01/0.2 | 0.01/0.2 | 0.002 | Pipeline correctness, one MLP |
| `smoke-refit` | 0.01/1.0 | 0/1.0 | 0.002 | Fixed-step correctness, MLP and sequence on CPU |
| `production` | 0.80/1.0 | 0.19/1.0 | 0.01 | Production; quality uses the same data with ridge + one MLP |
| `refit-production` | 0.99/1.0 | 0/1.0 | 0.01 | Fixed-step refit on train plus old validation shots |
| `paired` | 0.80/1.0 | 0.15/1.0 | 0.05 | Existing multi-salt lane protocol |

These are distinct protocols, not interchangeable score labels. Do not silently change shares.
The `refit-production` shares were approved on 2026-09-22 for this separate experiment; the
precommitted step budgets are explicit in
[`experiments/refit-production.json`](experiments/refit-production.json). They do not replace the
ordinary production protocol or its validation-based stopping.
The measured fixed-step fit and same-tail comparison are in
[`results/refit_production_20260922.json`](results/refit_production_20260922.json). The refit
artifact's own score and its contribution when added to the compact ensemble are distinct results;
the working control remains unchanged.
The follow-up single-sequence cosine refit is predeclared in
[`experiments/seq-cosine-refit.json`](experiments/seq-cosine-refit.json). It retains the same
train/score split and scales the previous cosine sequence's selected steps and schedule horizon
by the exact shot-batches-per-pass ratio. Its individual score and frozen provenance are in
[`results/seq_cosine_refit_20260922.json`](results/seq_cosine_refit_20260922.json). Its separate
fixed-weight addition to the recovered old five-member ensemble is recorded in
[`results/historical_five_cosine_refit_20260922.json`](results/historical_five_cosine_refit_20260922.json).

Before executing the fixed-step refit, the original full-shot train and validation auxiliary
targets can be composed into the 6971-shot cache in their exact order:

```bash
.venv/bin/python -m my_experiments.refit_cache \
  runs/compact-production-9301af36275620f7/base_s0/baseline.joblib \
  runs/refit-production-a5034fff0c4b4f22 \
  --data-dir ../downloaded_huggingface/hf_dataset \
  --record results/refit_cache_composition_20260922.json
```

The helper checks producer code, dependencies, input identities, full-shot lengths, every target
bit and score-tail exclusion before publishing the cache atomically. Its measured composition and
hashes are in
[`results/refit_cache_composition_20260922.json`](results/refit_cache_composition_20260922.json).
An independent verify-only pass is recorded in
[`results/refit_cache_verification_20260922.json`](results/refit_cache_verification_20260922.json).
It reuses computed targets; the refit itself still trains both models from scratch.
Production's enabled ensemble is four MLPs plus a sequence model. Validation is part of fitting;
scoring uses the SHA-1-ordered tail, and every member's train/validation filenames are checked for
overlap. The complete pinned corpus is checked before training so an ongoing download cannot
silently change the split.

```bash
make doctor
make test                 # snapshot + smoke + JSON result; repeats verify and reuse completed work
make quality
make prod
make check-workflow       # inexpensive orchestration and scorer-interface regression tests
make ci                   # lint, types, workflow tests, standard smoke
```

A declarative campaign names its params file, an existing profile, salts, model selection, explicit
`--set` overrides and scored models. `experiments/paired-example.json` is a still-unrun
paired-lane proposal. The separate compact cosine screen above has been measured on the production
salt-0 tail.

```bash
uv run python -m my_experiments.run prepare experiments/paired-example.json
# The command prints runs/<name>-<content hash>. Keep that path; edits do not alter this campaign.
uv run python -m my_experiments.run execute runs/<name>-<content-hash>
uv run python -m my_experiments.run status runs/<name>-<content-hash>
uv run python -m my_experiments.run summary runs/<name>-<content-hash> --control base
# Or prepare and execute in one command:
make experiment SPEC=experiments/paired-example.json
```

`prepare` freezes the source files and effective YAML for ALL arms before any fit. Identity includes
the source hashes, protocol, model configs and installed Python package versions. `execute` refuses
a changed snapshot or environment. Each arm/salt has its own artifact, live log, `metrics.json`,
preflight and completion record. Resume skips only completed runs whose artifact, config and metrics
hashes still match. An interrupted fit is retrained; this is not an optimizer checkpoint facility.
Required head/auxiliary targets are built automatically through the shared shot pool. Their cache
records full-shot lengths, applies the training frame selection separately to each shot, and keys
reuse on producer code, dependency lock and source file identities. Snapshots share the cache safely.
A file lock prevents simultaneous execution of the same campaign; worker process groups are
terminated together on interruption. `tail -F logs/latest.log` follows the active run.
Worker records include the host boot and PID namespace. Inspect or resume a live run in the same
process namespace; a sandbox that hides its workers must not classify them as terminated.

The runner uses the existing `train_eval.py` and `local_score.py` arithmetic. JSON preserves the
scorer's native precision (currently six decimals for S), records the exact scored filenames and
keeps per-functional detail. It does not parse progress text or implement another metric.
`dlcfs` in JSON is the **distance**, unlike the misleading legacy CSV column headed `D_LCFS`,
which contains `1-D_LCFS`. Summaries report paired deltas and SE, not an automatic statistical verdict.
The measured cold/warm cycle, verified resume and scorer self-checks are recorded in the
2026-09-22 workflow entry in [experiments_history.md](experiments_history.md), with machine-readable
evidence in [results/workflow_20260922.json](results/workflow_20260922.json). These are smoke checks,
not estimates of the current production score.
Frozen campaign snapshots include `tools/*.py` as well as their callers; an isolated import test
ensures the submission writer and its format helper resolve inside the snapshot rather than the
live checkout.

### Candidate quality and contribution to the ensemble

```bash
uv run python -m my_experiments.run compare \
  --baseline runs/<campaign>/base_s0/baseline.joblib --weights 5 \
  --candidate runs/<candidate>/base_s0/baseline.joblib --candidate-weight 1 \
  --share 0.05 --jobs 24
```

This example gives a five-member baseline artifact weight 5 and a single-model candidate weight 1;
set the weights to describe the actual blend. The command copies the inputs and source into a new
comparison directory and scores candidate alone, baseline alone and their decoded-map blend on
identical shots through the real scorer. `comparison.json` distinguishes replacement delta from
addition delta. Every artifact participates in the overlap check. Cross-salt members require a
common holdout reserved before fitting; `--on-holdout` selects it. Ordinary salt tails cannot make
an existing cross-salt blend locally scorable. Candidate selection and confirmation must be separate.
The new [four-salt campaign](experiments/cross-salt-four.json) uses an
[exact filename manifest](experiments/cross-salt-four-manifest.json) to retain the original
evaluation tail and full production train/validation counts under each new salt; its
[preregistered blend](experiments/cross-salt-four-review.json) is pending measurement. Prepare
and execute it through `my_experiments.run`; the separate verified auxiliary-cache prewarm must
finish before execution. It is a new float32 collection, not a reconstruction of the historical
six-salt submission.
Prepare the campaign with `python -m my_experiments.run prepare experiments/cross-salt-four.json`.
From its frozen `source/` directory, run `python -m tools.cross_salt_aux_cache` with
`--campaign`, `--data-dir`, `--reference-artifact` and `--jobs 24`, setting `FUSION_AUX_CACHE`
to the workspace cache. Then `python -m my_experiments.run execute <campaign> --data-dir <dataset>`
resumes verified salts in one GPU lane. Cache preparation checks every allowed input hash and
publishes exact per-salt target blocks before execution accepts the campaign.
After all four fits complete, `python -m tools.cross_salt_report` with `--campaign` and
`--data-dir` builds the declared float32 blocks and scores the twelve fixed rows through the
pinned native scorer. Its report checks every member artifact and filename before blending.
Use `--source-campaign runs/<candidate-campaign>` to score through the candidate's verified frozen
evaluator source after the working tree changes. The runner checks the candidate completion source
hash and saves the live comparison controller separately.
For the recovered older five-member artifact, use [`tools/historical_compare.py`](tools/historical_compare.py)
with its pinned original inference source and an explicit `--scorer-root`; the preregistered grids
under `grids/historical-five-*.json` state every fixed member weight and scored-file source.

### Direct commands and legacy recipes

For diagnostics that intentionally use the original entry points, give every fit a unique path:

```bash
FUSION_ARTIFACT=artefacts/lanes/baseline_named.joblib \
  uv run python my_experiments/train_eval.py 0.80/1.0 0.15/1.0 0.05 \
  --jobs 24 --score-models ensemble --metrics-json logs/named-metrics.json
```

The old `tools/*.sh` recipes now locate the repository relative to themselves, but their CSV-based
resume logic is historical and is not the new runner's integrity guarantee. Use frozen campaigns
for new experiments. `make predict_and_submit_to_hf` remains a separately authorized publishing
operation; none of the new runner commands uploads or pushes anything.

## The models

`params.yaml` is the whole configuration: which models are fitted, with which hyper-parameters,
and how the ensemble weights them. Which of its settings have been measured, and what they were
worth, is [experiments_history.md](experiments_history.md). The code carries no defaults to fall back on and rejects an
unknown key rather than ignoring it, so a mistyped hyper-parameter fails the run instead of
quietly not being applied. What is deliberately *not* configurable is how the targets are scaled:
the metric decides that one, so it lives in code.

All of them regress the same vector — `[50 PCA coefficients of ψ, q95, betaN]` — from the same
scaled feature vector, which is what makes them comparable and averageable. Two keys under
`features:` decide what those two things are, and both default to the behaviour that predates them
(see "The flux decomposition"):

- `subtract_coil_field` — fit on ψ − ψ_coil and add the coil field back at inference;
- `inputs` — `currents` (the 21 shipped signals), `coil_pca` (the coil flux in its own principal
  directions, plus the signals that produce no poloidal flux), or `both`.


| | |
|---|---|
| `ridge` | The linear baseline this fork started from. Trained and scored, but **not** an ensemble member — averaging a model that scores 0.12 into two that score ~0.7 only drags them down — 0.5936 with it against 0.7466 without, both measured on the same run. It stays as the figure everything else is read against. |
| `catboost` | Gradient-boosted trees, one `MultiRMSE` model over all 52 outputs. |
| `mlp` | 44 → 512 → 512 → 52 in torch, our own training loop, `device` deciding where the fit runs. The architecture itself is `models.build_mlp` and nothing else restates it, so batch norm, dropout or another layer is an edit in one function. |
| `seq` | The one model whose unit is the **shot**, not the frame: a per-frame trunk, a bidirectional GRU over efit time, and a correction added through a zero-initialised layer, so at step 0 it *is* the MLP. Off by default. Requires `<shots>/1.0` shares and refuses less — see below. |
| `ensemble` | The weighted average of `ensemble.members`. Averaging coefficients and averaging flux maps are the same thing here — the PCA decoder is affine and the weights sum to 1. |

### What a sequence model costs the interface: one argument

The pipeline turned out to be almost ready for a model that reads whole shots, and the two places
one would expect to have to change are exactly the two that needed nothing:

- **`predict` is unchanged.** `predict_row` is called with one shot's frames in efit time order —
  that is the unit the scorer works in, and `local_score` reaches it through the same
  `your_model_predict` a submission does. A sequence model gets its natural input for free. What it
  adds is a length check, because several shots concatenated would return the right SHAPE from a
  sequence that never existed.
- **The ensemble is unchanged.** `_predict_targets` is a weighted sum over the members' predictions
  in the scaled target space, so a member that works differently needs no accommodation: it earns a
  weight in `params.yaml` beside the four MLPs and the two are averaged.
- **`fit` needed one argument.** `_read_shots` concatenates every shot into one block, and the shot
  boundaries are the one thing that destroys. It returns the per-shot frame counts now, and the
  models that score frames independently accept and ignore them.

**Why it needs every frame.** Measured over 202 shots and 44 729 steps of `efit_times`: 97.86% of
steps are 20 ms, and every step is a multiple of 20 — but **77% of shots drop at least one frame**,
the largest gap is 2580 ms, and shots run from 2 to 373 frames. So the clock is one 20 ms grid with
holes in it, not a different rate per shot. Training at `frames: 0.2` would put the model's steps
~100 ms apart while inference feeds it 20 ms, which is learning one dynamics and being asked for
another; `train()` refuses that combination rather than letting it produce a disappointing number.
The holes that remain at 1.0 are what `features.frame_gaps` is for — the gap as an input, in units
of the base step, so a hole reads as "this step was three frames long".

### Target scaling, and why it is not a setting

Some scaling is unavoidable — the leading PCA component carries variance ~1e10 against ~1 for
betaN, so a raw squared-error loss is a fit of that one component and nothing else. But the
*right* scaling is not a matter of taste, so `TargetScaler` in `my_experiments/models.py` fixes it
rather than exposing a knob:

- **One shared divisor for the whole PCA block.** The basis is orthonormal, so by Parseval the
  pixel error of a flux map is the **unweighted** sum of the coefficient errors, ‖Δψ‖² = Σₖ Δcₖ².
  One divisor preserves that geometry, so minimizing the loss is minimizing R²ψ.
- **Each of the two scalars on its own std** — their term of the metric is an R² per scalar,
  against its own variance.
- **The blocks weighted as the composite weights them:** ψ coefficients divided by
  `sqrt(SS_tot_psi per frame / W_PSI)`, each scalar by `std / sqrt(W_QB / 2)`, with the weights
  imported from `fusion_scoring/common.py` rather than copied. The summed squared error then *is*
  the differentiable half of the composite, and the split lands at 71 / 29 — not 79 / 21, because
  `SS_tot_psi` is measured against a single flat mean rather than the mean image.

The alternatives were measured before being discarded, on `train_eval.py 0.01 0.001` with fixed
budgets of 500 iterations / 60 epochs:

| scaling | ensemble | catboost | mlp | ridge |
|---|---|---|---|---|
| as shipped | **0.7466** | **0.7370** | 0.7010 | 0.1157 |
| shared divisor, blocks left at 96 / 4 | 0.7408 | 0.7317 | 0.7023 | 0.1157 |
| every component standardized separately | 0.7155 | 0.7001 | 0.6796 | 0.1157 |

Ridge is identical in all three because it is separable per output and the scale cancels in its
solution — which made it a free check that the scaling step changed nothing it should not have.

### Frames inside one shot are near-duplicates

The equilibrium moves on the current-diffusion timescale, hundreds of milliseconds, while EFIT
frames come far faster — 234 per shot on average. So 16363 training rows are nowhere near 16363
independent observations, and of the psi variance in this data 53% sits BETWEEN shots against 47%
within. At a fixed row budget, shots are what the budget should be spent on.

Both fitting shares therefore take a `shots/frames` form: `0.05/0.2` reads 5% of the shots and
keeps every fifth frame of each. The frames are thinned as each shot is read, before anything is
concatenated, and they are **evenly spaced rather than randomly drawn** — a shot runs through
ramp-up, flat-top and ramp-down, and a uniform draw over 234 frames leaves clumps and gaps across
regimes that are physically different. A stride covers all three and needs no seed to reproduce.

The test share refuses a frame fraction: the metric is defined over every frame of the shots it
scores, and thinning them would produce a number that is not the competition metric.

What it is worth, at equal rows — 70 shots x every frame (16363 rows) against 352 shots x every
fifth (15422 rows), same 14 scored shots:

| | ensemble | catboost | mlp | ridge |
|---|---|---|---|---|
| `0.01 0.01 0.002` | 0.8111 | 0.7753 | 0.7668 | 0.3693 |
| `0.05/0.2 0.05/0.2 0.002` | **0.9319** | **0.9304** | **0.9030** | **0.4423** |

R²ψ goes 0.9240 to 0.9898 and Consistency 0.4472 to 0.7475 for the ensemble. Ridge gains too, and
ridge cannot use extra capacity — the gain is purely in WHICH frames it saw. This is the largest
single improvement measured in this fork, and it cost nothing but reading more files.

Note where CatBoost ended: iteration 2985 of the 3000 ceiling, i.e. it is budget-limited again now
that the data is more varied. Raising `iterations` is the obvious next thing to try.

### Early stopping

CatBoost and the MLP take the validation window as an upper bound on their own budget: CatBoost
with `eval_set` + `early_stopping_rounds` + `use_best_model`, the MLP by evaluating the validation
MSE each epoch and keeping the best epoch's weights rather than the last. `iterations` and `epochs`
in `params.yaml` are therefore ceilings, not targets. Ridge has no iterations, so it accepts the
validation set and ignores it.

`iterations` and `epochs` are ceilings set far above anything the fit should need; both models stop
themselves — CatBoost after ~5061 iterations keeping its best at 4861, the MLP after 686 epochs
keeping epoch 585.

**Does it pay?** Measured on the fixed 7-shot tail, so every row is comparable, with "off" meaning
train the full budget and keep the **last** iteration:

| | catboost | mlp | ensemble |
|---|---|---|---|
| 70-shot validation, patience 200 / 100 | 0.7438 | **0.7273** | — |
| 14-shot validation, patience 200 / 20 | 0.7439 | 0.7130 | 0.7427 |
| 70-shot validation, patience 200 / **20** | 0.7438 | 0.5521 | 0.6585 |
| off, 2000 iters / 300 epochs, keep last | 0.7435 | 0.7070 | 0.7462 |
| off, 500 iters / 60 epochs (the old fixed budget) | 0.7370 | 0.7010 | 0.7466 |

Four things that table says:

**CatBoost does not care.** 0.7435 to 0.7439 across every configuration, while its chosen stopping
point moves from 1942 to 4861 iterations. Three thousand extra trees bought nothing measurable.

**The MLP cares a lot, and about patience, not about early stopping as such.** The same validation
window gives 0.5521 at patience 20 and 0.7273 at patience 100, because its validation curve has a
false floor: patience 20 stopped at epoch 27 keeping epoch 6 — 192 Adam steps for a 75k-parameter
net, undertrained by construction. An epoch here is 32 minibatches, so patience counted in epochs
is a far shorter leash than it sounds.

**A wider validation window is not automatically better.** It moved the MLP from 0.7130 to 0.5521
until the patience was fixed to match. The window and the patience are one decision, not two.

**Validation MSE is not the composite.** The MLP's validation loss keeps improving past the epoch
where its score peaks. The loss sees 52 scaled targets; the score also sees `D_LCFS` and the seven
derived scalars. Do not read a falling validation curve as a rising S.

### Steps, not epochs

Everything above is stated in epochs because that is the unit those measurements were made in. The
MLP no longer has one. `max_steps`, `patience_steps`, `eval_every_steps` and `lr_t_max_steps` are
counted in Adam steps, and the training loop has no epoch boundary in it at all — it draws batches
from reshuffled passes until the budget or the patience ends it.

The reason is that an epoch is `rows / batch_size` steps, so an epoch-counted leash means something
different every time either moves, and this fork moved both: batch 512 → 4096 is 8x, and shares
0.60/0.1 → 0.80/1.0 is another 13x the other way. `params.yaml` used to carry a table of one
patience per share, with a warning that a grid which forgot to rescale would measure the leash and
call it data. The table is gone. One number now: **18400 steps of patience inside a 368000-step
budget**, which is exactly what every result recorded here was calibrated on.

The evaluation cadence became explicit in the same change. It used to be "once per epoch" by
accident of the loop's shape — 61 steps at the production shares, 515 at batch 512 — and it decides
both the resolution at which the best weights are picked and the granularity of the patience. It is
`eval_every_steps: 50` now, and it is a setting rather than a consequence.

### Historical model measurements (not the current baseline)

Two commands, and which one produced a number has to be said every time — see AGENTS.md, "How we
test the metric".

**`make test`** — the smoke run: 70 shots at a fifth of their frames (3272 rows), 14 scored, one
net on the GPU. **1 m 03 s** end to end, which is what makes it usable after every change:

```
             model         S    R2_psi     R2_qb   1-D_LCFS      Cons
               mlp    0.7566    0.8692    0.7718     0.9464    0.3408
          ensemble    0.7566    0.8692    0.7718     0.9464    0.3408
```

`ensemble` repeats `mlp` because `--only mlp` leaves it the sole member. Measured 2026-08-14:

| | |
|---|---|
| reading 140 shots (3272 + 3111 rows kept) | 2.6 s |
| PCA of psi + target scaling | 0.4 s |
| the Jacobian probe, 299 of 300 frames | 23.4 s |
| fitting the MLP, 755 epochs keeping 654 | 11.9 s |
| scoring 14 shots x 2 names, `--jobs 24` | ~24 s |

Note where the time went once the fit stopped dominating: **the Jacobian probe is now the longest
stage of a smoke run**, twice the fit. It is `jacobian_frames: 300` of per-frame scorer calls on
the CPU pool, and nothing about it got faster.

**`train_eval.py 0.80/1.0 0.19/1.0 0.01`** — the production run, what a submission is built from:
5633 shots to fit, 1338 to stop on, 70 scored. The table below was measured at the older
`0.60/0.1 0.15/0.1 0.01`, where the run took about 5 minutes with the shot cache warm; at the
current shares one net is 7.6 minutes of wall clock and 178 s of fit. CatBoost disabled for this one, so the ensemble is the MLP
alone, coil field subtracted, Jacobian loss metric:

```
             model         S    R2_psi     R2_qb   1-D_LCFS      Cons
               mlp    0.9875    0.9994    0.9905     0.9860    0.9530
          ensemble    0.9914    0.9997    0.9937     0.9879    0.9685
             ridge    0.7022    0.8076    0.5461     0.9513    0.4050
```

Salt 0. The ensemble is four MLPs differing only in their seed; it scores 0.9889 on salt 1 and 0.9898 on salt 2 — never compare
absolute scores across salts, only paired differences within one. See the noise tables in
[experiments_history.md](experiments_history.md) before reading any difference from these.

Four changes took this from 0.9809 on 2026-08-13/14: widening the MLP from 256x256 to 512x512
(+0.0022 over three salts), adding the Thomson profiles as features (+0.0039), averaging four
seeds (+0.0042), and 3168 → 4225 training shots (+0.0018). The earlier recipe on 1408 shots scored 0.9769, and 0.9677 before the
coil-field decomposition and the Jacobian loss. Consistency carries essentially all of the movement, while R²ψ barely moves and
R²qb pays for part of it. At this scale shots are still buying score, so a submission is retrained
at 0.45 and never at 0.05.

`loss.jacobian_delta` has to be revisited at this scale, because the probe must sit at the error
the model actually makes: √(1 − R²ψ) is 0.05 at quality scale and **0.033** here. Refitting it on
the production model moved S by +0.0029 — about two sigma against the paired sd of 0.0013 measured
later, so it is a real gain. One refit is enough: the refitted model implies 0.030, which is the
same number again.

**`train_eval.py 0.05/0.2 0.05/0.2 0.01`** — the quality run: 352 shots, 70 scored, roughly a tenth
of the cost. On the earlier 14-shot fold it scored `ensemble 0.9319 / catboost 0.9304 / mlp 0.9030
/ ridge 0.4423`.

**These numbers are not comparable to the ones in the section above.** The scored fold changed with
the shares: ridge went from 0.1157 to 0.3693 without a line of code changing, purely because the
tail of 14 shots is a different, easier set than the tail of 7. Only ever compare rows produced by
the same command — which is why the early-stopping table above was re-measured on one fixed fold
rather than quoted from whatever run produced it.

Only compare figures produced by the same command. Seven held-out shots is a small sample, and a
wider split lands somewhere else entirely — `train_eval.py 0.1 0.02` gave `S = 0.7193`,
`R2_psi = 0.8469` on 141 held-out shots for ridge alone, measured just before the plasma-current
axis fix was finalised. The evaluation sets differ, so the two numbers are not each other's
baseline.

Reference points measured on the same metric, worth keeping in mind: predicting a single flat
constant scores R²ψ = 0 by construction, and predicting the mean ψ *image* already scores 0.36. The
useful range of R²ψ starts around 0.36, not at zero.

### Checks that must keep passing

```bash
make ci                                                     # ruff, mypy, then the metric run
uv run python my_experiments/evaluate.py --share 0.0005 --mode perfect   # S = 1.0000
uv run python my_experiments/evaluate.py --share 0.0005 --mode zeros     # S = 0.0000
```

`make ci` lints and type-checks this fork's own code only — `my_experiments/` plus the root
scripts we edited. The organizers' files and the vendored `fusion_scoring/` stay byte-identical to
upstream and are excluded in `pyproject.toml`.

## The flux decomposition

```bash
uv run python my_experiments/eda_coil_field.py        # -> results/eda_coil_field.png
```

ψ = ψ_coil + ψ_plasma, and the first term is a Green's function times a current, not a modelling
problem. Every ingredient ships on every row of every split — `coil_R/Z/width/height`,
`coil_input_column` to join them to `magnetics_*`, `efit_grid_R/Z` — so `coil_field.py` computes it
for DIII-D and MAST alike and nothing it reads is withheld at test time.

**The calculation checks out, to about 10%.** Fitted outside the plasma boundary with a per-frame
plasma filament and a per-frame constant projected out, the F-coils come out as a group at
**0.87–0.89** depending on the sample, and individually mostly between 0.7 and 1.2 — F6A and F6B,
the cleanest geometry of the set, at 1.01 and 1.02. Independently, the Green's function annihilates
the Grad-Shafranov operator to 5·10⁻³ of itself, which is the finite-difference truncation level.

Two things had to be right to get there, and both were wrong first:

- the flux per radian is `μ0 I sqrt(aR)/(π k) [(1-k²/2)K(k) - E(k)]` — dropping the `1/π` puts every
  gain at 1/π, which is exactly what the first fit reported;
- the stored ψ has the machine's own sign, `AXIS_SIGN` in `fusion_scoring/common.py` (DIII-D −1,
  MAST +1). `coil_field` folds it in, so its output is directly comparable to `efit_psirz`.

**Do not quote a calibration from ten shots.** The first fit here gave 0.9969 for the F-coils and
looked like a confirmation to three decimals. It was not: at 40 shots the same fit gives 0.911, and
giving each frame its own plasma amplitude — a single filament is a crude 1 MA plasma, and pooling
its amplitude pushes the misfit into the coils — brings it to 0.87–0.89 at any sample size tried.
The residual spread across individual coils (F4A at 0.63, F1A at 1.19) is the parallelogram
approximation on F5/F7, the coils sitting on the grid edge, and the vessel eddy currents nobody
ships.

**The plasma filament belongs to the calibration and nowhere else.** It reads `efit_r_axis` /
`efit_z_axis`, which are labels. It is in the design only because without it the coil gains launder
the plasma's field into themselves — a 1 MA plasma against ~140 kA·turn per shaping coil reaches
well past the boundary — and the first fit that omitted it returned gains scattered from −0.15 to
−0.57 and both signs.

**`ECOILA` is not identifiable, so nothing hardcodes it.** It ships in kA with the turn count not
folded in, `ECOILB` is a second co-located group that is not shipped at all, and a solenoid's field
over this grid is nearly degenerate with a constant offset: its gain reads +142, +128, +94 or −10
depending on the sample and on whether a per-frame constant is free. The pipeline therefore fits
its own gains at training time (`coil_field.fit_flux_gains`) and stores them in the artifact. That
is a different question with no true value to recover — *what linear-in-current field leaves the
model the least to learn* — and the split stays exact for any gains at all, since whatever is
subtracted is added back unchanged at inference.

### What it is worth

**Almost nothing on DIII-D, and that is the measured answer.** +0.0088 of S at quality scale over
three seeds, and **+0.0007 at production** — with 3168 shots the model learns the coil field
perfectly well by itself. It stays because MAST has no DIII-D coils to learn from. Numbers and the
full story in [experiments_history.md](experiments_history.md).

Ridge scores the same to four decimals with the subtraction on and off. It should: a linear model
cannot tell the difference between fitting ψ and fitting ψ minus a linear-in-currents field that is
added back afterwards. That row is the check that the decomposition is exact and the add-back is
not quietly lossy.

The PCA basis is fitted on the RESIDUAL, not on the raw maps — the model predicts the residual, so
the basis should span what is predicted, and the coil structure has already been removed exactly.

## The loss the metric actually asks for

The models regress PCA coefficients and the decoder is **affine**, so every quadratic functional of
the map error is a fixed matrix on those coefficients — no autograd, no decoding to pixels, no
change to any model. `my_experiments/target_metric.py` builds that matrix; `TargetScaler` applies
it as one more linear transform of the target block. `loss.metric` picks which one:

| `loss.metric` | what the loss measures |
|---|---|
| `parseval` | the pixel error of the map. On an orthonormal basis that IS R²ψ, and nothing else. Kept as the **control** every number below is read against, and as the fallback where the probe cannot run. |
| **`jacobian`** | how the seven scored functionals actually respond to each coefficient, weighted by the competition's own weights. **The default.** |

The problem `parseval` has is not that it is wrong — it is exactly right for R²ψ — but that R²ψ is
0.55 of the composite and the term the score is losing in is Consistency, which does not read the
map's values at all. It reads its geometry. The magnetic axis is *defined* as where ∇ψ vanishes, so
ψ is flat there and a flux error R²ψ cannot see moves the axis a long way.

`jacobian` closes that by measurement rather than by proxy. Every PCA coefficient is perturbed by
±δ, the perturbed map goes through the scorer's own `derive_frame`, and the resulting Jacobian
assembles

    M = W_PSI/SS_tot_psi · I  +  Σ_j (W_CONS/7) · E[J_jᵀJ_j] / var(f_j)

so the loss becomes `W_PSI·R²ψ + (W_CONS/7)·Σ_j R²_j` — Consistency entering the loss for the first
time, with **no free parameter**: every weight from `fusion_scoring/common.py`, every denominator
from the data.

It is the largest confirmed gain in this fork: **+0.0117 of S at quality scale** over three split
salts and **+0.0097 at production**, essentially all of it Consistency, with R²ψ unmoved and R²qb
paying 0.02 because the ψ block now carries 0.55 + 0.20 of the composite instead of 0.55. See
[experiments_history.md](experiments_history.md) for the tables, for the hand-tuned `field` mode
that preceded it and was removed, and for the boundary term that was tried and refuted.

`loss.jacobian_delta` is the one number here that must be revisited when the scale changes: the
probe has to sit at the error the model actually makes, √(1 − R²ψ) — 0.05 at quality, 0.033 at
production.

D_LCFS is deliberately absent from `M`: it is a distance, zero at the ground truth, so a central
difference sees no derivative at all. It improves anyway, because the shape scalars that ARE in `M`
overlap it.

## `mapnet` — the flux map predicted directly

`my_experiments/mapnet.py` is the one model here with **no PCA basis anywhere**. `torch_mlp` and
`torch_seq` predict a target vector that a fixed basis decodes; mapnet predicts the 65x65 map
itself, through a learned latent and a decoder that is a learned linear map plus a convolutional
refinement, with q95 and betaN read off the same latent.

```
tools/run_mapnet.sh                    # train and score in one command
uv run python my_experiments/mapnet.py --shots 3000 --steps 60000 --latent 50 \
    --mask-weight 16 --sobolev 1.0 --save-artifact artefacts/lanes/baseline_map_s0.joblib
```

**As an ensemble member.** With `--save-artifact` it writes a self-contained artifact — its own
feature scaler and coil plan, weights stored as numpy so it unpickles without torch — and
`predict_row` recognises it by `kind: "mapnet"`. It then joins a blend exactly like any other
member:

```
uv run python local_score.py --models salts:q_base_s0+map_s0 --source local --files ...
```

That works because `salts:` averages **decoded maps in physical units**, never coefficients. The
comment at `baseline_model.py:116` says why coefficients could never be averaged across fits — each
has its own basis — and that same property is what lets a member with no basis at all join the
blend.

**Where it stands, measured on 352 tail shots of salt 0 (list sha1 `0d4a15c2aab058ed`, verified
identical to production's):**

| | S |
|---|---|
| one mapnet | 0.9855-0.9872 over four seeds, mean 0.986501, sd 0.00058 |
| four mapnets averaged | 0.989124 |
| four MLPs | 0.992976 |
| four MLPs + seq | 0.993453 |

**The gap is one term.** Consistency 0.9437 against 0.9740 — 85% of the shortfall, with `R2_psi`
already at 0.9996 against 0.9998. The map is right on average and the geometry is hard to read out
of it. Three levers moved that and are larger than the seed spread: a learned linear branch beside
the convolutions, weighting pixels inside the plasma envelope (57.8% of the grid, so a plain loss
spends 42% of itself where nothing is scored), and matching gradients. Bilinear upsampling,
coordinate channels and a 1/R weight on the gradient term all landed INSIDE the seed spread and are
unmeasured rather than refuted — a distinction worth keeping, since the spread was only measured
after those three had been read as results.

## The collection, and how a submission is assembled

The builder now keeps the flux map and scalar outputs in float32 through local and streaming
prediction, cache reuse, per-salt blending and final NPZ writing. The earlier float16 flux cast
has a measured local LCFS/Consistency cost; see the
[export audit](results/float16_export_audit_20260922.json) and
[writer round-trip](results/float32_export_parity_20260922.json). Cache keys include the output
format version, and an old half-precision cache is rejected. Rebuild it from source predictions;
upcasting its stored values cannot recover lost detail.

The score now comes from blending many models rather than from any one of them, and the pieces below
exist because assembling a blend naively is slow and quietly wrong in three separate ways.

**`tools/registry.py` — what is on disk.** The previous machine held 130 artifacts with 541 fitted
models across 12 splits. The tool lists the artifacts available locally, grouped by SPLIT rather than by name, because that is what decides blendability: `salts:a+b+c`
averages decoded predictions and `evaluate.py` refuses the run unless no member has seen a scored
shot. Members of one salt share the split, so each group is a ready-made blend with a clean 352-shot
tail. Scores are joined from `results/*.csv` on the artifact name, so most rows carry a number
without re-scoring anything.

**`tools/blend_search.py` — which members.** The user's design: rank by solo score, blend the top
25/50/75/100% equally, keep the best, then add one at a time while it improves. It is affordable
because the ground truth's LCFS extraction runs ONCE and each member is predicted ONCE — a
combination then costs an array average plus contour extraction, turning a greedy search from days
into hours. It deduplicates by FITTED WEIGHTS, not names: two lanes' control arms at the same salt
are identical byte for byte, and blending both would give that ensemble double weight and call it
diversity.

**Weights are per MODEL.** `salts:a*5+b` gives an artifact a weight, the same `*` convention
`_predict_targets` uses inside one artifact. Without it `z_npca_base` — itself five models — weighs
the same as one sequence model, and the first confirmation of a searched blend scored a different
thing than was searched for: 0.993115 against 0.993261.

**What the search found**, averaged over salts 1-3: the top 75% beats production by +0.000403, with
25/50/75 within 0.00004 of each other — a plateau, not a peak — while the full collection loses two
thirds of that and goes negative on one salt. The three members it drops everywhere are `parseval`,
`catboost` and `mapnet`, the three argued hardest for on structural-diversity grounds.

### Two caches, and why only one of them is safe

**MAST is cached** (`.mast_cache/`). `machines.predict_row` sends a MAST row to
`mast.predict.predict_row(row)`, which takes the row and NOTHING ELSE — no model, no artifact — so
every DIII-D submission was recomputing 1206 MAST shots that could not have changed, 2.7 of the 5.2
hours a submission took. The key is the shot cache's pattern: the code that produces the output plus
the identity of the inputs.

**Per-salt blends are cached** (`.blend_cache/s<salt>_top<pct>_<hash>/`) so a salt is predicted once
and a submission over N salts costs one salt rather than N. This is exact, not approximate: every
salt carries the same composition at the same per-model weights, so each salt's blend is a mean over
the same total weight, and the mean of equal-sized means is the overall mean — measured at 6.8e-08
relative, float32's own resolution. The directory names what it holds, and the hash is over each
member's FILE CONTENT: artifacts get overwritten here, and a stale blend is valid-looking output for
a model that no longer exists.

**Caching COEFFICIENTS was declined**, and the reason is worth keeping. It is 41x smaller — 39.5 MB
against 1.4 GB per artifact — and the basis is not the obstacle, since it lives in the artifact. The
obstacle is that the cache is then only valid while that FILE is unchanged, and coefficients decoded
through a different basis produce a plausible map rather than an error. A decoded map has no
invariant left to violate; that is the whole difference.

### Assembling one

    SALTS="1 2 3" tools/blend_by_salt.sh                    # fills the per-salt cache
    uv run python tools/blend_salts.py --out submission_3salts \
        --inputs .blend_cache/s1_top75_* .blend_cache/s2_top75_* .blend_cache/s3_top75_*
    FUSION_ARTIFACT=... make predict_and_submit_to_hf SUBMIT_DIR=submission_3salts MODEL=...

`SUBMIT_DIR=` matters: two builds share `submission/` by default and overwrite each other's arrays.
On 2026-09-04 only the accident that their phases did not line up kept one from pushing the other's
half-written file.

## The legacy lane runner

`tools/lane.sh` produced the historical paired tables; new experiments use the frozen runner above: for each salt and each arm it
trains one artifact, scores it, and appends a row to a CSV. It lived in a scratch directory until
2026-08-31 and was therefore invisible to the history — while carrying two bugs in one week, neither
of which left a trace anywhere reviewable. That is why it is here now.

```
JOBS=10 NAME=q OUTFILE=x_something.csv ARMS="base treat" MODELS="ridge mlp mlp1 mlp2 mlp3" \
  ARM_SET_base="" ARM_SET_treat="--set models.mlp.head_weight=0.1" \
  SALTS="0 1 2 3 4 5" tools/lane.sh [wait-pids...]
```

Salts are the OUTER loop and arms the inner, so every arm of a salt sits inside the same split and
the arms are paired with each other rather than merely each against its own control. Trailing
arguments are PIDs to wait for, which is how one lane queues behind another. It resumes from the
CSV, so a killed lane loses only the run in flight.

**`SCORE` names which members to score, default `ensemble` alone.** Scoring is about half a run at
~2.5 min a member, and reading one row while scoring six was pure waste. Pass `SCORE="mlp ensemble"`
when a per-model table is actually wanted.

**Two traps it has already sprung, both worth reading before touching it:**

- **Never edit `lane.sh` while a lane is running.** Bash reads a script as it executes, so an edit
  shifts byte offsets under the interpreter and the loop can misexecute or die mid-run. This
  happened on 2026-08-30; the data survived by luck, not by design. Stop the lane, edit, relaunch —
  it resumes from the CSV.
- **The row extractor reads TWO formats, and that is not redundancy.** `local_score.py` prints its
  comparison table only when more than one model is scored, so `SCORE=ensemble` — the default, for
  speed — produces no table at all. The extractor tries the table row, then falls back to the
  single-model block. Before the fallback existed, a finished 45-minute run was discarded with a
  "NO ensemble row" message; the same shape of failure had already cost three runs in the lambda
  sweep in August. **If a change alters what scoring prints, check this extractor.**

## Things that already cost time

**Changing `FEATURE_WIDTH` while a comparison is in flight costs the comparison.** The cache key
hashes the feature producers, so adding a block invalidates every entry — and the next run rebuilds
7041 shots from parquet instead of reading 27 GB of cache. That path is the memory-hungry one:
measured, the rebuild peaked at 89 GB where the cached read peaks at 54, and the run was killed.
Worse than losing it: the parent died and left **24 orphaned pool workers** holding the pipe open,
so `grep` never saw EOF, never flushed its block buffer, and the log looked like a run that had
produced nothing at all. Rebuild the cache deliberately first — `warm_cache.py --jobs 16`, which is
written to hold one frame per shot and so cannot do this — and only then start the runs that will
be compared.

**Editing an imported module breaks a run whose parent already imported it — through its POOL
WORKERS.** A worker started by `multiprocessing` re-imports the main module from disk, so a run
that is happily fitting will die the moment it reaches a stage that spawns a pool. Measured: a
75-minute sequence fit finished, saved its artifact, and then lost its scoring pass to an import
error introduced while it ran. The same edit killed four queued runs outright. Land module changes
between runs, or accept that anything still to spawn is running against the new code.

**And the change that did it was a cycle, which is worth its own line.** `baseline_model` is the
pipeline; `local_score` and the diagnostic CLIs are built ON it and import from it. So a pipeline
module must never import one of those at module scope — `baseline_model` -> `aux_targets` ->
`local_score` -> `baseline_model` is a cycle that only shows up at import time, and only for
whoever imports first. Both directions are now lazy, inside the functions that need them.

**The thread caps in the Makefile are load-bearing, and a script that skips them dies.** Every
target exports `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4`; a run launched
directly does not. OpenBLAS then reads the HOST's 64 cores and opens that many threads PER worker,
so 20 readers ask for 1280 threads against a container ceiling of 7680 shared with everything else,
and the fits die inside `spawn_main` with `pthread_create failed ... Resource temporarily
unavailable`. Worse, a killed run's workers survive as orphans and keep holding them: measured,
5658 of 7680 threads were held by processes whose parent was gone, which made every LATER run fail
the same way for a reason that had nothing to do with it. Kill orphans by PPID 1, not by name.

**The PCA stage is the memory peak, and `pca_frame_share` is a share.** At `0.98/1.0` it asks for
306k frames where production asks for 250k, and the randomized SVD's centred copy of that array is
the largest thing the run ever holds — measured, the run died at 58 GB there. The setting estimates
50 directions out of 4225 pixels and does not have to grow with the data: scale it DOWN when the
training share goes up, to hold the absolute sample fixed.

**`free` on this box reports the host, not the container.** It says 251 GB where the instance has
128. Anything sized from `free` is sized from a number twice too large.

**Test rows have no `efit_*` targets.** `experiments.load_shot_from_hf_row` reads `efit_psirz`
unconditionally and dies on every test row; `baseline_model.inputs_only_shot` reads only
`efit_times` + `magnetics_*`. Anything on the inference path must go through the latter.

**DIII-D: `magnetics_plasma_current_times` is a shared template, not a per-shot axis.** The same
30719 samples starting at −858.1871 ms are stamped into every shot, while `magnetics_time` really
does vary. For the ~70% of shots recorded at 0.05 ms that template is the wrong axis, so
interpolating Ip onto `efit_times` returns pre-shot noise — 4 kA where the trace sits near 1000 kA.
The correct origin is the shot's own `magnetics_time[0]`, which puts current under 100% of the
frames of every affected shot. `baseline_model.align_ip_times` applies that, and only to shots that
need it; `my_experiments/eda_ip_offset.py` prints the evidence.

**A low-coverage Ip axis is not the same thing as a wrong one.** The check above used to accept
the shipped axis when current flowed under 90% of the EFIT frames and force the correction below
that. On `d3d_shot_21f23b2392` — 1 shot in the 704 scored, 0 in 1104 sampled outside it — that
inverted the truth: it is a 0.5 ms shot whose shipped axis is correct, but its EFIT window is 15
frames over 160–440 ms sitting on the current ramp-up, where |Ip| is genuinely below 5% of its
peak for half of them. Coverage 0.53, "corrected" to 0.00, and the run died at load time.
`align_ip_times` now takes whichever of the two axes covers more frames and raises only if the
better one falls under 50%. Over 1104 shots that reproduces every previous decision exactly — 761
corrected, 342 left alone, none changed — and no shot comes near the floor.
`results/eda_bad_ip_shot.png` is the picture.

**Scoring a large fold used to need 71 GB.** `local_score` holds every scored shot for the whole
run, because the metric pools R² across the fold and nothing can be finalized shot by shot. It
kept the whole parquet row per shot: 101 MB, of which 77 are the raw magnetics traces — 480256
samples per signal at the 0.05 ms acquisition — whose only use is to be interpolated onto
`efit_times`, ~300 points, before anything reads them. `baseline_model.slim_row` now does that
interpolation once at load and keeps the (T, 21) result plus the coil geometry, and `local_score`
hands the pool 64 shots at a time instead of pickling all of them into the queue at once. The
marginal cost per shot went from 101 MB to 11.6 MB, so a 704-shot fold holds about 9 GB. The
metric is untouched — the smoke run scores the same to four decimals.

**The PCA of ψ was not reproducible.** `experiments.TargetPCA` builds `PCA(n_components=50)` and
leaves `random_state` at `None`, and sklearn picks the *randomized* SVD solver for 50 components
out of 4225 pixels. Two runs on the same shots therefore fitted different components — different
targets for every model, and a score that moved on its own: the MLP scored 0.6769 and then 0.6846
on the same 7 held-out shots. `train()` now pins the inner estimator's `random_state` to
`features.pca_seed` from `params.yaml`, rather than editing the organizers' file.

**Roughly 10% of DIII-D shots run reversed plasma current**, and `ridge` scores worse than a
constant on every one of them. The whole input vector flips sign with Ip (the shaping coils mirror
it), so the two polarities are two regimes for one global linear map. Measured on ridge only —
whether the trees and the net, which are free to split on the sign, still suffer from it has not
been re-measured.

**Shot order is load-bearing.** The scorer matches `shot_XXXX_*` keys positionally. Local reads use
`sorted(dir.glob(...))`, which reproduces the Hub's order because `datasets` resolves files through
`fs.glob` and fsspec sorts. Cheap check against the real thing, one shard rather than the whole
fold:

```bash
uv run python validate_submission.py submission/diii_d_public_test.npz \
    --config diii_d_public_test --max-shots 5
```

**One external dataset is used, and it must be disclosed in any methods report.** MAST's first wall
— a closed 37-vertex polygon in `solver/machine.py` — comes from the public MAST archive at STFC
Echo (`https://s3.echo.stfc.ac.uk/mast/level1/shots/{shot}.zarr`, CC BY 4.0). It is machine
geometry, byte-identical across all five campaigns M5–M9, and it replaced an inboard radius this
fork had guessed at 0.1963 against the machine's 0.19524. **Nothing else is taken from that
archive**, because its `efm/` group is the competition's withheld MAST ground truth bit for bit —
`efm/psirz` against `efit_psirz` on a demo shot differs by 0.000e+00 — and the hidden test fold is
drawn from the same campaigns. See `solver/mast_archive.py` and `mast/README.md`.

**MAST's magnetics clock has two defects, and both are per-signal rather than per-shot.** 9 of the
first 120 `mast_public_test` shots repeat exactly five timestamps at the same five indices — all in
the 15482-sample population, all `dt = 0` and never negative, so the samples are written twice
rather than scrambled. And in that same population a signal is padded with `nan` outside the window
it was acquired in: on `mast_shot_018ab61eba` the coil currents are live only over
−150.0…1349.8 ms of a record spanning −2500…5499, while `magnetics_tf_current` is live across the
whole record with nan scattered THROUGH it. `mast/shot.py` drops the repeats, interpolates each
signal over its own live samples, and raises if the EFIT frames fall outside that span — which is
the failure that would otherwise clamp to an endpoint and report a constant.

**0.12% of MAST frames have no closed flux surface**, and they are not a bug: 4 of 3243 over 60
shots, every one at |Ip| of 157–245 kA against a flat-top 600–800, and every one within the last
three frames of its shot. That is the current ramp-down, where the discharge is dying. `mast/`
emits the coil field alone for them and prints the count.

## Pointer zips

Each push writes `submissions/submission_pointer_<UTC timestamp>.zip` rather than one fixed
filename, so rebuilding never overwrites the zip of a submission already on the leaderboard — its
manifest pins a commit SHA, which makes the old zip a record of exactly what was scored. Upload
the newest; keep the rest.

## Not in git

`baseline.joblib` (retrain with the command above), `submission/`, `submissions/`,
`manifest.json` and `.env` — the last three hold the read token in plaintext.

**`results/` is now split rather than excluded wholesale.** The experiment tables — `x_*`, `y_*`,
`z_*` and the lettered grids, 48 files and about 190 KB — are tracked from 2026-08-31. They are
every paired number this project has produced, one row costs 45 minutes of GPU to reproduce, and
the salt lists behind them are recorded nowhere else. The per-frame dumps stay out:
`frame_costs*`, `fc_*` and `frame_sens.csv` are 113 MB of one-row-per-frame diagnostics, rebuilt on
demand from an artifact.

`artefacts/` (1.7 GB of per-salt fits), `logs/` (99 MB) and `.aux_targets/` (454 MB, 13 core-hours
but keyed to an exact file list) remain untracked for size.

`.env` is where the token lives, as `HF_READ_TOKEN=hf_...`; the Makefile includes and exports it,
so `make predict_and_submit_to_hf` needs no argument and the token never reaches the command line.
`.env.example` is the tracked stub. It is deliberately not the Makefile itself — that file IS
tracked, and this fork is public.
