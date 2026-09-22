# DIII-D: current experiment status

Updated 2026-09-23 (Moscow). This is the current index, not another experiment history.
Historical explanations and negative results remain in `experiments_history.md`, `backlog.md`,
`ideas.md`, and `single_model_results.md`. Their old "running" labels are historical.

## Two distinct baselines

- **Public leaderboard reference:** Codabench submission **915662** by `pshishkin` was created
  2026-09-05, is labelled **“6 salts”**, and scores **S 0.9941** to four decimals. The public
  row and retrieval timestamp are in `results/leaderboard_reference_20260923.json`, with the
  current manifest at `baselines/diii_d_20260905_leaderboard.json`. The earlier
  `baselines/diii_d_20260904.json` describes a **three-salt script-inferred historical recipe**;
  it does not identify this verified six-salt row. Original fitted artifacts, exact weights and
  submitted code hashes remain unavailable, so no local config is a certified reproduction.
- **Iteration control:** `baselines/iteration.params.yaml`, frozen from the checkout on 2026-09-22.
  Its full enabled ensemble has no fresh score. It retains the accepted configuration including the static auxiliary
  head and `cons_split`. Its sequence learning-rate schedule is `none`. The historical cosine
  result below is a candidate to verify against this control, not silently reapply.

## Latest recorded evidence

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
It is neither independent confirmation nor an accepted/submitted collection. Full fixed weights,
block identity checks and native terms: `results/historical_eleven_blocks_20260922.json`.

The fixed eleven-map/eight-scalar recipe has now been assembled locally for all 874 pinned
DIII-D public-test input shots as a float32 NPZ. The existing local validator passed every shot,
and a held-70 dry run of the builder matched the previously scored recipe bit for bit. This is a
reviewable local DIII-D candidate, not a leaderboard score or full two-machine submission; no
MAST build, upload or push occurred. A later Challenge1-only root ZIP was built and checked against
the validated NPZ. Output/source identities are in `results/public_diii_d_candidate_20260923.json`;
ZIP hash, CRC and access audit are in `results/challenge1_offline_zip_20260923.json`.

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
   float32 NPZ and a verified root-level Challenge1-only ZIP. It passed full local structural
   validation, archive CRC and extracted-entry SHA checks. Its ZIP has no MAST component, so it
   would not preserve a Challenge2 score. It has no public leaderboard result or upload: this
   workspace has no accessible authenticated Codabench/HF submission route. Keep the input
   manifest, artifacts and exact weights in `results/public_diii_d_candidate_20260923.json`
   with the archive/access record in `results/challenge1_offline_zip_20260923.json`.
2. A trustworthy generalization check for this adaptively selected collection needs shots held
   out **before every member fit**. The repeatedly scored salt-0 tail and a new ordinary split
   salt do not create an independent dataset for these existing artifacts; confirmation would
   require a predeclared common holdout and new fits, or an actual public leaderboard evaluation.
3. If the original September six-salt submission artifacts and weights become available, verify
   their hashes and inference code against submission 915662. The recovered August five-member
   artifact is a useful local control but is not the verified submitted collection.

EMA/Huber, ordinary scalar auxiliary outputs and blind width increases are not reopened merely
because a new assistant is working on the repository. Read their recorded negative results first.
