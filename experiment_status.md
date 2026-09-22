# DIII-D: current experiment status

Updated 2026-09-22. This is the current index, not another experiment history.
Historical explanations and negative results remain in `experiments_history.md`, `backlog.md`,
`ideas.md`, and `single_model_results.md`. Their old "running" labels are historical.

## Two distinct baselines

- **Submitted reference:** S = **0.9941**, submitted 2026-09-04, confirmed by the user on
  2026-09-21. `baselines/diii_d_20260904.json` records the provenance and uncertainty. The
  repository's submission script names salts 1/2/3, ten artifacts per salt and 26 model weights
  per salt (78 total). Original artifact hashes and the submitted code revision are unavailable;
  the current configuration is not proof of the submitted recipe.
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

The four 2026-09-22 geometry-loss screens have completed on this same salt-0 tail. The best
matched replacement was `env_weight=1` at ΔS +0.000144 against the decoded two-member control;
`cons_gain=2` was +0.000053, while `cons_gain=4` and `field_weight=0.3` decreased S. This is
single-salt screening with no measured paired variance, so the compact control remains unchanged.
Full native terms and the distinct candidate-alone scores are in
`results/geometry_loss_screen_20260922.json`; interpretation is in `experiments_history.md`.

Numbers below are paired within salt, recomputed from the named CSVs; they are historical
experiments, not measurements of the newly frozen control. SE is the sample standard deviation
of paired deltas divided by sqrt(n), not an independent-shot confidence guarantee.

| Experiment | Evidence | Pairs | Mean delta S | SE | Positive | Current reading |
|---|---|---:|---:|---:|---:|---|
| Sequence cosine | `results/x_seq_cos_prod.csv`, `c_cos` minus `c_base` | 8 | +0.00010138 | 0.00002000 | 8/8 | Positive historical production result; current config still uses `none` |
| Consistency reweight | `results/x_cons_reweight_prod.csv`, `r_x3` minus `r_base` | 12 | +0.00004075 | 0.00004481 | 6/12 | Accepted by user; gain unresolved, not refuted |
| PCA 70 as replacement | `results/z_npca.csv`, `z_npca_p70` minus `z_npca_base` | 7 | -0.00000571 | 0.00004100 | 4/7 | No established replacement gain; retained as a blend member |
| Static PCA head | `backlog.md`, `1n-canon`, 2026-08-30 | 12 | +0.000065 | See historical CI | — | Accepted with uncertainty and measured +29.6% runtime cost |
| Three-salt submission | User's leaderboard report | — | — | — | — | 0.9941 at four decimals; no clean local score for the complete blend |

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

The iteration workflow has been exercised on real DIII-D data; see the 2026-09-22 infrastructure
entry in `experiments_history.md` and `results/workflow_20260922.json`. Its smoke score is not a
fresh production baseline. The complete dataset download is verified against the pinned SHA-256s.

1. Restore the exact submitted artifacts if available; otherwise reproduce the iteration control
   and retain its provenance separately from the historical leaderboard reference.
2. Compare a candidate alone, the fixed baseline blend, and baseline plus candidate using
   `run compare`. Select and confirm on separate predeclared folds. A cross-salt blend needs a
   common holdout excluded before every fit; another salt is not an independent new dataset.
3. Revisit final train-plus-validation refitting with a fixed training duration (ideas A4),
   explicitly agreeing on the changed shares before running it.
4. Screen the implemented geometry loss options (`cons_gain`, `field_weight`, `env_weight`),
   then confirm contribution to the actual ensemble.
5. Gate richer sub-frame signal features cheaply before building a new temporal encoder.

EMA/Huber, ordinary scalar auxiliary outputs and blind width increases are not reopened merely
because a new assistant is working on the repository. Read their recorded negative results first.
