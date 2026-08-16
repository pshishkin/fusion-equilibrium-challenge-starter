# The candidate submission — three feature sets, averaged

Not a fit but an ENSEMBLE OF FITS, and the members are three different feature sets rather than
three seeds. Everything here is trained on salt 0's usual split, so the tail 1% is untouched by all
of them and they can be averaged and scored together.

| file | members | features | local S, salt 0 tail |
|---|---|---|---|
| `baseline_prod.joblib` | mlp, mlp1, mlp2, mlp3, seq | the production set | 0.9945 |
| `baseline_vesseld.joblib` | four MLP seeds | + leaky integrals of dI/dt on ECOILA and plasma_current | 0.9944 |
| `baseline_vesselp.joblib` | four MLP seeds | + the same on all twenty poloidal signals | 0.9940 |
| **all three, averaged** | | | **0.9950** |

`baseline_prod.joblib` is byte-identical to `artefacts/20260815-prod-4seeds-seq.joblib`, which is
the fit behind the 0.9932 leaderboard entry. The other two are new, and their own scores are worth
reading before the combination: one is a hair above production and one a hair below, and neither
would be worth submitting alone. Together they are worth +0.0005 on salt 0 and **+0.0009 on salt 3,
which nothing was selected against**.

## Scoring it

```bash
FUSION_ARTIFACT=artefacts/candidate/baseline_prod.joblib \
  uv run python my_experiments/evaluate.py --share 0.01 --jobs 20 \
  --models ensemble "salts:prod+vesseld+vesselp"
```

`salts:a+b` averages the DECODED flux maps of several artifacts rather than their coefficients,
which is what makes this legal at all: each fit has its own PCA basis, target scaler and coil
gains, so the coefficients are not commensurable and only the maps are.

## Submitting it

```bash
FUSION_ARTIFACT=artefacts/candidate/baseline_prod.joblib \
  env -u HF_TOKEN make predict_and_submit_to_hf MODEL="salts:prod+vesseld+vesselp"
```

Three artifacts are loaded and every shot is predicted three times, so the build takes about three
times as long as a single-artifact one — roughly ten minutes for the 874 public-test shots.
