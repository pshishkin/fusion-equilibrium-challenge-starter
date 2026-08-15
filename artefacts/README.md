# Saved artefacts

A fit is `my_experiments/baseline.joblib`, and every run overwrites it. What lands here is a copy
kept on purpose: the one a submission was built from, so the model behind a leaderboard number can
still be loaded after the working file has moved on twenty times.

Load one by pointing the pipeline at it — the path is an environment variable, so nothing else
changes:

```bash
FUSION_ARTIFACT=artefacts/20260815-prod-4seeds-seq.joblib \
  uv run python my_experiments/evaluate.py --share 0.01 --jobs 24
```

## 20260815-prod-4seeds-seq.joblib

The first fit to include a model whose unit is the shot.

| | |
|---|---|
| local S, salt 0, 70 scored shots | **0.9945** |
| members | `mlp`, `mlp1`, `mlp2`, `mlp3` (seeds 0-3) and `seq`, equal weights |
| trained on | 5633 shots (80%), every frame; 1338 to stop on |
| inputs | both/poloidal derivatives, thomson te,ne,p,pos, frame_gaps False |
| targets | 50 PCA components + q95 + betaN, `jacobian` loss metric |
| optimiser | batch 4096, lr 2.8e-3, **grad_clip 1.0** |

What each piece is worth, all confirmed on salts nothing selected against except where noted:

| | |
|---|---|
| every frame instead of every fifth | +0.0018 over eight salts, t = 5.02 |
| gradient clipping | +0.0010 over three salts, t = 3.97 — and it stops 15 fits in 28 from blowing up |
| derivatives of every poloidal signal | +0.0018 over two salts |
| four seeds instead of one | +0.0009 (salt 0) |
| the sequence model on top of four seeds | +0.0005 (salt 0 only; the pair is +0.0007 confirmed) |
