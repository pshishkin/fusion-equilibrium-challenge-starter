# Fast magnetics input audit — 2026-09-22

Run `.venv/bin/python my_experiments/diagnose_fast_magnetics.py` to reproduce
[`fast-signal-audit.json`](fast-signal-audit.json). The script reads only five input columns from
the first 100 DIII-D training shots and first 20 public-test shots in salt-0 `shot_key` order. It
verifies the SHA-256 of each selected parquet against the pinned manifest. The JSON records every
filename, verified data hash, source hash, and the full descriptive results. No targets, model fit,
or competition score enter this audit.

The 100 training shots contribute 23,087 EFIT frames; the 20 public-test shots contribute 4,183.
The plasma-current axis needs the established per-shot re-origin in 71/100 training shots and
16/20 public-test shots. Raw `ECOILA` sampling has a median 0.05 ms step in 71/100 training shots
and 16/20 public-test shots, and 0.5 ms in the others. Plasma-current sampling is 0.5 ms in all
120. All proposed centered windows at half-widths 0.2, 5, and 20 ms have **zero frames outside
raw acquisition support** on these shots. There are zero duplicate or decreasing timestamps and
zero nonfinite input samples. These are sample findings, not a whole-corpus guarantee; a producer
must reject unsupported windows with a filename and bounds.

The first aggregate comparison was misleading. Across all frames, the corrected plasma-current
derivative at ±20 ms appeared to retain 55.6% of its variance after linear projection on the
current level, existing raw ±1 ms derivative, and EFIT-grid derivative (74.8% on public test).
That apparent novelty is concentrated in **916/23,087 training frames (3.97%)** and **184/4,183
public-test frames (4.40%)** at shot edges or next to EFIT gaps. On the 22,171 and 3,999 strictly
20-ms-spaced interior frames, the new ±20 ms current derivative equals the existing EFIT-grid
derivative: correlation 0.999999999998 and RMSE below 1e-6 kA/ms in both splits. It is a possible
gap/edge feature, not evidence of generally unused high-frequency current information. The
earlier unstratified interpretation is withdrawn.

`ECOILA` at ±5 ms remains distinct on the regular interior frames: 29.2% of its training variance
and 33.9% of its public-test variance remains after projection on the same three existing
features. Its correlation with the existing EFIT derivative is 0.836 and 0.802 respectively.
The ±0.2 ms derivatives and ±5 ms second differences have near-zero correlation between adjacent
20-ms EFIT frames. That alone cannot show whether their frame-local fluctuations are useful or
noise; this screen does not include them. Plasma current at ±5 ms likewise has weak adjacent-frame
correlation and is left for a separate hypothesis if needed.

The pre-fit screen therefore has two explicit arms against the unchanged control: **A** appends
only the centered `ECOILA` ±5 ms derivative; **B** appends A plus the corrected plasma-current
±20 ms derivative to test gap/edge handling. This audit establishes input availability and feature
relationships only. A scored held-out comparison is needed to establish gain or loss.
