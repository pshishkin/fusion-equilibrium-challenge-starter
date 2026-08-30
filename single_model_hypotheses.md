# Single-model hypotheses — DIII-D / Challenge 1

Changes measured on ONE model rather than on the ensemble. Results go to
[single_model_results.md](single_model_results.md); the ensemble backlog stays in
[backlog.md](backlog.md).

## Why this file exists, and what it changes

Every arm before 2026-08-30 was read through the ensemble, and the ensemble is where small effects
go to die: a change measures zero when its work is already done by another member. The rule is
real — `ema` and `huber` both died that way, while `seq-cosine` transferred because `seq` is a
single member with nothing to dilute it — but it made single-model improvements invisible rather
than absent.

**The protocol, the user's, 2026-08-30.** Screen each change on the single model it touches, keep
everything with a positive point estimate, ACCUMULATE the keepers, and confirm the STACK on the
ensemble later. The screen does not have to prove anything; it filters direction. The proving
happens once, on the stack, where the effects add.

**The acceptance rule for the stack is deliberately left open** — the user's call, to be settled
before the first stack is assembled rather than after it is measured.

## What the measurements already say about this file

Two numbers govern everything here, and both are measured rather than assumed.

**A single model is 2-5x noisier than the ensemble.** Paired sd of dS, over 12 salts of the
`pca-head` arm and 12 of `cons-reweight`:

| model | head arm | cons-reweight arm |
|---|---|---|
| mlp | 0.000507 | 0.000283 |
| mlp1 | 0.000426 | 0.000473 |
| mlp2 | 0.000424 | 0.000375 |
| mlp3 | 0.000711 | 0.000674 |
| seq | — | 0.000831 |
| **ensemble** | **0.000209** | **0.000155** |

So a screen here is a BLUNTER instrument than the ensemble test it replaces, not a sharper one.
That is the price of the protocol, and it is paid knowingly: the point is cheapness of coverage and
additivity, never per-arm sensitivity.

**A single screen barely enriches.** At 6 salts on one model, a change truly worth nothing shows a
positive point estimate **50%** of the time; one truly worth +0.00005 shows it **61%** of the time.
Keep that in view when a list of accepted hypotheses starts to look long: roughly half of anything
worthless will be on it.

**But a stack of three is already visible.** At the ensemble's sd of 0.000155, three changes each
truly worth +0.00005 give t 3.4 over 12 salts, 96% power. Six give t 6.7. Individually unmeasurable,
summed trivial to measure — that is the entire case for this file.

## Conventions

- **One model per hypothesis.** State which, and why the change belongs to that model rather than to
  the ensemble. A change that would help every member alike belongs in `backlog.md`.
- **Screen at 6 salts** unless the entry argues otherwise, paired within salt against the same model
  in the control arm.
- **Record the point estimate, its sd and the salt list**, not a verdict. The stack needs the
  estimates later to check whether the parts add.
- **Report the ensemble number too, even though it decides nothing here.** A screen that helps a
  member and visibly hurts the ensemble is worth catching before it enters a stack.

## Open hypotheses

_None yet. First entries to be drawn from `backlog.md`: those whose mechanism is architecture-local
rather than shared across members._
