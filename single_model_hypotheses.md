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

Moved out of `backlog.md` on 2026-08-30 and deleted there. Each changes ONE model's configuration
and shares nothing with the other members, which is the criterion for living here.

### S1. `seq-ema` + `seq-huber` — model: `seq`
**This is the entry the whole protocol was built for, and it says so in its own last sentence:**
*"`ema` measured ZERO in production for the MLP, so this is a screen-only expectation until
confirmed in production."* Under the old rule that sentence was a dead end — the production number
is the ensemble number, and the ensemble is where a single member's gain disappears. Under this file
it is a screen result awaiting a stack.

`seq` had neither mechanism. Both are plumbed, Huber scaled by 2 so an arm differs from its control
in outlier handling alone rather than also in how often `grad_clip` fires. The EMA is proven wired,
not assumed: at decay 0.9 the averaged weights won 4 evaluations, at 0.999 they won 0 and the score
was byte-identical to base — which is why the model REPORTS its win count, since zero wins and
never-wired-in are otherwise the same number. It won 965-3019 evaluations per run on the MLP, so the
mechanism is not architecture-specific, and `seq` is the only member without it.

**Screen:** two arms (`ema`, `huber`) against `seq` alone, 6 salts, paired within salt. Report
`seq`'s own dS, and the ensemble's beside it.

### S2. `seq-sweep` — model: `seq`
`seq` is a full production member that has never been swept, and it earns the attention: `mlp+seq`
= 0.9943 against 0.9940 for `mlp` plus three extra seeds, so **one extra architecture beats three
extra seeds**. Arms left: `learning_rate` (3e-4 / 3e-3), `seq_hidden` 512, `seq_layers` 2,
`patience_steps`. Dropout is already refuted. ~72 min a run.

**Screen:** one arm per knob, 6 salts each. **This is where the selection arithmetic bites hardest
in this file** — four knobs screened on a model whose paired sd is 0.000831 means the best of them
is inflated by roughly +0.0005 by chance alone at 6 salts. Take the whole set into a stack or take
none; do not take the winner.

### S4. `lstm` — a THIRD architecture, model: a new member beside `seq`
**User's request, 2026-08-30.** `seq` is a bidirectional **GRU** over the shot with a per-frame
encoder and a zero-initialised residual head. An LSTM is the obvious sibling and has never been
tried.

**The case is diversity, not accuracy, and the numbers say so.** `seq` alone LOSES to the MLP —
0.9914 against 0.9931, worse on every term of the composite — and earns its place only in the
average. And the average is where the score actually lives: members sit at 0.9910-0.9912 while the
ensemble sits at 0.9935, so **0.0024 of the score is member disagreement alone**, thirty times the
effects being chased elsewhere. The one measurement that bounds this directly: `mlp+seq` = 0.9943
against 0.9940 for `mlp` plus three extra seeds — **one extra architecture beats three extra
seeds**, which is the strongest evidence in the file for spending on a new member rather than on
more of the same.

**So the reading is not "is the LSTM good", it is "does the LSTM disagree with the GRU".** An LSTM
that scores identically and errs on the same frames adds nothing; one that scores slightly worse but
errs elsewhere may add more than a fourth MLP seed. Report the LSTM's own composite AND its
frame-level correlation with `seq` — seeds correlate 0.56 and feature sets 0.75, so those are the
numbers a new architecture has to beat to be worth its training time.

**Cheap to try, which matters under `0-speed`.** `TorchSeqModel` already packs to true lengths,
handles the residual head and carries `loss`/`ema_decay`/`grad_clip`; `torch.nn.LSTM` has the same
call signature as `torch.nn.GRU`. This is plausibly a `seq_cell: gru|lstm` field and one line at the
construction site, not a new model class.

**Screen:** the LSTM against `seq` on the same salts, 3 salts on rung one of the ladder. If it
survives to a stack it enters as an ADDITIONAL member, not a replacement, since the whole argument
is diversity.

### S5. `mlp-lr-adaptive` — a rate that responds to the curve, model: the MLPs
**User's request, 2026-08-30.** The MLPs run at a CONSTANT rate: `lr_schedule: none` in all four
blocks. Only two schedules exist in the code, `none` and `cosine`, and cosine on the MLP was
refuted — `emacos` measured **-0.00015, 4 of 16 signs** against the accepted `ema`.

**But that refutation does not cover what is being asked, and the difference is the entry.**
`emacos` asked whether a FIXED decay curve helps a run that already averages its weights, and the
answer was no, for a stated reason: both mechanisms shrink the same thing — where the trajectory
ends up — so having one makes the other redundant. An adaptive rule (`ReduceLROnPlateau`: cut the
rate when validation stops improving) is not a fixed curve. It cuts **when the curve says so**,
which is information a cosine schedule does not have and weight averaging does not supply either.

**The honest counter, recorded because today supplied it.** `head-cons-weights` tested exactly this
shape of argument — a feedback rule against a fixed allocation — and the feedback rule did NOT beat
the constant: adaptive +0.000028 against static +0.000055 over 11 salts, indistinguishable. Worse,
the feedback rule converged to a fixed point predictable in advance (correlation 0.97 with a
quantity measured two days earlier), so the loop was **measuring rather than steering**. An adaptive
LR could fail the same way: if it always cuts at roughly the same step, it is a fixed schedule
wearing a feedback costume, and the cheaper test is to read WHERE it cuts and hard-code that.

**So the reading is pre-registered in two parts:**
1. The composite against constant-rate control.
2. **Where and how often it cut**, per fit. If the cut steps cluster tightly across salts and seeds,
   the adaptive machinery is unnecessary regardless of the score — the same verdict the head's
   feedback rule earned, reached the same way.

**Also worth trying in the same family, since the plumbing is shared:** a WARMUP, which neither
schedule has. The MLPs clip 0.2-0.5% of steps with a max gradient norm up to 201 against a threshold
of 1 — the largest excursions are early, and a warmup is the standard answer to exactly that.

**Screen:** 3 salts on rung one, against `lr_schedule: none`. Cheap: one field plus a scheduler
construction beside the existing cosine branch.

### S3. `ema-patience` — model: the MLPs
`ema` reaches its best at 65669 steps against the baseline's 76884 — **15% earlier** — and
`patience_steps` was never revisited afterwards. One line of config.

**Screen:** one arm, 6 salts. Cheap enough that it should go first, and it is the only entry here
whose mechanism is already measured rather than argued: the trajectory demonstrably peaks earlier,
so the only question is whether the stopping rule should follow it.
