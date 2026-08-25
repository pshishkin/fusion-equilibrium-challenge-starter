# MAST / Challenge 2 — ideas, ranked

Nothing here is a claim. An idea leaves this file in one of two directions: it gets measured and
moves to [experiments_mast.md](experiments_mast.md) with a verdict, or it gets argued away and
moves there as refuted. The DIII-D half of the fork keeps its own list in [ideas.md](ideas.md).

Ranked best-first by expected ΔS over cost and risk.

## The arithmetic every rank is read against

Board entry 5 stands at **S = 0.56777**. The composite is
`0.55·R²ψ + 0.15·R²qb + 0.10·(1 − D_LCFS) + 0.20·Consistency`, and where it currently pays:

| term | weight | board value | paying | unclaimed |
|---|---|---|---|---|
| R²ψ | 0.55 | 0.8807 | 0.4844 | 0.0656 |
| R²qb | 0.15 | 0.2157 | 0.0324 | 0.1176 |
| 1 − D_LCFS | 0.10 | 0.513 | 0.0513 | 0.0487 |
| **Consistency** | **0.20** | **0.0** | **0.0000** | **0.2000** |

Two things follow, and they set the whole order below.

**Consistency is the largest pool in the challenge and has never paid a thousandth.** All seven
derived scalars are negative, so all seven floor at zero. Anything that lifts even two of them to
+0.1 is worth 0.20 × (0.2/7) = **+0.0057**, four times what the leave-one-out ensemble bought and
twice what the wall bought.

**Unlike Challenge 1, this challenge is not near its ceiling.** Challenge 1 has 0.0056 of S left in
total. Challenge 2 has 0.43 unclaimed and a board that is noiseless, so any real gain is visible.

Read every rank against the standing rule from the board: **scalar calibrations transfer at ~91%;
map tuning fitted on three shots does not.** The ideas that lead are the ones with few parameters
and a physical reading.

---

## M0. Everything below was written before the span fix — re-price it — *2026-08-19*

`span.py` landed after this list was ranked, and it moved the board arithmetic underneath every
entry: locally S 0.720 → **0.780**, Consistency 0.069 → 0.277, R²ψ 0.936 → 0.960, D_LCFS 0.413 →
0.316. Three of the seven derived scalars crossed from negative into positive. The table at the
top of this file is the *old* state; re-read the ranks against
[experiments_mast.md](experiments_mast.md) before spending a run on any of them.

In particular **M1 below is now largely DONE** — the systematic geometric bias it proposed to fit
with two parameters turned out to have a physical cause (a frozen flux span) and a physical fix,
which is why it paid an order of magnitude more than the two-parameter version would have.

## M1. The map's geometry is systematically biased, and Consistency is what that costs

*2026-08-19. The leading idea, and the one the βN result accidentally proved.*

Two measurements of the same thing, which look contradictory until they are put side by side:

- our plasma **volume** scores **R² = −6.85** against the truth's volume (Consistency);
- our plasma volume, through a fitted linear map, predicts the truth's **βN** at **+0.92**.

Both are true because the second only needs our volume to be *monotone* in the truth; the first
needs it to be *equal*. So our geometry carries the information and lands in the wrong place —
which is a **systematic bias, not noise**, and a systematic bias is exactly the low-parameter,
physical, machine-level thing the board has shown does transfer.

Three independent measurements already say the same thing and give the parametrisation:

1. our LCFS comes out **0.929×** the true one — a scale error;
2. our magnetic axis sits **−0.036 ± 0.026 m** in Z against the truth — a bias of over half a
   grid cell, and `diagnose_boundary.py` charges about a third of D_LCFS to position;
3. `edge = 0.03`, which expands the boundary by letting the source run past the last closed
   surface, was worth **+0.0083** — the largest gain after the solve itself, and it moved R²ψ and
   Consistency together.

That third one is the encouraging part: it is the same family of correction, it was fitted on
three shots, and it *did* transfer. The proposal is to go after the remaining two axes
deliberately rather than through `edge` alone.

**Gate.** Two parameters at most (a vertical shift and a boundary scale), fitted by the same
double test that selected the scalar models: leave-one-shot-out **and** coefficient stability
across the three fits. Pre-register that both must hold before it is submitted.

**Watch for:** the terms move against each other. The ensemble bought R²qb 0.183 → 0.197 and paid
D_LCFS 0.472 → 0.487. Score the composite, never one term.

## M2. Sweep `updown` — **DONE 2026-08-19, and the per-shot gate split it**

Swept: pooled optimum at −0.010 worth +0.0044, but only −0.005 is unanimous across the three
demo shots (28348 is *hurt* at −0.010). **Not yet re-swept on the per-frame-profile solve**,
where the vertical field and the profile interact — that is the open part.

### The original entry

*2026-08-19. Cheap, and it is M1's first axis on its own.*

`Calibration.updown` scales every upper coil gain by `(1 + updown)` and every lower one by
`(1 − updown)`. It exists precisely because the gain fit ties each up/down pair to one number — a
turn count belongs to the coil, not to which way its current runs — **and that symmetry is exactly
what cannot represent a vertical field error.** It is set to `0.0` and has never been swept.

The vertical bias it is meant to absorb is measured at −0.036 m, and Z_axis is the worst
Consistency scalar on the board at **−27**.

One parameter, already plumbed, one sweep. This is the cheapest thing on the list and should go
first, if only to price the axis on its own before M1 fits two parameters at once.

## M8. A secant, not a bisection, for the per-frame gamma — *2026-08-19, speed only*

`_solve_frame` costs seven solves a frame, so inference went from 12 minutes to two hours over the
1206-shot split and the three-shot bench from 25 s to 155 s. The span responds to gamma as very
nearly a power law — 0.039 at gamma 0.30 and 0.215 at 3.0, an exponent near 0.74 — so a secant in
log space reaches the same tolerance in about three solves, and warm-starting the bracket from the
previous frame's gamma (frames are 5 ms apart, gamma moves smoothly) would take it to roughly two.
**~3.5x**, for an hour of work.

**Deliberately not done yet.** A secant lands on a slightly different gamma than the bisection
does, which moves every feature the scalar models read and invalidates their fit — the same trap
that made the per-frame profile look like a regression until they were refitted. It buys no score,
only time, and two hours of build is currently tolerable. Do it when the solver is otherwise
settled, and refit the scalars in the same change.

## M3. Take the two Consistency scalars that are nearly in the money

*2026-08-19.*

kappa (−0.52 / −0.66) and li (−0.35 / −0.40) are within one decimal of paying; the other five are
between −2.8 and −27. Because the composite floors each scalar at zero independently, a scalar at
−0.35 and a scalar at −27 are worth the same today — but they are nowhere near equally far away,
and the ranking by "how negative" is the wrong ranking.

kappa is elongation, which is a boundary-shape quantity and therefore downstream of M1. li is the
internal inductance, which is a **profile** quantity — so it is the one Consistency scalar that
M1's geometric correction will *not* fix, and it needs the current profile (`alpha`, `gamma`,
`beta`) rather than the boundary. Worth pricing separately for that reason.

## M4. Re-sweep the profile constants against the corrected composite

*2026-08-19.*

`alpha = gamma = 1`, `beta = 0.3` were re-confirmed in the `edge` sweep, but that sweep was
optimising a composite whose Consistency term was **structurally zero** — every candidate scored
0.0 there, so the sweep was blind to 20% of the metric by construction, not just by the scorer
bug. If M1 or M2 lifts any Consistency scalar off the floor, the profile constants are being
chosen against a different objective and should be re-swept, exactly as `edge`'s optimum moved
once the scorer was fixed.

Cheap, mechanical, and only worth running *after* something makes Consistency non-zero.

## M5. D_LCFS, which pays 0.0513 of a possible 0.10

*2026-08-19.*

`1 − D_LCFS` sits at 0.513. About a third of it is charged to axis position, which M1 and M2
address; the rest is boundary shape. Largely downstream of M1 — listed so it is not double-counted
when M1 is priced.

## M6. Ask the organisers about wider use of the public MAST archive

*2026-08-19. Blocked on an answer, not on work.*

The archive is bit-identical to the withheld truth, so the narrow reading taken so far — machine
geometry only, disclosed in three places — is the safe one and is what should stand until there is
an explicit answer. **Do not widen it unilaterally**: the terms make de-anonymising or memorising
the hidden truth disqualifying, and this is the one source where that has teeth.

Fixed machine properties beyond the wall (coil conductor geometry, in particular for P3 and the
solenoid, which the competition dataset under-describes and which the gains are currently standing
in for at 8.69 and 169.8) would be the natural next request, because they are properties of the
tokamak rather than of any discharge. That is a question for the organisers, not a judgement call.

## M7. More calibration shots — there is no way to get them

*2026-08-19. Recorded so it stops being re-proposed.*

Three demo shots is the entire local truth, and they are consecutive numbers from one campaign.
Everything above is fitted on them. This is the binding constraint on Challenge 2 and there is no
legitimate way around it — which is why the double gate (held-out **and** coefficient stability)
is not optional here, and why a twelve-feature model that cross-validated at +0.972 still had to
be thrown away.

## Closed, and why — so they are not re-proposed

- **DIII-D as a proxy for MAST.** Refuted 2026-08-19, twice over: the vertical pin that makes
  DIII-D solvable has no MAST counterpart, and held-out R²ψ on DIII-D is negative at every
  calibration size up to 24 shots because ECOILA is degenerate with a constant. Task #32 would have
  to close first, and the per-frame flux gauge is what it would have to close.
- **Thomson-driven pressure source.** Refuted on the board at −0.0514. The mechanism stays
  implemented; what failed was fitting one global constant on three NBI-heated discharges.
- **Thomson in the βN model.** Refuted 2026-08-19 on the bench: it makes βN *worse* than dropping
  it (+0.786 with, +0.864 without).
- **Shrinking the scalar affines.** Refuted: the composite floors R²qb *after* averaging q95 and
  βN, so shrinkage pays for insurance the floor already provides.
- **A twelve-feature ridge for q95.** +0.972 held out and thrown away on coefficient instability.
  See experiments_mast.md — the reasoning is the reusable part.
