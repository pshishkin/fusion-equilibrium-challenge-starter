# DIII-D / Challenge 1 — what we tested, and what came of it

> Current baseline, protocol and status: [experiment_status.md](experiment_status.md).
> Historical dates, running labels and superseded verdicts below are retained as evidence.

**2026-09-23 UTC — one fixed input-only li guard on the full-normal TRAIN oracle:**
Before looking at guarded score, a strict **|li_final−li_source| > 0.05** rule was frozen: replace
the whole corrected ψ frame with the original source ψ bytes, and leave q95/βN exact. On the
same **100 original-TRAIN OOF shots/21,667 frames**, this reverted **433 frames** across
**49 shots**, fraction **1.99843%**, just within the declared **2%** fallback ceiling. Native S
was **0.991298** source, **0.992090** unguarded truth-assisted full-normal map, and **0.993144**
guarded map. The guard recovered **+0.001054** over unguarded; its **+0.001846** vs source still
missed the predeclared **+0.002600** investment margin. Raw D_LCFS **0.013430 → 0.007076**
(unguarded **0.006741**), Consistency **0.969871 → 0.976140**, li R² **0.938689 → 0.937334**
(unguarded **0.884629**); q95/βN bytes exact and native failures zero. Building the selected
maps took **41.15 s**, one native scorer pass **69.48 s**. Verdict: the input-only mismatch is a
useful validity signal, but this fixed truth-assisted contour family fails its own investment
rule even after the safeguard. No threshold sweep, remaining OOF base fits, correction learner,
evaluation-tail score or public action. [All native terms, fallback identities and hashes](results/oof_li_guard_20260923.json).

**2026-09-23 UTC — did source-li compensation preserve its own input-only target?**
Read-only on the same **100 original-TRAIN OOF shots/21,667 frames**, using only the saved source
and full-normal ψ maps plus input grid geometry, the byte-pinned scorer extracted each map's own
LCFS and li. The absolute final-minus-source li difference had median **0.007745**, p95
**0.019743**, p99 **0.147994**, maximum **1.783671** and RMS **0.042665**; only **0.688%** of
frames agreed within **1e−4**. The largest **1%** of frames supplied **89.44%** of squared li
discrepancy, with a few whole shots carrying much of it. The previously measured native true-li
R² was **0.938689 → 0.884629**. Verdict: the nine-candidate spatial source-li step did not
actually hold its own input-only li target, especially in rare high-impact frames. Squared
source-to-final disagreement is not itself an attribution of native truth error, so this does
not justify tuning the candidate grid or claiming an li-exact operation would raise S. The
audit took **31.32 s**, requested no EFIT target columns, and touched no internal gate200,
evaluation tail70 or public shots.
[Input/output hashes, distribution and top frames](results/oof_li_discrepancy_20260923.json).

**2026-09-23 UTC — full scorer-normal displacement against the failed K8 pilot:**
On the exact same first **100** original-TRAIN OOF shots/**21,667** frames, the only changed
operation was retaining all **512** normal-displacement components instead of Fourier DC plus
modes 1–8. The fixed additive ψ correction, input-only source-li compensation, submitted scalars
and native scorer were unchanged. The pinned original source scored **S 0.991298**; K8 scored
**0.991736** (**+0.000438**) and the full-normal arm **0.992090** (**+0.000792**).
High-frequency detail thus added **+0.000354**, but the full arm still missed the earlier
**+0.002600** investment margin. Raw D_LCFS improved **0.013430 → 0.006741**, Consistency
**0.969871 → 0.970681**, and li R² still fell **0.938689 → 0.884629**. No native LCFS or
derived failures; q95/βN arrays remained byte-identical. Source snapshot preparation took a
few seconds; the full-normal map/profile build took **749.59 s** and one native scorer pass
**68.06 s**; the exact source control JSON was reused by hash. Verdict: this constrained
normal/additive/profile family has too little observed score headroom on the OOF TRAIN pilot to
justify folds 1–4 or a correction learner under the frozen recipe. This is neither a bound on
all contour methods nor an evaluation-tail/public result.
[Full native terms, hashes and per-shot movement](results/oof_full_normal_diagnostic_20260923.json).

**2026-09-23 UTC — fivefold OOF compact-MLP K8 normal-contour pilot, stopped after fold 0:**
One frozen fold-0 MLP trained for exactly **33,000** steps on **5,416** allowed shots and
**1,200,575** full frames, excluding its own **1,355** OOF shots, the separate 200-shot native
gate and the original 70-shot evaluation tail. Its first **100** original-TRAIN OOF shots
(**21,667** full frames) supplied a truth-assisted K8 normal-displacement pilot. K8 retained
**0.957341** of pooled normal energy, with median low-pass residual **0.000889 m** against a
**0.026562 m** minimum grid spacing, so its representation gate passed. The unchanged native
source scored **S 0.991298**; the K8 truth-assisted additive map with input-only source-li
compensation scored **0.991736**, just **+0.000438** against a frozen **+0.002600** investment
margin. D_LCFS improved **0.013430 → 0.008043**, while Consistency fell
**0.969871 → 0.969525**; li R² fell **0.938689 → 0.884688**. Submitted q95/βN arrays were
byte-identical and the native scorer reported zero LCFS or derived failures. Building/verifying
the allowed-only 6,771-shot auxiliary cache took **1,065.49 s**, the base fit **214.7 s**
(optimizer **77.6 s**), map/profile construction **757.45 s**, and the two native scorer passes
**69.00/69.25 s**. Verdict: this fixed K8 branch failed before the other four base fits, the
correction learner, original tail70 or any public artifact. The pilot is on TRAIN shots with a
base prediction that did not train on them; it does not prove all contour learners lack headroom.
[Frozen files, hashes, full native fields and timing](results/oof_normal_k8_pilot_20260923.json).

**2026-09-23 UTC — input-only full-coordinate LCFS MLP, one frozen fit:**
The predeclared Stage-0 truth-assisted gate passed, so one 209→512×3→1024 MLP trained on
**5,633** compact TRAIN shots (**249,737** thinned frames) with **1,338** validation shots
(**59,199** thinned frames); the original salt-0 tail70 was untouched. Its input-only predicted
contours were applied at fixed half strength to the compact source on the first **200** validation
shots/**44,911** full frames. All predicted corrections were finite, with **zero** source
fallbacks or native scorer failures; the corrected-to-intended contour median/p95 distance was
**0.333/1.403 grid cells**, inside the predeclared geometric limits. The unchanged source scored
native **S 0.991623**, learned map **0.990890**, and learned map plus input-only source-li
compensation **0.991350**, a final **−0.000273** against the required **+0.001300**. Raw D_LCFS
was **0.012443 → 0.012526**, Consistency **0.970709 → 0.969531**; submitted q95/βN were
bit-identical. Cache construction took **2,291.63 s**, memmap assembly **10.24 s**, one GPU fit
**89.22 s** (best validation checkpoint at step 8,000), prediction/correction build
**1,413.66 s** and the three sequential native scorer passes **438.02 s** total. Verdict: this
fixed full-coordinate learner did not transfer the truth-assisted contour headroom to an
input-only gain. Stop before reserved tail70 evaluation, public output or a parameter retry.
This early-stopping validation window is a selection screen, not independent generalization.
[Native fields, source/config/cache hashes and timing](results/direct_full_contour_stage2_20260923.json).

**2026-09-23 UTC — fixed half-contour mechanism gate before any direct-polygon MLP:**
On first **200 compact validation shots/44,911 full frames**, the unchanged compact two-member
source had native **S 0.991623**. Replacing each predicted contour displacement with a fixed
**half** shift toward scorer-native TRUE validation LCFS, through actual additive ψ maps, gave
**0.993561**; fixed source-derived li preservation gave **0.993861**, a preregistered Stage-0
effect **+0.002238** and a PASS over +0.001300. Raw D_LCFS **0.012443 → 0.006948**;
Consistency **0.970709 → 0.979143**; ψ **0.999732 → 0.999734**; q95/βN bytes exact.
All **44,911** maps had native LCFS, no source fallback or derived failure, and the scorer's
corrected-to-intended-contour distance had median/p95 **0.158/0.981 grid cells**. Work took
**3.35 s** frozen preparation, **53.69 s** input-only compact inference,
**1,414.10 s** contour/profile construction and **145.12/146.16/145.92 s** for the three
sequential native scorer passes. Verdict: this fixed operation has selection-window mechanism
headroom sufficient to warrant testing one full-resolution input-only contour learner. True
contours were used to construct the treatment; the compact source already used these validation
shots for early stopping. The result is neither an achievable model gain nor a public-fold
forecast. No new MLP, tail70 evaluation or submission followed.
[Full native fields, hashes and stage timing](results/direct_full_contour_stage0_20260923.json).

**2026-09-23 UTC — true magnetic O-point as fixed polar LCFS origin:**
On the same **100 compact TRAIN shots/300 fixed frames**, the scorer-native O-point of each true
ψ map replaced the polygon centroid; the 512 shifted ray angles and tolerances were unchanged
and frozen before measurement. All **300** O-points lay strictly inside, yet **9 of 153,600**
rays on nine frames had three distinct positive LCFS intersections. The other **153,591** had
exactly one, with no missing or collinear rays. The narrowest distinct radius gap was **174×**
the predeclared shared-vertex tolerance, so this is genuine local concavity, not an endpoint
artifact. Preparation took **1.00 s** and the CPU probe **4.40 s**. Verdict: the one-O-point
positive-radius chart fails its strict gate, closing this simple star-shaped branch before any
log-radius PCA, third TRAIN sample, learner fit, evaluation-tail score or public submission.
[Frozen sample, scorer identities and nine pathological frames](results/direct_contour_opoint_20260923.json).

**2026-09-23 UTC — fixed centroid-ray star-shapedness of scorer-native LCFS:**
The same **100 compact TRAIN shots/300 fixed frames** supplied true scorer LCFS polygons; the
half-bin-shifted 512-ray construction froze its centroid, intersection and shared-vertex rules
before measurement. All **300** centroids were strictly inside. **153,596/153,600** rays had one
unique positive hit, but four rays on four different shots had **three**; zero rays had no hit or
a collinear ambiguity. Each failing ray intersected three actual polygon segments at interior
positions, so endpoint deduplication cannot erase the concavity. Preparation took **1.00 s** and
the probe **4.24 s**. Verdict: the strict 100% star-shaped gate fails for this centroid-origin
single-chart representation; no log-radius PCA, third TRAIN sample, learner fit, evaluation-tail
score or public submission. This does not refute topology-safe contour modeling generally.
[Per-frame identities and source hashes](results/direct_contour_star_20260923.json).

**2026-09-23 UTC — single public EFIT-AI HDF5 sample schema/access audit:**
GitLab project 36523277 exposes one checked-in DIII-D example, `179565.h5` (**19,572,048 bytes**, SHA
`d7764e6a…`), at a pinned public commit. The file has **13** equilibrium/Thomson times,
**129×129** ψ, q95/βN/Ip labels, **53** Thomson channels with Te/ne and coordinates, and
**24** PF-current constraints per equilibrium slice. It does not expose the challenge's named,
full-rate **18 F-coils + ECOILA/bcoil/Ip** streams. The public `dvc` branch has metadata pointers
to SSH remotes at NERSC/GA, not an unauthenticated bulk corpus. Repository BSD text was found,
but a dataset-wide HDF5 license and shot-disjointness evidence were not. Verdict: this sample
cannot supply a compatible clean external training set; no bulk download, public/private shot
fingerprinting or model fit. [Pinned source, fields and access evidence](results/efitai_sample_feasibility_20260923.json).

**2026-09-23 UTC — can a fixed mean-line projection rescue one failed PCA contour?**
Read-only on the sole non-simple held TRAIN frame, PCA64 had one crossing between polygon
segments 349 and 357 at **(R,Z)=(1.371973,−1.060385) m**. Its fitted PCA mean was simple and
in-grid. Along the fixed line from reconstruction toward that mean, the first simple polygon
at **t=0.875** raised scorer-native physical Hausdorff error **0.005393 → 0.014768 m**
(**0.203 → 0.556 grid cells**), beyond the earlier half-cell gate; every later dyadic point was
worse. Verdict: a topology repair of this constrained form loses too much geometry. No third
TRAIN sample, model fit, evaluation-tail score or gate relaxation. The result says nothing about
all topology-preserving methods. [Frozen basis and nine-point diagnosis](results/direct_contour_pca_topology_20260923.json).

**2026-09-23 UTC — official-source external DIII-D data feasibility, no model run:**
The challenge permits public external tokamak archives with disclosure, and EFIT-AI describes
2019 DIII-D reconstructions. The official material reviewed did not yield a verified public
shot-level download with the challenge's allowed currents, Thomson profiles and ψ labels, a
data license, and proof of disjointness from anonymous public/private competition shots.
Verdict: no external supervised corpus can be treated as ready for this fork on current evidence;
no large files or labels were downloaded. [Primary links and missing gates](results/external_diii_d_data_feasibility_20260923.json).

**2026-09-23 UTC — disjoint TRAIN-shot physical-contour PCA gate:**
One full-SVD PCA basis was fit to 300 fixed scorer-native LCFS contours from the first 100
compact TRAIN shots. A separate next 100 TRAIN shots/300 fixed frames measured two predeclared
ranks, with no validation or reserved tail used. PCA32 reconstruction had median/p95 native
Hausdorff **0.006816/0.015145 m** (**0.257/0.570 grid cells**) and no topology failures,
so it missed both frozen distance limits. PCA64 had **0.002487/0.006200 m**
(**0.094/0.233 cells**) but one reconstructed contour in 300 was non-simple, above the zero
failure allowance. Neither rank passed the combined gate; a high L² reconstruction ratio was
again insufficient by itself. Preparation took **2.05 s**, basis contour extraction **3.75 s**,
PCA fit **0.030 s**, held TRAIN contour extraction **4.05 s**, and the whole probe **15.76 s**.
Verdict: stop this fixed low-dimensional absolute-contour family before full target extraction
or a correction learner. No threshold repair, third rank, eval70 score or public submission.
[Frozen basis/check samples and failed-frame identity](results/direct_contour_pca_20260923.json).

**2026-09-23 UTC — train-only absolute LCFS Fourier representation gate:**
Following the user decision to reserve all original 70 tail shots for evaluation only, a frozen
calibration extracted scorer-native true LCFS on **300 fixed frames from 100 compact TRAIN shots**.
No base-model prediction or held-tail label was read. K8 reconstruction retained **99.9927%**
of pooled centered-shape L² energy but had median/p95 symmetric Hausdorff error
**0.043215/0.051528 m** (**1.627/1.940 grid cells**); K16 retained **99.9990%** but still
missed by **0.021218/0.028415 m** (**0.799/1.070 cells**). Both had zero self-intersecting or
out-of-grid reconstructed polygons. Neither met the frozen median≤0.25-cell and p95≤0.5-cell
gate. Preparation took **1.65 s** and the train-only probe **6.00 s**. Verdict: these two smooth
absolute Fourier targets erase scorer-sensitive boundary detail despite excellent L² fidelity;
stop this family before any full target extraction or fit. This is not an S score or a general
negative for learned contours. [Frozen sample, scorer hashes and per-K results](results/direct_contour_representation_20260923.json).

**2026-09-23 UTC — fixed input-only contour-displacement learnability screen:**
After the user delegated the split choice, the preregistered first **35** salt-0 tail shots
(7,267 frames) supplied only supervised contour targets; last **35** (7,563 frames) used
target-stripped inputs until predictions were sealed. K8 normal Fourier retained **93.915%**
of training normal-displacement energy; median reconstruction RMS was **0.000494 m** against
one grid cell **0.026562 m**, so the representation gate passed. One 216-feature weighted Ridge
(α=35, no search) then predicted a contour correction. Frozen native S on the evaluation35:
original best11 map **0.994876**; predicted additive map **0.994813** (**−0.000063**);
that map plus deterministic source-li preservation **0.994841** (**−0.000035**).
Raw D_LCFS was **0.009526 → 0.009702 → 0.009703** and Consistency
**0.981311 → 0.981086 → 0.981245**. q95/βN were bit-identical and native extraction
had zero failures. Target/representation work took **132.64 s**, Ridge fit **0.031 s**,
input-only map construction **65.78 s**, source-li postprocessing **502.12 s**, and three
sequential native scorers **52.01 / 51.77 / 51.98 s**. An initial frozen snapshot stopped at
an import-path provenance check after some training parquet columns had been loaded, before
target-contour extraction or a fit; the repaired v2 snapshot completed without changing the
protocol. Verdict: **fails the fixed +0.001 screen gate**. Good low-dimensional target
representation did not translate into useful prediction from these permitted inputs under this
one Ridge specification. This repeatedly observed tail is not independent confirmation or a
public-fold estimate. No K, α, feature, weight or model retry and no public submission.
[Full native fields, source and filename hashes](results/contour_learnability_screen_20260923.json).

**2026-09-23 UTC — source-derived li preservation after additive contour correction, full tail:**
The fixed one-grid-cell inside-polygon taper and nine monotone a values targeted li computed
from the ORIGINAL predicted ψ/C_source, not EFIT truth. On the same **70 shots/14,830 frames**,
mean absolute error to that source-derived target fell **0.016031 → 0.007934**. A nonzero a was
chosen on **3,468** frames; **50** nonzero candidate evaluations were explicitly rejected
(39 axis shifts, 11 contour shifts), and a=0 remained valid everywhere. Against the previously
scored truth-assisted additive-map control, native **S 0.996257 → 0.997035**
(**+0.000778**): li R² **0.939465 → 0.967447**, Consistency
**0.985709 → 0.989704**, R²ψ **0.999867 → 0.999828**, D_LCFS
**0.003863 → 0.003861**, and R²qb stayed **0.997165**. No LCFS/derivation failures occurred.
Compensation build and verification took **3,300.46 s**, including **1,825.22 s** for exact
polygon-segment distances, plus about **98.2 s** for one frozen native scorer pass. Verdict:
deterministic source-li targeting works on this reused tail, but the preceding contour correction
still consumed true labels, so the combined score is not deployable or an independent/public
result. No parameter-grid expansion, model fit or public submission was made.
[All native terms and per-shot target movement](results/geometry_source_li_full70_20260923.json).

**2026-09-23 UTC — source-li target compensation, first-three-shot mechanism screen:**
The target for every profile parameter is `internal_inductance` of the ORIGINAL predicted ψ,
not the true EFIT li or a stored label. A first global-u formulation found that vacuum grid
edges can satisfy `u∈[0,1]`: all **4,624** nonzero candidates changed boundary ψ bits, so the
exact a=0 control was selected on all **578** frames. A predeclared spatially restricted version
kept the same nine monotone a values and multiplied only the interior flux difference by a
one-grid-cell smoothstep of exact polygon-segment distance. All nine candidates then passed
the frozen validity checks on all 578 frames; 263 frames selected nonzero a, and mean absolute
error to the source-derived li target fell **0.0190 → 0.0094**. The native score on these
three shots moved **S 0.981702 → 0.980183** and true li R² **0.610578 → 0.562825**, as the
truth-assisted additive map was already better than the original source on this subset.
This is a correctness/mechanism result, not a reason to retune a or infer a full-70/public gain.
The additive contour stage still uses true labels. [V1 null](results/geometry_source_li_v1_null_20260923.json),
[spatial v2 smoke](results/geometry_source_li_v2_smoke_20260923.json).

**2026-09-23 UTC — fixed contour-local additive ψ correction on the full labeled tail:**
The scorer-native C_pred/C_true contours supplied one fixed TPS displacement; a Gaussian shell
of **two physical grid spacings** and **λ=1** applied the first-order map update
`ψ_new=ψ_old−w u·∇ψ_old`. q95/βN remained bit-identical. An initial full-70 run stopped on a
four-grid-diagonal displacement cap inherited from the old *inverse* warp. On its offending
frame the unchanged additive map was finite and edge-exact, so a preregistered v3 removed only
that unrelated cap; all nine three-shot prediction arrays and the native JSON remained bit-exact.
The completed **70 shots/14,830 frames** scored native **S 0.995080 → 0.996257**,
**+0.001177**, with D_LCFS **0.009378 → 0.003863**, Consistency
**0.982616 → 0.985709**, kappa **0.965355 → 0.994937** and volume
**0.983522 → 0.999592**. Li deteriorated **0.972388 → 0.939465**; R²qb was unchanged
at 0.997165, and no LCFS/derivation failures occurred. Raw scorer-native contour-to-true
distance improved on **13,776/14,830** frames. Build+verification took **552.13 s**, then
control and candidate native scoring about **98 s** each. Verdict: real map-space geometry
sensitivity under a fixed **truth-assisted** construction, not a deployable predictor or an
independent/public gain. The repeatedly used tail and unavailable C_true at inference prohibit
that interpretation. [Full terms and per-shot contour movement](results/geometry_additive_full70_20260923.json),
[initial gate measurement](results/geometry_additive_v2_failure_20260923.json),
[bit parity](results/geometry_additive_v3_parity_20260923.json).

**2026-09-23 UTC — fixed geometry map families closed after full-70 numerical gates:**
The truth-assisted A contour warp with boundary taper and damped Newton produced **578/578**
valid maps on the preregistered first three shots. Safety-only v7 maps and native JSON were
bit-identical to v6; native S was **0.944391 → 0.965705** on those same three shots, with
D_LCFS **0.013572 → 0.003870** and kappa R² **0.106798 → 0.940405**, but li R² fell
**0.554423 → 0.093443**. This uses true contours and a small, atypical fold, so it is a
mechanism/correctness result only. Its sole v7 full-70 attempt stopped after ten complete shots
at shot index 11 frame 51: dense physical Jacobian condition **11.619 > 10**, a distinct frozen
safety failure. No full-70 A map or S was produced; this fixed invertible-warp family is closed
without raising the gate. The nine-value B remap also passed the three-shot smoke, then stopped
on full-70 v2 contour-level proxy and v3 scorer-native LCFS identity gates, with no full-70 map
or score. These are numerical feasibility failures of fixed constructions, not evidence that all
contour or profile corrections are ineffective. [A smoke](results/geometry_A_v6_smoke_20260923.json),
[safety parity](results/geometry_A_v7_parity_20260923.json),
[final A failure](results/geometry_A_v7_failure_20260923.json),
[final B failure](results/geometry_B_v3_failure_20260923.json).

**2026-09-23 UTC — three-shot geometry map mechanism smoke and numerical gates:**
The fixed first three fit-excluded DIII-D training-corpus shots contain 578 frames. The unchanged
map/control scored native **S 0.944391** on this tiny subset. The v2 truth-assisted contour warp
failed on the first frame because raw TPS displacement leaked at the grid edge and its inverse
went out of bounds; no A map or score was published. The independent nine-value interior li
remap made 578 valid maps, kept q95/βN bit-exact, and scored **S 0.953937** on the same shots;
scorer-derived li R² moved **0.554423 → 0.862264**. This is a correctness/mechanism check,
not a full-tail or deployable-model gain. A separately preregistered full-70 B attempt stopped
after five complete shots at frame 5 of shot index 6, when the resampled contour's bilinear
flux spread exceeded a numerical proxy by 1.3%; no B map or score was published. The amended
B spec drops only that proxy, preserving all nine post-map contour/axis checks. Verdict: B is
feasible on the small smoke; A and the first full-70 B attempt have numerical feasibility
failures, not scientific negatives. [Smoke](results/geometry_map_smoke_20260923.json),
[A edge diagnosis](results/geometry_A_boundary_20260923.json),
[B proxy diagnosis](results/geometry_B_v2_failure_20260923.json).

**2026-09-23 UTC — private HF float32 HEAD restored and local f16 bulk removed:**
After the f16 probe, the private dataset's DIII-D file was restored to the verified float32
SHA-256 `7b39cdc94ec8b95b2c89dfbee4c0fcd3db4898e2671ac3e0ac136efed5b5d704`, with
the existing MAST and other files unchanged. The original float32 pointer still pins its
previous valid commit; no new pointer or Codabench submission was made. Locally generated and
downloaded f16 NPZs plus the f16 pointer copy were deleted after hash checks, freeing
3,069,182,365 bytes including temporary HF/Xet cache. Historical remote commits, score pages
and small tracked evidence remain. [Restoration](results/private_hf_f32_restore_20260923.json),
[cleanup](results/f16_local_cleanup_20260923.json).

**2026-09-23 UTC — explicit best-eleven float16 flux-only probe:**
The fixed eleven-map/prior-eight-scalar float32 prediction was cast once to float16 **only for
ψ**; q95/βN remained bit-identical float32 and no member was refitted or reweighted. On the
exact same 70-shot tail through the frozen native scorer, S **0.995080 → 0.994704**
(**−0.000376**), with R²qb unchanged at 0.997165, raw D_LCFS **0.009378 → 0.010053** and
Consistency **0.982616 → 0.981072**. The full 874-shot quantized public NPZ passed the local
validator and was uploaded as a separate private HF revision with a labelled pointer; the prior
float32 pointer stays pinned. A strict preflight rejected a truncated public source SHA in the
first spec before conversion; the v2 spec records this hash-only correction.
[Public submission 941187](https://www.codabench.org/api/submissions/941187/get_detail_result/)
subsequently scored **0.993731**, versus **0.994082** for the float32 control 941038:
**−0.000351**. Public R²ψ and R²qb were unchanged at reported precision, raw D_LCFS worsened
**0.010898 → 0.011618**, Consistency **0.979525 → 0.978130**, and MAST was unchanged.
Verdict: this controlled float16 cast reduced both local and public score; it is not a production
export improvement. [Native/artifact evidence](results/best11_f16_probe_20260923.json),
[HF receipt](results/private_hf_f16_probe_20260923.json),
[official public fields](results/best11_f16_public_20260923.json).

**2026-09-23 UTC — previous private DIII-D prediction versus current public candidate:**
The private Hugging Face dataset's parent commit `836da676877283aa8fe0608dc501ffe49f5e072e`
held a 1,372,856,730-byte DIII-D NPZ (SHA-256 `46aeb3bf5930ca621a3ba47b8189f72b454dbd49a7b2f1a5893b75d9df6e71de`),
last modified 2026-09-04 22:45:28 UTC. The new uploaded NPZ and parent have exactly the same
874 shot keys, 193,359 frame counts and 2,622 array shapes; no values are nonfinite. Old ψ is
float16 and new ψ float32, but only **7.1893%** of old ψ values equal new ψ after quantizing the
new array to float16. Mean absolute old-minus-new differences are **0.0006157** for ψ,
**0.02519** for q95 and **0.01176** for βN. Both scalar fields are float32 in both files.
The old pointer basename in a screenshot is close in time to the parent commit, but its manifest
revision is unavailable, so this does **not** prove that the parent file is submission 915662.
Verdict: the old and new public-input predictions differ beyond dtype; without labels this
comparison makes no performance claim. Full hashes, finite/shape checks and 23.8-second comparison
time: [results/old_vs_new_private_hf_20260923.json](results/old_vs_new_private_hf_20260923.json).

**2026-09-23 UTC — eleven-map/eight-scalar candidate on the public leaderboard:**
Codabench [submission 941038](https://www.codabench.org/api/submissions/941038/get_detail_result/)
scored DIII-D **S 0.994082** over 874 shots, versus **0.994073** for the prior
[submission 915662](https://www.codabench.org/api/submissions/915662/get_detail_result/).
The change **+0.000009** combines R²qb **−0.000550** with lower D_LCFS by **0.000157** and
Consistency **+0.000384**; R²ψ moved −0.000001. The 2026-09-23 10:13 UTC
[public leaderboard](https://www.codabench.org/api/phases/29580/get_leaderboard/?page=1&page_size=100)
ranked the new row **10th of 69**. MAST S remained 0.568092. Verdict: the selected local
70-shot gain did not translate into a material public S improvement; no single member or scalar
block has been isolated as the cause on the public fold.

**2026-09-22 UTC — public leaderboard reference corrected:**
The public Codabench phase 29580 API returned submission **915662** owned by `pshishkin`, created
2026-09-05 08:33 UTC, with method label **“6 salts”** and DIII-D **S 0.9941** to four decimals.
It ranked 8th in that retrieved response; third place was 0.9954. The earlier
`baselines/diii_d_20260904.json` described a three-salt recipe inferred from repository scripts,
not this row's verified fitted composition. Original submitted artifact hashes, exact weights
and code revision remain unavailable. Verdict: the public row is the reference, and no current
local artifact/recipe is certified as its reproduction. Time-stamped API source and row fields:
[results/leaderboard_reference_20260923.json](results/leaderboard_reference_20260923.json).

**2026-09-22 UTC — offline Challenge1-only delivery archive:**
The locally validated float32 DIII-D public-input NPZ was packaged as a root-level
`ZIP_STORED` archive containing only `diii_d_public_test.npz`. The archive is **2,967,287,355
bytes**, SHA-256 `32044c6a70cf83893afa29e055bf260aa8776693f540e4e6a0b7e206468cf9f0`;
its sole entry passed CRC32 `24302ceb`, and the extracted entry SHA equals the validated NPZ's
`7b39cdc94ec8b95b2c89dfbee4c0fcd3db4898e2671ac3e0ac136efed5b5d704`.
Creation/entry verification/end-to-end walls were **3.49 / 2.58 / 9.98 s**. Current official
[submission-format rules](https://www.codabench.org/competitions/17739/#/pages-submission-format)
permit a Challenge1-only root ZIP under the 15 GB quota, but an archive without MAST cannot
preserve a Challenge2 score. No authenticated Codabench/HF route is available in this workspace;
there was no upload, public score or push. Verdict: a reviewable offline Challenge1 package, not
a leaderboard submission. [results/challenge1_offline_zip_20260923.json](results/challenge1_offline_zip_20260923.json).

**2026-09-22 UTC — every member of the recovered older five, scored individually:**
`tools/historical_compare.py` used the archived original inference revision `eb70e25` for the
already fitted `artefacts/20260815-prod-4seeds-seq.joblib`, then the pinned native scorer on the
same 70-shot tail. No member was retrained or selected. Native solo S: `mlp` **0.993140**,
`mlp1` **0.991943**, `mlp2` **0.990809**, `mlp3` **0.991653**, `seq` **0.992175**; their original
equal-weight five-member ensemble scored **0.994478**. Four MLPs and sequence are ensemble
members; the artifact's fitted ridge has zero ensemble weight. Prediction walls totalled
**82.25 s** and five separate CPU4 native scorer walls **492.39 s**. Verdict: the ensemble's
gain over every member is directly measured on these shots; these are descriptive scores on a
reused tail, not a new fit or independent confirmation. Full native fields, exact filenames,
source/artifact hashes and phase times: [results/historical_five_members_20260922.json](results/historical_five_members_20260922.json).

**2026-09-22 UTC — B11 pooled versus position-conditioned MLP loss, production screen:**
`run prepare experiments/position-metric-screen.json` froze the 0.80/1.0 train, 0.19/1.0
validation, 0.01 scored salt-0 control and one fixed-strength candidate. Both arms used the same
100 complete training-shot Jacobian triplets, probe step 0.060919 Wb/rad, three group counts
375402/497118/375937, pooled scalar variance, global target scaler, PCA/input transforms and
global-MSE stopping. Every artifact probe/form metadata field was hash-identical except
`position_metric.mode`; the candidate alone rotated only its training quadratic by λ=0.5. All
scores below use the same 70 unseen shots. The original compact decoded MLP+sequence anchor is
S 0.993386, but **the paired B11 comparison is candidate versus its own pooled control**:

| B11 arm | MLP alone S | 50/50 arm MLP + original sequence S | Fixed old-five + arm 5:1 S |
|---|---:|---:|---:|
| pooled control | 0.991412 | **0.993584** | **0.994586** |
| position-conditioned | **0.992075** | 0.993294 | 0.994523 |

Thus conditioning improves MLP alone by **+0.000663** but reduces the paired compact mixture by
**−0.000290** and the fixed old-five addition by **−0.000063** versus pooled control. Compact
replacement terms pooled/conditioned were R²ψ 0.999766/0.999778, R²qb 0.995809/0.995703,
raw D_LCFS 0.011007/0.011277 and Consistency 0.977213/0.975943. Fit-only times were
503.9/395.8 s; child walls 741.24/649.32 s, with concurrent public874 CPU4 preparation. The
two compact scorer comparisons took 105.15/110.15 s. Separate old-five additions at fixed 5:1
both beat old-five S 0.994478 slightly (+0.000108/+0.000045), but on the reused tail neither is
accepted and the conditioned arm adds less. Verdict: the direction mechanism survives its input
diagnostic, yet this λ=0.5 row-dependent loss does not improve either tested ensemble comparison;
no extra strength/grid is inferred. Full scorer fields, exact YAML, shared probe hashes and
filenames: [production screen](results/position_metric_screen_20260922.json),
[old-five additions](results/historical_five_position_metric_20260922.json).

**2026-09-23 local preparation — fixed eleven-map/eight-scalar DIII-D public candidate:**
`tools/public_blend.py` decoded six frozen predictors on all **874** pinned, SHA-verified
`diii_d_public_test` input shots, with no target columns, and assembled the preregistered map
weights **5:1:1:1:2:1** and earlier eight-member q95/βN weights **5:1:1:1**. The local float32
NPZ contains **2622** arrays over **193359** frames, 2,967,287,097 bytes, SHA-256
`7b39cdc94ec8b95b2c89dfbee4c0fcd3db4898e2671ac3e0ac136efed5b5d704`.
`validate_submission.validate` passed all 874 DIII-D shots. A held-70 dry run of the same
streaming assembler reproduced all **210** float32 arrays bit for bit from the previously scored
S 0.995080 diagnostic. Observed wall: source predictions **1354.69 s**, final assembly
**158.55 s**, local validation **57.45 s**, total **1579.57 s**; the host concurrently ran B11
GPU fitting, so these are not isolated throughput benchmarks. Verdict: the DIII-D public-input
artifact is locally built and structurally validated, but has **no public score**. MAST was not
built or replaced, and there was no ZIP, upload or push. Exact sources, ordered filenames, input
manifest and output hashes: [results/public_diii_d_candidate_20260923.json](results/public_diii_d_candidate_20260923.json).

**2026-09-22 UTC — fast magnetics A/B production screen with fresh unchanged control:**
`run prepare experiments/fast-signal-screen.json` froze the new evaluator and the production
`0.80/1.0 0.19/1.0 0.01` salt-0 three-arm campaign. The separately keyed auxiliary targets were
rebuilt from the new scorer producer in **1171.48 s**; full-shot fast sidecars took **33.70 s** for
A and **39.90 s** for B. All 6971 train-plus-validation sidecars per spec were checked for full
frame counts, finite float32 values, frozen producer keys and pinned parquet digests. A fresh
MLP control without new columns reproduced **every native scorer field** of the original compact
MLP, S **0.992149**, on the same 70 unseen shots. The candidate comparisons held the original
compact sequence fixed at weight 0.5 and used the matched decoded control **S 0.993386**:

| Added input columns | Candidate MLP S | Replacement S | ΔS vs decoded control | R²ψ / R²qb / D_LCFS / Consistency of replacement |
|---|---:|---:|---:|---|
| A: `ECOILA` centered ±5 ms dI/dt | 0.991120 | 0.992938 | −0.000448 | 0.999732 / 0.995813 / 0.010973 / 0.974053 |
| B: A + corrected plasma-current ±20 ms dI/dt | 0.991114 | 0.993323 | −0.000063 | 0.999758 / 0.995494 / 0.011468 / 0.976392 |

Fit-only times were 161.8 / 217.2 / 307.2 s for control/A/B; full child walls were 468.32 /
416.49 / 497.01 s, and the two four-row matched comparisons took 109.59 and 115.02 s.
Verdict: this narrow input bank did not improve the compact mixture on one salt; B repairs most
of A's loss but does not beat the unchanged control. No candidate is promoted. The input-only
audit's distinct variance was not proof of useful signal. Native scorer terms, exact YAML,
weights, files and artifact hashes: [screen](results/fast_signal_screen_20260922.json),
[frozen cache preparation](results/fast_cache_prewarm_20260922.json).

**2026-09-22 UTC — single temporal-convolution sequence member:**
`run prepare experiments/seq-conv-production.json` from the frozen pre-feature source changed
only `models.seq.seq_cell=conv`; constant learning rate, original features/loss, seed 0 and the
existing 1,2,4,8,16,32 dilations stayed fixed. On the same 70 unseen shots the candidate scored
**S 0.990636** alone versus the original GRU sequence's 0.990514. The matched decoded 50/50
mixture with the unchanged compact MLP scored **0.993279** versus **0.993386** for the original
MLP+GRU pair, **ΔS −0.000107**. Replacement terms were R²ψ 0.999775, R²qb 0.995673,
raw D_LCFS 0.011232 and Consistency 0.975873. Fit-only time was 387.6 s, best step 22500
after 32500; child wall 580.32 s. Verdict: the historical residual-correlation diversity
motivation did not translate to a compact 50/50 replacement gain on this split. A separate
preregistered fixed 5:1 addition to the recovered old five scored **0.994615** against
**0.994478**, **+0.000137**; terms were R²ψ 0.999840, R²qb 0.996684, raw D_LCFS 0.009805
and Consistency 0.980904. That is a small same-tail diversity diagnostic, not an independent
confirmation or automatic promotion. Native evidence and frozen config:
[compact replacement](results/seq_conv_screen_20260922.json),
[old-five addition](results/historical_five_conv_20260922.json).

**2026-09-22 UTC — expanded old-ensemble stack and a scalar-block control:**
One preregistered decoded-map blend added the two-model constant-rate refit and the cosine
sequence fixed-step refit to the previously measured eight-member old5 + `cons_gain=2` MLP +
`env_weight=1` MLP + cosine sequence collection, at fixed member weights **5:1:1:1:2:1**.
On the same already reused 70-shot tail, the full eleven-member blend scored **S 0.995046**.
Keeping those eleven members' flux maps but taking q95/βN bit for bit from the prior eight-member
blend scored **S 0.995080**, versus **0.994872** for the prior eight and **0.994478** for the old
five. The block-spliced result's native terms were R²ψ 0.999855, R²qb 0.997165, raw D_LCFS
0.009378 and Consistency 0.982616. The exact map/scalar audit checked 70 map arrays and 140
scalar arrays. Verdict: the expanded map adds a local geometry gain while its full scalar block
gives some back; this is an adaptive diagnostic after repeated use of these shots, not independent
confirmation, a production weight search, or a submitted leaderboard result. The fixed grid,
source/artifact weights, predictions, hashes, files and native scorer fields are in
[results/historical_eleven_blocks_20260922.json](results/historical_eleven_blocks_20260922.json).

**2026-09-22 — export precision audit and writer repair:**
`tools/precision_audit.py` changed only the flux-map serialization of two existing 70-shot
decoded prediction sets, leaving q95/βN bits, filenames and scorer fixed. The recovered old five
scored **S 0.994478 in float32** and **0.994100 after the old float16 cast**, ΔS **−0.000378**;
its Consistency fell 0.001556 and raw D_LCFS rose 0.000667, while R²ψ and R²qb were unchanged at
native six-decimal precision. The current eight-member diagnostic stack scored **0.994872** in
float32 and **0.994475** after float16, ΔS **−0.000397**. This is deterministic serialization
loss on the same predictions, not fit noise or independent leaderboard evidence. The corrected
`submission_skeleton.py` local and streaming writers now preserve float32 flux and scalars.
An actual writer round-trip retained all **210** old-control arrays byte for byte; the native
scorer reproduced every result field and S 0.994478. MAST and per-salt caches carry a new output
format version and reject stale float16 arrays, and per-salt blending accumulates in float64 with
one final float32 cast. Verdict: remove the measured export loss end to end; do not upcast an old
float16 cache and call precision restored. Evidence:
[quantization audit](results/float16_export_audit_20260922.json),
[float32 writer parity](results/float32_export_parity_20260922.json).

**2026-09-22 — cosine sequence fixed-step refit on the train-plus-validation union:**
`python -m my_experiments.run prepare experiments/seq-cosine-refit.json` used the frozen source of
the prior refit, then `run execute` fitted only the cosine sequence on the exact ordered 6971-shot
union with zero validation shots. The original cosine sequence selected step 56500 with horizon
80000 on 176 full shot batches per pass; the 217-batch union fixed step **69662** and horizon
**98636** before fitting. The model kept its last step. On the identical 70 unseen shots, its
individual native **S 0.992808** versus **0.992205** for the earlier 80%-train cosine sequence;
terms were R²ψ 0.999797, R²qb 0.991521, raw D_LCFS 0.010604 and Consistency 0.976261.
q95/βN fell while geometry improved. Fit-only time was **2000.6 s**; child train/score time
2157.04 / 58.85 s, wall 2220.03 s. The GPU fit overlapped a CPU24 rebuild of the new fast-screen
auxiliary producer and sidecar prewarm, so these are not isolated throughput benchmarks. Verdict:
the individual sequence improved on this split. A separate preregistered fixed 5:1 addition to
the recovered old five-member ensemble scored **0.994824** against **0.994478**, ΔS
**+0.000346**; terms were R²ψ 0.999849, R²qb 0.996512, raw D_LCFS 0.009478 and Consistency
0.981892. This reused the same 70 shots and is a diagnostic addition, not independent
confirmation; no collection changes yet. Complete frozen YAML, source/artifact hashes, split proof
and native fields: [sequence fit](results/seq_cosine_refit_20260922.json),
[historical addition](results/historical_five_cosine_refit_20260922.json).

**2026-09-22 — recovered older five-member control and additive diagnostics:**
`tools/historical_compare.py` loaded `artefacts/20260815-prod-4seeds-seq.joblib` with its original
inference source, decoded its four MLPs and sequence model, and scored the same 70 unseen salt-0
shots through one pinned current native scorer. The old artifact has the same 5633 train and 1338
validation filenames as the compact control, but is an August control, **not evidence of the
September leaderboard submission recipe**. Its local **S 0.994478** is the reference for the
following fixed member-count blends:

| Candidate added to old five | Artifact/member weights | S | ΔS vs old five |
|---|---:|---:|---:|
| `cons_gain=2` MLP | 5:1 | 0.994690 | +0.000212 |
| `env_weight=1` MLP | 5:1 | 0.994585 | +0.000107 |
| cosine sequence | 5:1 | 0.994688 | +0.000210 |
| convolutional sequence | 5:1 | 0.994615 | +0.000137 |
| fixed-step cosine sequence refit | 5:1 | 0.994824 | +0.000346 |
| fixed-step two-model refit | 5:2 | 0.994707 | +0.000229 |
| `cons_gain=2` + `env_weight=1` + cosine sequence | 5:1:1:1 | **0.994872** | **+0.000394** |

The accumulated eight-member blend's native terms are R²ψ 0.999845, R²qb 0.997165,
raw D_LCFS 0.009452 and Consistency 0.981641. Separately, replacing the old sequence within
the old five by the cosine sequence scored **0.994438** against its matched decoded control
**0.994479**, **−0.000041**: addition and replacement answer different questions. These are
adaptive diagnostics on the already reused 70-shot tail, not independent confirmation or a
weight search, and no collection is promoted. The old artifact hash, original inference revision,
fixed grids, decoded prediction hashes, native scorer JSON and filename checks are in
[geometry](results/historical_five_geometry_20260922.json),
[cosine](results/historical_five_cosine_20260922.json),
[stack](results/historical_five_stack_20260922.json), and
[refit](results/historical_five_refit_20260922.json), and
[cosine refit](results/historical_five_cosine_refit_20260922.json), and
[convolution](results/historical_five_conv_20260922.json).

**2026-09-22 — fast magnetics input audit and pipeline identity, before any production fit:**
`python my_experiments/diagnose_fast_magnetics.py` read only input columns from 100 train and 20
public-test shots, all 120 parquets SHA-256 verified, covering 27,270 EFIT frames. `ECOILA` raw
sampling is 0.05 ms on 71/100 train and 16/20 public shots, 0.5 ms on the rest; plasma current is
0.5 ms on all 120. Every tested centered window stayed inside raw acquisition support. On regular
interior frames, the corrected plasma-current ±20 ms derivative almost duplicates the existing
EFIT-grid derivative (correlation 0.999999999998); its novelty concentrates at edges and gaps.
`ECOILA` ±5 ms retains 29.2% train and 33.9% public-test variance after projection on existing
inputs. This is descriptive input evidence, not predictive gain; the two-arm hypothesis is in
`ideas.md`. A frozen CPU smoke fit of the two-feature path completed with 70 train / 70 validation /
14 scored shots and a trainable, statically weighted functional head. On 69 frames, full and slim
rows gave bit-identical inputs and predictions. Independently, the unchanged compact artifact
scored through the new evaluator reproduced every native and decoded scorer field on the original
70-shot tail. These are correctness checks, not a new model score. Evidence:
[input audit](experiments/fast-signal-diagnostic.md),
[smoke](results/fast_signal_smoke_20260922.json),
[old-model identity](results/fast_signal_identity_20260922.json).

**2026-09-22 — D8/S6 whole-shot aggregates with Thomson, validation-only diagnostic:**
`python -m tools.diagnose_d8_whole_shot --mode run --jobs 2` used the recovered August five-member
artifact and its original source, reading all 1,338 stored validation shots directly; 1,000 were
reserved for five-fold alpha selection and Ridge fitting, the final 338 for the predeclared gate.
Inputs were per-shot mean and population std of the exact 84 model inputs plus duration; the target
was mean psi-coefficient residual in the artifact's metric-scaled coordinates. All-input outer
explained variance was **0.005938**, against the **0.25** gate; frame-count-weighted variance was
**−0.022983**. Without Thomson, the same measurements were **−0.045878** and **−0.024142**. The
mean-only control explained 0.000000 by the same variance diagnostic. Five one-frame shots were
retained with 0 ms elapsed duration. The first extraction failed on that boundary and a second
was interrupted before fitting to add frame-weighted reporting; both pre-fit amendments and their
logs were retained. A 332-frame prediction reconstructed from old coefficients was bit-identical
to the archived old-source output. **Verdict:** the proposed linear aggregate gate fails; no
context branch or native 70-shot score follows. The 338 diagnostic shots were already used for
early stopping by the historical ensemble, so this is not independent confirmation and does not
rule out nonlinear/global context. Exact file hashes, filenames, protocol, alphas and results:
[preregistration](experiments/d8_whole_shot_prereg.json),
[result](results/d8_whole_shot_20260922.json).

**2026-09-22 — B11 shared-denominator direction replication:**
The 2026-08-16 B11 diagnostic had reported different Jacobian form directions by shot position,
but `diagnose_regime.py` used a different scalar variance denominator in each group. This
preregistered control took one frame at relative **index** positions 0.1, 0.5 and 0.9 from each of
the first 100 eligible compact-artifact training shots, all 100 with complete finite Jacobians.
All groups used the same pooled seven scalar variances and the artifact's original Consistency
split, with the rounded 0.06092 Wb/rad probe step from its training log. After trace
normalisation, the three relative Frobenius distances were **0.5260 / 0.7196 / 0.3726**, mean
**0.5394**, against **0.2837** at the 95th percentile of 100 within-shot label permutations; the
predeclared gate also required at least 0.20. Bootstrap resampling of complete shot triplets gave
a 95% interval **0.4622–0.7181**. Raw form traces were 48.28 / 25.95 / 33.51. On the SAME probes,
the old separate-denominator calculation gave mean distance **0.5653**, so the denominator
confound changes the magnitude but does not explain away the position-dependent orientation.
**Verdict:** the directional mechanism passes a controlled diagnostic; no row-dependent loss was
built or scored, and no S gain is implied. These are frame-index groups, not independently
identified physical ramp phases. Exact artifact/data/source hashes, all filenames and indices,
finite counts, top-seven angles and uncertainty:
[preregistration](experiments/b11_shared_variance_prereg.json),
[result](results/b11_shared_variance_20260922.json).

**2026-09-22 — A4 fixed-step refit on the old validation window:**
`python -m my_experiments.run prepare experiments/refit-production.json`, then `run execute` on
the frozen campaign. The 6971 full-frame train shots are the exact ordered union of the compact
control's 5633 train and 1338 validation shots; the validation block is empty. MLP and sequence
kept their last weights after the preregistered 40898 and 36989 steps, respectively. All scores
use the same 70 unseen production-tail shots, with no overlap. The candidate ensemble by itself
scored **S 0.992860** (MLP 0.992006, sequence 0.990179) versus the original compact ensemble
**0.993385**, a replacement loss of **−0.000525**. Its terms were R²ψ 0.999775, R²qb 0.995106,
raw D_LCFS 0.011501 and Consistency 0.974338.

The separately scored equal-weight blend of the two complete two-member artifacts scored
**S 0.994032**, **+0.000647** over the original; each of the four fitted models effectively
has weight 0.25. That addition gain is a different measurement from refit's failed replacement.
The ready auxiliary cache was composed in 4.50 s from verified old blocks. The refit worker's
native train/score times were 1042.27 / 94.25 s, child wall 1140.84 s; fit-only MLP/sequence
times were 101.3 / 775.9 s. A separate pinned-source comparison took 93.01 s. Independent
historical CPU4 scoring overlapped the GPU run, so these wall times are not isolated throughput
benchmarks. Verdict: this fixed-step refit does not improve the compact model on its own, but it
adds useful diversity to that small ensemble on one salt. It is not promoted to the working
control or claimed as an independent confirmation. Exact YAML, native terms, hashes and files:
[results/refit_production_20260922.json](results/refit_production_20260922.json).

**2026-09-22 — cosine sequence as a compact-ensemble replacement:**
`python -m my_experiments.run prepare experiments/sequence-cosine-screen.json`, then `run execute`
at production `0.80/1.0 0.19/1.0 0.01`, salt 0 and seed 0. Only the sequence model was fitted;
its schedule changed to cosine with the preregistered 80000-step horizon. On the identical 70
unseen files, candidate sequence alone scored **S 0.992205** against the original sequence's
0.990514. The matched decoded 50/50 mixture with the unchanged original MLP scored **0.993692**
against **0.993386** for the original pair, **ΔS +0.000306**. Replacement terms were R²ψ
0.999792, R²qb 0.996001, raw D_LCFS 0.010184 and Consistency 0.977126. Fit-only time was
**1375.3 s**, keeping step 56500 after 66500 steps; the full child train and scorer wall was
1559.86 s, and the matched four-row comparison took 110.90 s. Historical CPU scoring overlapped
the late fit, and the comparison overlapped the separate refit GPU fit, so these elapsed times are
not isolated throughput estimates. Verdict: positive compact single-salt screen, consistent with
the earlier positive multi-salt cosine result but without a paired variance estimate on this
compact replacement; the working baseline is not changed. Native scorer fields, weights, effective
YAML, scored filenames and hashes: [results/sequence_cosine_screen_20260922.json](results/sequence_cosine_screen_20260922.json).

**2026-09-22 — four geometry-loss MLP screens against the compact ensemble:**
`python -m my_experiments.run prepare experiments/geometry-loss-screen.json`, then `run execute`
on the frozen campaign, production `0.80/1.0 0.19/1.0 0.01`, salt 0 and seed 0. Each arm changed
one loss value and fitted only MLP; the sequence model remained the original artifact. All scores
below use the identical 70 unseen filenames. The replacement mixtures average decoded maps with
weights 0.5 candidate MLP / 0.5 original sequence; the matched original-member decoded control
is **S 0.993386**. The historical native compact ensemble is separately **0.993385** because
decoded averaging changes numerical order.

| Single change | Candidate MLP S | Replacement S | ΔS vs decoded control | R²ψ / R²qb / D_LCFS / Consistency of replacement |
|---|---:|---:|---:|---|
| `cons_gain=2` | 0.992381 | 0.993439 | +0.000053 | 0.999753 / 0.995553 / 0.011110 / 0.976764 |
| `cons_gain=4` | 0.991828 | 0.993123 | −0.000263 | 0.999747 / 0.995347 / 0.011008 / 0.975304 |
| `env_weight=1` | 0.992519 | **0.993530** | **+0.000144** | 0.999781 / 0.996094 / 0.011248 / 0.976805 |
| `field_weight=0.3` | 0.991210 | 0.993204 | −0.000182 | 0.999742 / 0.995706 / 0.011166 / 0.975533 |

Fit-only times were 289.4 / 300.6 / 349.7 / 218.1 s in table order. The complete child
train-and-score runs took 490.83 / 486.99 / 530.41 / 395.53 s; each replacement comparison took
about 105–106 s. Verdict: `env_weight=1` is the best single-salt screen, but its +0.000144
replacement effect has no paired variance estimate and is not promoted to the baseline. More
`cons_gain` and the field form did not improve this compact mixture on these shots. Native scorer
fields, exact YAML, weights, files, artifact/comparison hashes and all timings:
[results/geometry_loss_screen_20260922.json](results/geometry_loss_screen_20260922.json).

**2026-09-22 — replacement comparator identity:**
`python -m my_experiments.run compare-replacement` with the compact artifact as both baseline and
candidate, original `mlp` and `seq` each at weight 0.5, production tail of 70 unseen shots.
The frozen original ensemble reproduced all native scorer fields, S **0.993385**. Averaging the
decoded projections of the same two fitted members instead scored **0.993386**; replacing the
projected MLP with itself reproduced every field of that decoded control exactly. The +0.000001
S offset is numerical arithmetic order, not a fitted-model improvement. Verdict: replacement
effects must use the matched decoded control. Full scorer fields, hashes and filenames:
[results/geometry_identity_20260922.json](results/geometry_identity_20260922.json).

**2026-09-22 — fixed-step refit pipeline validation:**
`python -m my_experiments.run execute runs/refit-smoke-da16cac19c60b796`, CPU, 4 shot workers,
full-frame 70-shot train / zero-shot validation / 14-shot score. Both MLP and sequence ran for
exactly five steps after their functional-head prefit, kept their last weights, and scored through
the native scorer. Artifact records `fit_mode=fixed_steps`, 70 train filenames, empty validation
filenames and zero overlap with the 14 scored filenames. Separately, 32 ordinary validation-mode
prediction, state and step arrays for both models matched the frozen pre-change code exactly in
focused CPU tests. Verdict: the no-validation path works end to end; the tiny-fit scores are only
correctness checks. Evidence: [results/refit_smoke_20260922.json](results/refit_smoke_20260922.json),
[results/refit_validation_compat_20260922.json](results/refit_validation_compat_20260922.json).

**2026-09-22 — exact auxiliary-cache reuse for refit:**
`python -m my_experiments.refit_cache` composed the verified 5633-shot train and 1338-shot
validation target blocks into the ordered 6971-shot, 1,544,510-frame refit cache in **4.50 s**.
The helper checked producer and dependency identities, all full-shot lengths, target bits
including NaNs, and zero overlap with the 70 scored shots before atomic publication. SHA-256 of
the ready cache is `1de2785f5f7d8de3ce89086ccd6362e8600db581e36847be34edff1813fc1df4`.
Verdict: reuse removes the cold auxiliary extraction step from this exact refit without changing
the targets. Evidence: [composition](results/refit_cache_composition_20260922.json),
[independent verification](results/refit_cache_verification_20260922.json).

**2026-09-22 — compact production control and iteration cost:**
`make experiment SPEC=experiments/compact-production.json`, production shares
`0.80/1.0 0.19/1.0 0.01`, salt 0, seed 0 for `mlp` and `seq`, frozen iteration configuration
restricted to these two models, RTX 3090 and 24 reader/scorer workers. On the same 70 unseen
shots: MLP **0.992149**, sequence **0.990514**, equal-weight ensemble **0.993385**;
the blend adds **0.001236** over MLP. Verdict: a measured compact working control; sequence earns
its ensemble place on this split despite a lower individual score. No uncertainty claim or
comparison to the historical leaderboard fold is licensed. The child took **3511.48 s** cold:
approximately 18 minutes building auxiliary targets, 21 minutes reading/caching frames,
104.8 s further preprocessing, **151.1 s MLP fit**, **817.4 s sequence fit**, and **84.26 s scoring**.
MLP kept step 33000 after 51400 steps; sequence kept 30000 after 40000. A separate warm read of
the same full-frame train/validation windows took **25.52 / 3.98 s**, with auxiliary cache check
0.03 s. Combining those timings gives **19.8 minutes estimated per warm iteration** at the same
fit durations; no second full fit was run. Campaign `runs/compact-production-9301af36275620f7`,
completion hashes verified. Native scores, all scored filenames and provenance:
[results/compact_production_20260922.json](results/compact_production_20260922.json).

**2026-09-22 — iteration workflow validation:** `make test`, smoke protocol
`0.01/0.2 0.01/0.2 0.002`, salt 0, frozen iteration recipe, RTX 3090: S = 0.721159
on 14 held-out shots, identical in the cold and warm runs. Child process wall time was 144.46 s
cold and 91.03 s warm; verified resume took 1.30 s. `run compare` with the same artifact as
baseline and candidate returned zero replacement/addition deltas on those same 14 shots.
**Pipeline validation only, not a production-quality result or a comparison with 0.9941.**
Provenance, native metrics and final checks: [results/workflow_20260922.json](results/workflow_20260922.json).

**This file is Challenge 1 only.** MAST's record is [experiments_mast.md](experiments_mast.md)
and its open ideas are [ideas_mast.md](ideas_mast.md).

One line per experiment: what was tried, the number it produced, the verdict. Untested ideas live
in [ideas.md](ideas.md) and move here once there is a number attached — whichever way it went.

Gains are **paired** differences of S within one split salt; "quality" is
`0.05/0.2 0.05/0.2 0.01`, "prod" is `0.45/0.1 0.15/0.1 0.01`. What a run can resolve is measured,
not assumed — see "The noise, measured properly" below: at production a single paired run resolves
**0.0013**, and three seeds on one salt resolve 0.0008.

| date | tested | result | verdict |
|---|---|---|---|
| 08-11 | Thomson explains the failing shots | R² correlates −0.07 with Te, 0.00 with ne, 200 shots | refuted |
| 08-11 | the error is a per-shot flux offset | removing the constant recovers 21% of it | refuted |
| 08-11 | normalize polarity by sign of Ip | 0.68 → 0.63 | rolled back |
| 08-12 | standardize each PCA component | 0.7466 → 0.7155; whitens the loss | refuted |
| 08-12 | ridge as an ensemble member | 0.7466 → 0.5936; dilutes, not diversifies | rolled back |
| 08-12 | early stopping for CatBoost | 0.7435 → 0.7439 across every budget tried | no effect |
| 08-12 | MLP patience | 0.5521 at 20 vs 0.7273 at 100 — the patience IS the hyper-parameter | kept at 100 |
| 08-12 | more shots, fewer frames per shot | 0.8111 → **0.9319** at a fixed row budget | kept |
| 08-12 | `inputs: coil_pca` instead of raw currents | +0.0102, −0.0042, +0.0052 over 3 seeds | no effect; kept as an option for MAST |
| 08-12 | subtract the analytic coil field | quality **+0.0088** (3 seeds) but prod **+0.0007** | see "reversals" |
| 08-13 | `field` loss, hand-set λ | +0.0048 at λ=0.01, +0.0059 at λ=0.1, −0.0054 at 0.3 — a flat plateau | removed; it showed the effect exists and could not locate it |
| 08-13 | `jacobian` loss metric | quality **+0.0117** (3 salts, t=4.71), prod **+0.0097** at n_pca 50 | kept, default — the one change that pays at production |
| 08-13 | refit `jacobian_delta` on the prod model | +0.0029, one run | not measured; direction agrees with theory |
| 08-13 | boundary term in the metric | +0.0021 over 3 salts, and `1−D_LCFS` did not move (1 sign of 3) | refuted |
| 08-13 | `n_pca` above 50 | prod: 75 gives −0.0010, 100 gives 0.0000; the ceiling only rises +0.0003 of S anyway | saturated, do not raise |
| 08-13 | `n_pca` below 50 | quality: 30 beats 50 by **+0.0043** (3 salts, t=4.84). prod: 50 beats 30 by 0.0014, 20 by 0.0055 | see "reversals" |
| 08-13 | training shots 1408 → 3168 | 0.9769 → **0.9809**, every term up | kept |
| 08-13 | is the Jacobian's linearisation valid at the model's real error? | over/under-estimate and correlation per scalar, 318 held-out frames — see below | measured; the metric is miscalibrated per scalar |
| 08-13 | per-scalar calibration from random full-vector probes | ratios 1.08–1.28 for all seven, against li 1.65 / tri_bot 1.51 / R_axis 0.98 from the model's real error | refuted as built — the ratio is a property of the error DIRECTION, not of the functionals |
| 08-13 | **the noise, on 3 salts x 3 seeds at production** | seed sigma **0.0009**, salt spread 0.0018 — see below | measured; the 0.0060 we had been using was five times too big |
| 08-13 | are frames where extraction fails a material share of the loss? | `lcfs_fail_frac` and `cons_fail_each` are both **0.0%** at production | refuted, for free — there is nothing there to repair |
| 08-13 | where the 9 minutes go | MLP fit 352 s of 467 s training; reading shots 82 s; PCA 10 s; Jacobian 19 s | measured — the fit dominates, not the I/O |
| 08-13 | what holds the memory | peak 21.6 GiB against 1.5 GiB of data; ~19 GiB is idle pool workers | measured; fixed by `parallel.release()` and the shot cache |
| 08-13 | is the DataLoader worth its overhead? | 0.269 s/epoch against 0.130 s for the same batches by indexing | measured — 52% of every epoch is DataLoader bookkeeping |
| 08-13 | decoded-shot cache + pool release + sampler-driven batches | prod run **8:45 → 4:54**, peak **21.6 → 8.5 GiB**, S unchanged at 0.9809 and the same best epoch | kept — pure speed, verified bit-identical |
| 08-13 | **it-1** temporal smoothing of the predictions (A1) | the prediction is **half as rough** as the truth: 0.021 against 0.039. Smoothing raises coefficient MSE at every window past 60 ms | refuted, no production run spent |
| 08-13 | **it-2** smoothing only the tail components, in time not frames | best cell −2.7% of the error in the loss metric, which is ΔS ≈ +0.0004 against a resolution of 0.0013 | refuted — below what any run could read |
| 08-13 | **it-3** post-hoc de-biasing of the seven scalars (A7) | biases are ≤ 0.22 of the RMS miss, slopes 0.92–1.01; a perfect affine correction buys **+0.0004**, and that is fitted in-sample | refuted |
| 08-13 | **it-4** are reversed-Ip shots harder for the MLP? (A12) | metric R² 0.9898 on 5 reversed shots against 0.9208 on 65 normal ones, t = −3.85 | refuted with the **opposite** sign — reversed shots are easier |
| 08-13 | **it-5** where the residual actually is | **one shot of 70 carries 23% of the squared error**, two carry 33%, ten carry 54% | measured — the tail, not the average shot, is the target |
| 08-13 | it-5b are the failing shots out of distribution? | \|z\| max 2.8 and 1.8 against a fold median of 2.3; the most extrapolated shots are not the worst. Error spread evenly over frames | refuted — ordinary inputs, uniformly wrong |
| 08-13 | it-5c how much of the error is a per-shot constant? | **45.9% of the fold**, 85.4% of the worst shot; removing it takes the fold's metric R² 0.9594 → 0.9780 | measured — the dominant failure is a shot-level bias |
| 08-13 | it-5d is that constant a flux gauge? | **98.3% is a shaped map**, 1.7% a uniform offset | refuted as a gauge — it is a real field, and one that lasts the whole shot |
| 08-13 | **it-6** is that per-shot constant predictable from the current history? | 252 history features, ridge fitted on 400 validation shots: **0.008** explained on the held-out fold against 0.003 for predicting the mean | refuted — the history carries no signal about it |
| 08-13 | **it-7a** the `li` probe ratio printed `nan` at production | one full-vector probe in a thousand leaves `li` undefined, and one nan poisoned the sum over 300 frames | fixed — finite probes counted, the drop rate printed |
| 08-13 | **it-7** `calibrate_scalars: true` at production, 3 salts | +0.0007 / −0.0003 / −0.0006, mean **−0.0001**, 1 sign of 3 | refuted — it moves budget between terms, not into the score |
| 08-13 | **it-8** MLP 256x256 → **512x512**, 3 salts | +0.0032 / +0.0021 / +0.0013, mean **+0.0022**, 3 signs of 3, every term up and Consistency +0.0131 on salt 0 | **KEPT** — the first accepted change of the loop |
| 08-13 | **it-9** derivative features of `ECOILA` and `Ip`, salt 0 | `raw` −0.0011, `interp` −0.0001, `both` +0.0007 against 0.9841 | only `both` positive, and below what picking the best of three arms produces by chance |
| 08-13 | **it-9** `both`, confirmed on 3 salts | +0.0007 / +0.0003 / +0.0011, mean **+0.0007**, 3 signs of 3 | **not measured** — 0.93σ, below the pre-declared +0.0015; reverted |
| 08-13 | **it-10** Thomson scattering as 11 feature columns, 3 salts | +0.0027 / +0.0033 / +0.0056, mean **+0.0039**, 3 signs of 3; R²qb +0.0064 and Consistency +0.0086 on salt 0, and ridge 0.7022 → 0.7562 | **KEPT** — the largest single gain measured in this fork |
| 08-13 | **it-11** MLP 512x512 → **1024x1024**, salt 0 | 0.9866 against 0.9868, and the fit costs 706 s instead of 230 | refuted — the capacity line is exhausted, the constraint is now the data |
| 08-13 | **it-12** which Thomson groups matter, salt 0 | `te` 0.9840, `te,p` 0.9861, `te,ne,p` **0.9874**, `te,ne,p,shape` 0.9874 | measured — temperature alone is worth nothing, pressure carries it, density adds +0.0013, `shape` adds exactly zero |
| 08-13 | **it-13** `te,ne,p` against the it-10 set, 3 salts | +0.0006 / −0.0025 / +0.0022, mean **+0.0001** | not measured — the two sets are indistinguishable; kept the group-expressible one on grounds other than score |
| 08-13 | **it-14** drop the four columns permutation importance called dead | 0.9862 against 0.9874, **−0.0012** | reverted — see "importance is not ablation" below |
| 08-13 | permutation importance over every input column | pressure integrals top the Thomson block at 253%/244%; the four position columns 0.1–2.3%; `bcoil` 91% | measured — see "importance is not ablation" |
| 08-14 | **it-15** ensemble of 4 MLP seeds, 3 salts | +0.0034 / +0.0041 / +0.0050 over the seed mean, mean **+0.0042**, 3 signs of 3 | **KEPT** — the largest gain of the loop |
| 08-14 | seed sigma, re-measured on the current recipe | 0.00085 / 0.00132 / 0.00155 by salt, pooled **0.0013** | the morning's 0.0009 landed on the quietest salt; thresholds updated |
| 08-14 | **it-16** MLP patience 100 → 300, salt 0 | ensemble 0.9904 against 0.9902; the one undertrained seed gained +0.0012 (best epoch 121 → 645), the other three did not move at all | not measured — the ensemble already absorbs a weak member, at nearly double the fit time; reverted |
| 08-14 | **it-17** training shots 3168 → **4225** (0.45 → 0.60), 3 salts | +0.0012 / +0.0029 / +0.0013, mean **+0.0018**, 3 signs of 3; the seed spread also narrowed from 0.0018 to 0.0010 | **KEPT** — and it is what iteration 11 predicted: the constraint was the data |
| 08-14 | **it-18** the whole accepted stack, on salts **3 and 4** which no iteration ever saw | +0.0077 and +0.0057 against +0.0115 on the three salts used for selection | measured — about 40% of the claimed gain was selection bias; the real gain is **+0.0067** |
| 08-14 | **it-19** CatBoost at production, as a fifth ensemble member | ensemble 0.9916 against 0.9914; CatBoost alone 0.9838 against 0.9875–0.9885 for the MLPs, and it took **3829 s against 296 s**, hitting the iteration ceiling undertrained | refuted on both counts — the gain is noise and the cost would have made the iteration loop impossible |
| 08-14 | **it-20** raw Thomson PRESSURE profile (16 points per system) instead of / beside the summaries | summaries 0.9914, summaries+raw 0.9901, raw alone 0.9900 | refuted — raw alone equals raw+summaries, so the summaries add nothing to it, and yet three numbers per system BEAT sixteen points by 0.0014 |
| 08-14 | **submitted** the four-change stack | leaderboard **0.9896**, 4th place, against 0.9764 for the previous submission — **+0.0132** where the local salt-0 gain was +0.0145 | 91% of the local gain transferred; the fresh-fold estimate of 58% was too pessimistic |
| 08-13 | is the EFIT frame step really 20 ms? | 98.0% of intervals are, but **31% of shots carry a gap over 100 ms**, up to 900 ms | measured — any derivative or time window must use the real step |
| 08-13 | `pca_frame_share: 0.2` | PCA fit 9.1 s → 4.6 s of a 295 s run, memory 7.79 → 7.81 GiB, S 0.9809 → 0.9816 (+0.0007, sd 0.0013) | not measured on score, negligible on cost — knob kept at 1.0 for when shots grow |
| 08-14 | where the MLP's epoch actually goes, on a V100 | 44→512→512→52 at batch 512 is ~50 MFLOP against 15.7 TFLOP/s; four seeds batched cost 0.186 s/epoch against 0.200 s for one | measured — the loop is bound by kernel LAUNCHES, so the GPU was idle and the fourth seed is nearly free |
| 08-14 | does batching the seeds compute the same thing? | forward 6.9e-7, every gradient ≤ 8.5e-7 against four separate nets at initialisation | measured — float32 round-off and no more; see "Bit-identity is the wrong test" below |
| 08-14 | the four ensemble seeds as ONE vectorized model, on CUDA | quality S **0.9639 → 0.9639** with ridge identical to four decimals, fit **803.5 s → 61.3 s** (13.1x); on CPU it still pays 1.36x at production shapes | **built, measured, then removed** — see below |
| 08-14 | `device: cuda` for the MLP | paired on `make quality` (0.60/0.1 0.15/0.1 0.01, one net, salt 0, seed 0): S **0.9873 GPU against 0.9878 CPU**, ΔS −0.0005 = **0.4σ**, and `ridge` identical to four decimals. Fit **1037.6 s → 189.0 s** (5.5x), whole run 19:42 → 5:33 | **kept** — same score within noise, and the deterministic control says nothing else moved |
| 08-14 | do the two devices stop at the same epoch? | GPU 626 epochs keeping 525, CPU 704 keeping 603, same data and seed | measured — they do NOT, and should not be expected to: Adam turns float round-off into a different realisation of the same fit within a few dozen steps |
| 08-14 | is README's production table still the configuration it describes? | its `ridge` row says **0.7022**, which it-10 records as the PRE-Thomson value (Thomson took ridge to 0.7562); measured now, ridge is **0.7615** on the same command | the table was stale — ridge is deterministic, so it cannot disagree by chance, and the row had never been re-measured after a change it-10 records as kept |
| 08-14 | does batch size move the epoch time? | 183 batches 0.307 s, 91 → 0.152, 45 → 0.080, 22 → 0.033, 11 → 0.020 — time tracks the NUMBER of batches, not their size | measured; a candidate in ideas.md, NOT a free speed-up — it changes the optimisation |
| 08-14 | **grid E** more data: 0.60/0.1 → 0.80/0.1 → 0.80/0.2 → 0.80/1.0, with patience and the epoch ceiling rescaled per arm to hold the step budget at 18400 | against a measured `fast` control of 0.9885 on salt 0: **+0.0000 / +0.0020 / +0.0025**. The frames axis flattens hard — 0.1→0.2 is +0.0020 and 0.2→1.0 is only +0.0005 for 2.3x the wall clock and 51.9 GiB of RSS | **0.80/0.2 is the knee.** Five times more frames beyond it buys nothing measurable |
| 08-14 | **grid F, the confirmation**: 0.80/0.2 against 0.60/0.1 on salts 3 and 4 | **+0.0048 and +0.0025, mean +0.0037**, signs agree, for one extra minute of wall clock | **ACCEPTED** — and note the gain is LARGER on the salts nothing selected against (+0.0037) than on the selection salt (+0.0020), which is the opposite of selection bias. The largest confirmed gain in this fork |
| 08-14 | **grid G**: is the frames axis really flat past 0.2? 0.80/1.0 on salts 3 and 4 | the 0.2→1.0 increment is **+0.0005 / −0.0002 / +0.0025** on salts 0 / 3 / 4 — mean **+0.0009**, signs disagree, 1.2 sigma on the standard error of three | **not accepted, and not refuted either.** The effect is below what three runs resolve; settling it at 2 sigma needs ~8 salts, about 90 min. Left at 0.2 because the possible +0.0009 costs 2.2x the wall clock and 2x the memory, where the step TO 0.2 bought +0.0037 for one extra minute |
| 08-14 | did the frames-beat-shots prediction hold? | predicted shots would buy more, from the 08-12 result at a fixed ROW budget. Measured: shots 0.60→0.80 gave +0.0000, frames 0.1→0.2 gave +0.0020 | **prediction wrong, and recorded as wrong.** The 08-12 measurement traded frames AGAINST shots at a fixed budget; this one adds both, which is a different question |
| 08-14 | **grid D, the confirmation**: the two quality candidates re-measured on salts 3 and 4, which nothing selected against | `cos600` +0.0013 on salt 0 → **+0.0001 / −0.0014**, mean −0.0007. `deep3` +0.0012 → **−0.0014 / +0.0008**, mean −0.0003. Both have OPPOSITE signs on the two salts | **both refuted.** Not "fell short" — they went negative. Opposite signs across folds is the signature of noise, and the pre-registered arithmetic said this would happen: the observed best (+0.0013) was already below what selection alone produces over thirty arms (+0.0029) |
| 08-14 | **grid D**: batch 4096 with a sqrt-scaled rate AND gelu, composed | S **+0.0007 / −0.0005**, mean **+0.0001**; fit **2.29x and 2.31x** faster on the two salts | **kept** — parity to within a tenth of a sigma, and the two accelerators compose as their mechanisms predict: 1.89x from the batch, 1.61x from gelu's faster convergence, 2.30x together |
| 08-14 | **grid B** batch size, with patience AND the epoch ceiling rescaled so every arm sees the same 18400 steps of leash and 368000 of budget | 256 → +0.0009, 512 → baseline, 1024 → −0.0011, 2048 → +0.0001, **4096 → +0.0002 at 1.89x the speed**. All five inside the 0.0013 sigma | **score is insensitive to batch size over 256-4096**, and the time falls almost twofold. The speed result of the day |
| 08-14 | sqrt vs linear rate scaling with the batch | batch 4096 at 2.8e-3 (sqrt) gives +0.0002; the same batch at 8e-3 (linear) gives **−0.0047** | sqrt is right and linear is wrong here, by four sigma with an unambiguous sign |
| 08-14 | **grid C** weight_decay 1e-5 / 1e-4 / 1e-3 | +0.0009, −0.0009, **−0.0123**; monotone in strength | refuted — only the near-off setting is non-negative |
| 08-14 | **grid C** dropout 0.1 / 0.2, at patience 100 and 300 | −0.0015, −0.0050, −0.0008, −0.0061; monotone in strength at both patiences | refuted, and the longer patience did not rescue it |
| 08-14 | **the pre-registered prediction, tested**: do dropout and weight decay close the gap without moving validation? | train/val gap goes 3.39x (baseline) → 1.90x (BN) → 1.19x (wd 1e-3) → **0.56x** (dropout 0.2), and validation goes 0.0650 → 0.0662 → 0.1511 → 0.0965 | **confirmed, and worse than predicted.** Three independent regularisers, one answer, monotone in strength: the model is limited by BIAS, not variance, and the 3.4x gap is not overfitting worth removing |
| 08-14 | **grid C** gelu / silu instead of relu — is the dying-ReLU story real? | **+0.0000 and +0.0002**, the agreeing sign that was declared the criterion in advance | refuted. Both smooth activations land on exactly zero — but they converge in 392 and 425 epochs against 626, which is **1.6x of free speed** |
| 08-14 | **grid C** tanh in the hidden layers | −0.0031 | refuted, as written down before the run: it saturates on BOTH sides, so it cures dying units less well than relu, not better |
| 08-14 | **grid C** a third hidden layer, [512, 512, 512] | **+0.0012** — the best thing in grid C and second-best of all thirty runs | at the threshold, not past it. Note it-11 refuted extra WIDTH; depth behaves differently, which is what the bias-limited diagnosis predicts |
| 08-14 | **what thirty screened configurations are worth, honestly** | best result over grids A+B+C is **+0.0013**, against an expected maximum from selection alone of σ·sqrt(2 ln n) = **+0.0029** over the ~12 plausibly-neutral arms and +0.0034 over all thirty | **nothing here has been shown to help.** The observed best is SMALLER than pure selection would be expected to produce, which is the strongest available evidence that these knobs do nothing |
| 08-14 | **grid A** batch norm x learning rate, jointly, on `make quality` (one net, salt 0) | BN peaks at lr 3e-3 with **0.9873 against the baseline's 0.9873**, and falls away either side (1e-3 0.9866, 1e-2 0.9861, 3e-2 0.9816); it costs 30-38% more per epoch | **refuted at its own best rate** — the strong form. Measuring it at the baseline's rate alone would have called it harmful, which is a different and wrong claim |
| 08-14 | is the learning rate itself already right? | 3e-4 gives 0.9858, **1e-3 gives 0.9873**, 3e-3 gives 0.9830 — unimodal, peak on the value the fork has always used | measured; the rate was never tuned and is nonetheless optimal. Axis closed |
| 08-14 | does BN buy learning-rate robustness? | lr 3e-3 costs −0.0043 without BN and 0.0000 with it | measured — BN delivers exactly what it promises, and here that is worth nothing, because 1e-3 was already the optimum |
| 08-14 | **cosine** LR decay to 0.01x over the epoch ceiling | **0.9882 (+0.0009)** and the fit drops 191.2 s → **161.1 s**, 626 → 533 epochs | +0.0009 is BELOW the pre-registered 0.0013, so the score is not confirmed. Carried to the confirmation phase on the SPEED argument instead, which is a different criterion and is labelled as one |
| 08-14 | cosine on top of batch norm | 0.9879 against 0.9882 for cosine alone, at 228.3 s against 161.1 | BN adds nothing over the schedule either, and costs 42% more time |
| 08-14 | did anything else drift across the nine grid-A runs? | `ridge` is **0.7615 in all nine**, to four decimals | the deterministic control held — between configurations, only what was set changed |
| 08-14 | **A16**, the training loop recounted in STEPS: `max_steps` / `patience_steps` / `eval_every_steps` / `lr_t_max_steps`, no epoch boundary left in it | acceptance on two salts against the recorded controls: salt 0 **0.9906 against 0.9905**, salt 3 **0.9886 against 0.9890** — +0.0001 and −0.0004, both a third of a sigma | **accepted.** Not a result, a unit change: the shares→patience table in params.yaml is deleted and a grid can now move the data without silently moving the leash with it |
| 08-14 | **grid H**, the frames axis settled properly: 0.80/1.0 against 0.80/0.2 on EIGHT salts, with the step budget identical by construction and the PCA's sample held fixed | **+0.0001 / +0.0012 / +0.0032 / +0.0024 / +0.0009 / +0.0017 / +0.0024 / +0.0022** — mean **+0.0018**, every sign positive, standard error 0.0004, **t = +5.02**. Costs 7.6 min against 4.7 and 178 s of fit against 102 | **ACCEPTED, and it overturns grid G.** G read the same axis as flat at 1.2 sigma — but G gave the 1.0 arm `pca_frame_share: 0.1` and the 0.2 arm 1.0, so its PCA was estimated from half the sample on one side. Holding that fixed turns "unresolved" into five sigma. A confound in the CONTROL, not in the treatment |
| 08-15 | **submitted** the stack: every frame, gradient clipping, poloidal derivatives, four seeds and the sequence model | leaderboard **0.9932, first place**, against 0.9896 and 4th for the previous submission. Local salt 0 was 0.9945 | **+0.0036 on the board against +0.0031 locally — 116% transferred**, where the previous submission managed 91%, and the local-to-board gap narrowed 0.0018 → 0.0013. Over 100% is noise around 100%; the result is that a day of heavy selection cost nothing, which is what confirming on unseen salts is for |
| 08-15 | **the pair confirmed on salts nothing selected against** — mlp+seq at 0.80/1.0, gaps off | salt 3 **+0.0002**, salt 4 **+0.0012** against +0.0011 on the selection salt; both signs positive, mean of the unseen **+0.0007** | **accepted.** Smaller on the unseen folds than on the selected one, which is what selection bias looks like and is why the rule exists — but positive on both, so it survives its own test |
| 08-15 | how salt-sensitive the sequence model is, on its own | alone it scores 0.9922 / 0.9874 / 0.9900 on salts 0 / 3 / 4 — a spread of **0.0048** where the MLP's is 0.0016 | it trains on 5633 SEQUENCES, not 1.25M rows, so which shots it draws matters three times as much. That is the cost of the unit being the shot, and it is why its contribution to the pair swings from +0.0002 to +0.0012 |
| 08-15 | **splitting the output vector between models** — the map from one, q95 and betaN from another | `psi=mlp+seq,qb=mlp` gives **+0.0010 / +0.0005 / +0.0010** against the single MLP, mean of the unseen **+0.0008**. The REVERSE, `psi=mlp,qb=mlp+seq`, gives +0.0001 / −0.0003 / +0.0002 — **flat** | the control did its job: the sequence model's whole contribution is in the FLUX MAP and none of it is in the two directly regressed scalars. So taking the scalars from the MLP is mechanism, not tuning — and it costs a rescore, since both members are already fitted |
| 08-15 | is the arithmetic on the printed tables trustworthy? | the composite is linear in its four terms and each term reads one block, so a split is predictable from two tables. Rescored exactly on salt 4: **0.9927 / 0.9925 / 0.9917 / 0.9915** against the predicted same four | it is — checked rather than assumed, because the rounding at four decimals is the same size as the effect being read |
| 08-15 | **every combination of three families**, from ONE fit — mlp, catboost, seq and their averages at 0.80/1.0, salt 0 | **mlp+seq 0.9934** and mlp+catboost+seq 0.9934; mlp+catboost 0.9922, catboost+seq 0.9920; alone: mlp 0.9919, seq 0.9914, catboost 0.9869 | **the pair is the answer, and CatBoost is not in it** — all three score exactly what two do. The model that LOSES alone adds the most: seq is 0.0005 below the MLP by itself and worth +0.0015 in the average, all of it through Consistency (0.9697 → 0.9762). Decorrelated errors are what an ensemble is for, and a structurally different model is where they come from |
| 08-15 | CatBoost on the GPU, the cost that got it dropped | **1053 s** against 3829 s on the CPU, and `best iteration 7999 of 8000` again — still undertrained at its ceiling | the cost argument is gone and the score argument stands: +0.0003 on top of the MLP, nothing on top of the pair. Its `iterations` comment still calls 8000 "far above anything the fit should need", which is false at this data scale |
| 08-15 | **`frame_gaps` costs the per-frame MLP** — measured for free, as the same configuration with and without | 0.9919 with the two gap columns against **0.9931** without, salt 0, everything else identical and the pipeline deterministic | **−0.0012, where the comment predicted zero.** 97.86% of the steps are 20 ms, so the column is nearly constant — and a nearly constant input is not free: it is two more directions for the first layer to fit noise in. The pair was measured WITH it, so the pair's own number is a lower bound |
| 08-15 | **the sequence model, trained to convergence** — 200000-step ceiling, patience ended it at 55500 with the best at 45500, 41 minutes | **S 0.9914 against the MLP's 0.9931** on the same salt, shares and features — and worse on EVERY term: R2_qb 0.9914 against 0.9954, Consistency 0.9706 against 0.9755, 1−D_LCFS 0.9872 against 0.9885. But its validation **loss is LOWER**, 0.033245 against 0.0335, and the shot moves its prediction by 31.6% | **refuted as a replacement for the MLP, and diagnosed rather than merely lost.** It optimises the objective it was given better and scores worse, which is the same gap A13 is about — and the mechanism is one this fork already measured: the prediction is HALF AS ROUGH as the truth (0.021 against 0.039), so smoothing was refuted in 08-13. A bidirectional recurrence smooths further, and the functionals that read the map's geometry charge for it |
| 08-15 | **the sequence model, first arm** — bidirectional GRU over the shot, residual on the per-frame MLP | 0.9911 against the MLP's 0.9931 on the same salt, shares and features. But **best step 40000 OF 40000**, still improving, and the correction branch had grown to **32.1%** of the per-frame prediction | **not a result, a budget.** The screen was stopped after one arm rather than spending 2.4 hours on five more numbers about the ceiling. The budget was calibrated to match the MLP in PASSES over the data, and that was the wrong analogy: a pass here is 176 steps against 305, and a gradient averaged over a whole sequence moves less per step |
| 08-15 | **grid K**, the derivatives confirmed on the two salts nothing selected against | salt 3 **+0.0023**, salt 4 **+0.0013**, mean **+0.0018** — against +0.0010 on the selection salt | **ACCEPTED, default.** Larger on the unseen salts than on the selected one, which is the opposite of selection bias, and the second time that has happened in this fork (the first was the frames axis) |
| 08-15 | **C2, the representation floor** — ground truth pushed through the artifact's own 50 components and scored (`evaluate.py --mode basis`), same 70 shots as production | ceiling **S = 0.9990** against the model's 0.9945. Per term: R2_psi **1.0000** vs 0.9998, 1−D_LCFS **0.9950** vs 0.9902, Consistency **0.9974** vs 0.9802 | **of the 0.0055 left, 0.0045 is reachable by fitting and 0.0010 is the basis.** The bottleneck is not the bottleneck: 50 PCA directions reproduce the geometry the scorer reads to within 0.0010 of S, where B9 was written on an assumed 0.0175 of headroom there |
| 08-17 | **Thomson isolated at last** — derivatives kept, Thomson removed, the cell of the 2x2 nobody had filled | **0.9917**, against 0.9940 with both blocks, 0.9937 with interp derivatives only, and 0.9907 with neither. So **Thomson alone is worth 0.0023** and the derivatives 0.0010. Ridge falls 0.7651 → 0.7139 | **needed, not ballast — and it corrects a standing assumption.** Thomson was kept because it "costs no training time" after its correlation with the failing shots measured ~0, and C3's 0.0604 was reliance rather than necessity; the necessity is 0.0023, above the 0.0013 a paired run resolves. Ridge losing 0.0512 says the information is linearly accessible and simply absent from the currents, which is what the physics says too: pressure is the second free function of Grad-Shafranov and magnetics cannot see it. It also improves the Challenge 2 arithmetic — the expensive 0.0023 of the transfer amputation is Thomson, and MAST HAS Thomson, on a different chord geometry that could be re-summarised |
| 08-17 | **`derivatives: interp` alone, after C3 said the raw block is unused** | ensemble **0.9937 against 0.9940** for `both`, i.e. **−0.0003** for dropping twenty columns | **permutation and refitting agree here, and the contrast with Thomson is the lesson.** C3 put the raw block at 0.0003 and removing it costs 0.0003 — for a block that is genuinely UNUSED, permutation is an accurate estimator. For Thomson, used but replaceable, it read 0.0604 against a refit cost of 0.0033, twenty times out. So the two measurements disagree exactly when the block has a substitute, which is also exactly when the answer matters |
| 08-17 | **go/no-go for a MAST prediction: are the probe features commensurable?** — the same 6x6 grid-fraction probes computed on a DIII-D shot and a MAST shot, from each machine's own shipped rectangles | probe values RMS **0.2 on DIII-D against 2.45 on MAST, a 12x scale gap**. Mean shape after per-frame normalisation correlates **−0.626** | **one obstacle is real and one is not.** The sign is not: `AXIS_SIGN` is −1 on DIII-D and +1 on MAST, `coil_field` folds it in, and the scorer is invariant to the global flux sign anyway — `choose_psi_sign` picks one bit for the fold. So the shape agreement is 0.63 in magnitude, which is broadly similar fields, as two tokamaks should give. The 12x scale gap IS real and is the same turn-count/description mismatch: MAST's 656 solenoid filaments carry what DIII-D's single one leaves to a fitted gain. A DIII-D-trained net fed MAST probes would be extrapolating in scale and interpolating badly in shape |
| 08-17 | **C3, permutation importance over every input block** — permute one block across frames, rescore, restore; 26 blocks, 35 held-out shots, nothing refitted | Blocks the coil add-back does NOT touch: **thomson 0.0604**, deriv:interp 0.0051, deriv:raw **0.0003**, gaps 0.0000, vessel 0.0000. Level blocks (feature AND coil reconstruction): **ECOILA 0.4432**, F6A 0.1040, F7A 0.0928, F7B 0.0860, F6B 0.0665, plasma_current 0.0594, down to **bcoil 0.0005** | **the method validates itself and then says two surprising things.** `bcoil` is toroidal and drives no poloidal flux, and it reads 0.0005 — a signal that physically cannot matter, measuring as not mattering. Then: Thomson is the most relied-on block after the coils, at 0.0604, where this fork has treated it as near-free ballast. And the RAW derivative block is worth 0.0003 against interp's 0.0051, though A10 accepted the two together and never split them. The solenoid at 0.4432 dwarfs everything, which makes its under-determined gain a real obstacle to transfer rather than a technicality |
| 08-17 | **permutation reliance is not necessity, and the two differ by 20x here** | Thomson permuted at inference costs **0.0604**; Thomson and the derivatives REMOVED and the model refitted without them costs **0.0033** (0.9940 → 0.9907, measured for the transfer floor) | **a fit leans on what it was given far harder than it needs to.** Permutation asks "what would break this model", refitting asks "what does the problem need", and the second is the one a feature decision should be made on. Worth stating because the first number is the one that is cheap and therefore tempting |
| 08-17 | **why the coil gains differ between the machines, and it is not units** — filament counts per current column on both | DIII-D ships **19 filaments over 19 columns**: one per coil, so the TURN COUNT is nowhere in the geometry. MAST ships **812 over 11**, with **656 on the solenoid alone** and 4–23 on each P-coil. Raw current magnitudes are comparable (F-coils median 56, ECOILA 19.6, ratio 2.9), so ECOILA's +110 gain is not a kA/A conversion | **the two machines describe their coils at different levels of detail, and the fitted gain is what covers the gap.** DIII-D's single filament per coil omits the turns, so the gain absorbs them — which is exactly why ECOILA reads +110 and why building probes on the raw Green's function cost 0.0149 there. On MAST the turns are IN the geometry, so unit gains would be approximately right. That makes the gain a per-machine correction for description depth rather than an arbitrary fudge, and it turns Challenge 2's blocker into a specific, statable one: DIII-D's solenoid contribution is under-determined (the README's own calibration reads +142, +128, +94 or −10 for it depending on the sample) while MAST's is fully specified |
| 08-17 | **probes built on the RAW Green's function instead of the gain-scaled maps** — the change that was supposed to make the form crossable, both input modes, no derivatives, no Thomson | **0.9732** and **0.9718**, against 0.9881 and 0.9861 with the gains. **−0.0149**, and it lands on the flux term: R2_psi **0.9804** against 0.9995 | **reverted, and the reasoning that produced it was wrong in an instructive way.** The gains do TWO jobs. Besides absorbing whatever makes the subtraction optimal — which genuinely has no true value to recover — they carry the per-column UNIT conversion: ECOILA ships in kA with its turn count not folded in, which is why its gain reads near +100 on its own. A probe is a linear combination ACROSS columns, so weighting them wrongly by two orders of magnitude gives a mixture that no per-feature standardisation afterwards can undo. **The unit half of the gains is physics, not nuisance** — and it is exactly the half MAST cannot supply, which makes this a real obstacle rather than a plumbing detail |
| 08-17 | **`inputs: flux_shape`, the scale removed as well** — the probe vector divided by its own RMS with the scale returned in logs, no derivatives, no Thomson | ensemble **0.9861**, against 0.9881 for the same probes with their scale, 0.9907 for raw currents at the same amputation, and 0.9940 for the full DIII-D set. Ridge **0.6301** against 0.6885 | **a further 0.0020, and the cost is the PARAMETERISATION rather than the information.** 36 probes become 35 shape numbers plus one scale, so nothing is thrown away — but the net now has to learn a product, `exp(log s) · p̂`, that it was previously handed. Ridge losing 0.058 over a change that is information-preserving for a nonlinear model is the clearest possible statement of that. Full price of a form that could cross: **0.0079** |
| 08-16 | **the price of a configuration that could actually be fed a MAST row** — probes with NO derivative and NO Thomson columns, against raw currents at the same amputation, production, salt 0 | probes **0.9881** (38 features), currents **0.9907** (21), full DIII-D control **0.9940** (84). Ridge 0.6885 against 0.7078 | **transferability costs 0.0059, and it splits in two.** Dropping the derivative and Thomson blocks costs **0.0033** — they are d/dt of DIII-D signals BY NAME and summaries of a chord geometry MAST does not share, so no model carrying them can be handed a MAST row whatever its input mode. The probe form then costs a further **0.0026**. Against a Challenge 2 score of exactly zero that is a trade worth making — IF the thing transfers, which this does not yet show |
| 08-16 | **`inputs: flux_probe`, the first feature form that means the same thing on two machines** — the coil flux at a 6x6 lattice of grid FRACTIONS instead of the 21 raw current columns, production, salt 0 | ensemble **0.9922 against 0.9940, −0.0018**, and `ridge` **0.7540 against 0.7651**. Individually 0.9898 / 0.9889 / 0.9902 / 0.9881 | **the form costs, and it is worth knowing the price before spending it.** 36 probes CAN in principle recover 19 coil-flux dimensions, but the recovery is ill-conditioned — neighbouring probes see nearly the same flux — and ridge dropping 0.0111 says the information really is lost rather than merely rearranged. Against a Challenge 2 score that is currently ZERO, −0.0018 on DIII-D is a trade worth making many times over. What it does NOT yet establish is transfer |
| 08-16 | **A9, the scored functionals as auxiliary outputs** — the net predicts kappa and li (or all seven) alongside the 52 real targets, discarded at inference. Production, salt 0, four seeds | ensemble **0.9941** narrow, **0.9942** broad, **0.9935** at four times the weight, against **0.9940**. The individual models go the OTHER way: narrow averages 0.9913 against the control's 0.9921 | **flat, and the sign flip between "alone" and "in the average" is the interesting part.** Teaching the net the functionals makes each seed slightly WORSE and the average of four slightly better — the same decorrelation the whole day has been about. As a fourth member of the confirmed feature-set ensemble it adds **+0.0000** (0.9950 either way), which is the saturation already measured. Heavier weight hurts monotonically, so the auxiliary block competes with the real targets rather than complementing them |
| 08-16 | **B5, the four remaining arms** — `seq_hidden` 512, `seq_layers` 2, `dropout` 0.1, against `lr` 3e-4 and the production control | seq alone: q2 (lr 3e-4) **0.9922**, control 0.9914, q4 (hidden 512) 0.9913, q5 (2 layers) 0.9908, q6 (dropout 0.1) 0.9896, q3 (lr 3e-3) 0.9844. Delta share 41.6 / 31.6 / 26.9 / 33.0 / **429.4** / 23.6% | **only the learning rate moves it, and only downward from 1e-3.** Capacity does nothing (512 wide and 2 layers both lose), and dropout is a disaster whose delta share of 429% says exactly why: the head and the correction branch end up fighting rather than composing. Five arms, one winner, worth +0.0001 in the ensemble — the screen is complete and it does not pay |
| 08-16 | **what a BETTER sequence model is worth in the ENSEMBLE** — q2's seq spliced into the production artifact in place of its own, everything else identical | **0.9946 against 0.9945.** The two artifacts were checked bit-identical first: PCA components, target scaler centre and scale, the loss metric `L`, the training file list and the coil gains all match, so the two seq models live in the same scaled space and the splice is legitimate rather than convenient | **+0.0008 alone becomes +0.0001 in the average, and that bounds the whole screen.** With the seeds controlled the recurrence is worth +0.0004 to the ensemble in total, so no tuning of it can pay more than that unless it changes what the member CONTRIBUTES rather than how good it is alone. The four MLPs already carry most of what a better seq would add |
| 08-16 | **B5 arm q3, the learning rate raised instead** — 1e-3 → 3e-3 | seq alone **0.9844**, and the shot moves the prediction by only **23.6%** against 31.6% at 1e-3 and 41.6% at 3e-4 | **the rate axis is monotone and the screen's first two arms bracket it**: 3e-4 gives 0.9922, 1e-3 gives 0.9914, 3e-3 gives 0.9844, and the delta share falls with it. A rate too high does not merely fit worse, it cuts the residual branch off — which is the mechanism the delta share was added to expose |
| 08-16 | **B5 arm q2, the learning rate the screen never got to try** — `models.seq.learning_rate` 1e-3 → 3e-4, everything else the production sequence model's, salt 0, 0.80/1.0 | **seq alone 0.9922 against production's 0.9914, +0.0008.** It runs longer for it — best step 91000 of 101000 against 71000 of 81000 — and the recurrence is used MORE: the shot moves the prediction by **41.6%** against 31.6% | **the first arm of a screen that was never run lands, which is the point of running it.** Every number this architecture has produced came from one untuned configuration while the MLP beside it was tuned over thirty runs. +0.0008 is inside the 0.0025 that selection over six arms alone would produce, so it decides nothing yet — but the delta share moving with it is mechanism, not noise |
| 08-16 | **B7, Grad-Shafranov cleanup at inference** — damp the predicted coefficients along the quadratic form that measures toroidal-current error OUTSIDE the vessel envelope, `c' = (I + lam Q)^-1 c`, swept, no retraining | **monotonically worse.** lambda 0 gives **0.9939**, 0.1 gives 0.9937, 1.0 gives 0.9910, with Consistency 0.9772 → 0.9762 → 0.9651. 1781 of 4225 grid points carry the penalty; `Q`'s condition number on the 50 components is 13.6 | **refuted, by its own clause — "refuted if the damping sweep's optimum is zero", and it is exactly zero.** The maps already satisfy `Delta* = 0` in the vacuum region as well as this basis permits, which C2 said in advance: R2_psi is 0.9998 of a 1.0000 ceiling. Pushing them further does not clean them, it moves them off the truth. The per-frame masked version was gated on this showing a sign, so it does not follow |
| 08-16 | **what the machine-independent input form costs** — `inputs: coil_pca`, the coil flux in its own principal directions instead of the 21 raw current columns, production, salt 0 | ensemble **0.9938 against 0.9940**, i.e. **−0.0002**, inside noise. Individually 0.9908 / 0.9918 / 0.9920 / 0.9913 against 0.9920 / 0.9916 / 0.9920 / 0.9927. `ridge` is **0.7651 in both**, to four decimals | **the coordinate change is free, and it is necessary but NOT sufficient for MAST.** Ridge reproducing exactly is the check that the two forms carry the same information — the coil flux is linear in the currents, so a linear model cannot tell them apart. What the run does not buy is transfer: `coil_pca_W` is a rotation of THIS machine's own current columns, defined by the flux they produce on THIS machine's grid, so MAST's 13 columns on a grid from R = 0.06 would give a different basis meaning different things. Closing that needs a description of the coil flux that means the same thing on both machines, which is a project and not a flag |
| 08-16 | **B11's free diagnostic — does the loss metric point somewhere ELSE on ramp-down?** `M` built separately on three regimes by position in the shot, 60 probe frames each, 25 held-out shots | **the directions differ, not only the scale.** Principal angles between the top-10 eigenspaces: median **10.8°** ramp-up vs flat-top, **16.9°** for both pairs involving ramp-down, with maxima of 58–89°. Relative Frobenius distance between the NORMALISED forms: **0.584 / 0.803 / 0.492** — 0 would be identical up to a scale. Traces 146.8 / 47.2 / 51.8, the 3x that C7 already measured | **the premise survives, and the entry now has to compete on cost rather than on plausibility.** Averaging `JᵀJ` over the shot blurs direction as well as magnitude, and ramp-up is the outlier on both. But C7 also showed sensitivity does not explain where the cost is, and A19 has since taken the ramp-down gain through FEATURES for eight columns and no rework — where this needs a per-row quadratic form moved out of `TargetScaler` and into the MLP loss |
| 08-19 | **the first MAST sweep that could see 30% of the composite** — with the scorer no longer grading MAST on DIII-D's grid, the source profile swept against the FULL S rather than R2_psi alone, predictions written to an npz and scored by `local_score.py --pred` so there stays exactly one implementation of the metric | **alpha = gamma = 1 and beta = 0.3 survive** — confirmed against the composite for the first time, with (0.5, 1.0), (2.0, 1.0), (0.5, 1.5) and beta 0.2/0.4 all worse. The one gain is a new knob, `edge`: letting the source run past the last closed surface, as a fraction of the axis-to-boundary span, is worth **S 0.5737 -> 0.5823** on a broad plateau — 0.5791 / 0.5823 / 0.5823 / 0.5795 at 0.01 / 0.02 / 0.03 / 0.04, against 0.5563 at -0.02. It raises R2_psi (0.9158 -> 0.9242) AND Consistency (0.0204 -> 0.0452), and costs a little D_LCFS | **it is aimed at a measured deficit, not fitted blind.** `diagnose_boundary.py` decomposed D_LCFS on the correct grid: raw 0.3317, with position removed 0.2401, with position and size removed 0.1126 — roughly a third each — and our contour is **0.929x** the true one, with the worst point at the outboard midplane on 81 of 115 frames, which is where a size error shows first. `edge` is the one number that moves size. It is also physical rather than a fudge: there is real current outside the separatrix. **Two harness bugs found by numbers, not by reading.** The knob was a no-op at first — psi_N already reaches 1 at the last closed surface, so the shape factor is zero however far `inside` extends, and S was identical to four decimals for edge 0.00 to 0.20 until psi_N was normalised to the moved surface. And two sweeps sharing one scratch filename raced: the second scored the first's last cell and reported R2_psi 0.8621 where two independent routes agree on 0.91579 |
| 08-19 | **shrinking the two affine calibrations toward the constant they degenerate into** — the leave-one-out study put R2_qb at 0.2771 in sample and 0.1663 held out, the only term with a real in-sample bias, so the obvious move is to keep less of the fit | **refuted, and the composite disagrees with the term it was aimed at.** Held-out R2_qb peaks at keep = 0.75 (0.1756 against 0.1663), but held-out S falls monotonically: 0.6166 / 0.6140 / 0.6077 / 0.5983 / 0.5853 at keep 1.00 / 0.75 / 0.50 / 0.25 / 0.00 | **the scorer floors `max(0, R2_qb)` before weighting it, and that is the whole explanation.** One of the three folds scores R2_qb negative and contributes zero to S however negative it is; shrinking lifts that fold's number without buying anything, while taking real value off the two folds that are positive. So a term-level improvement and a composite-level loss are both true at once. Keep the affines at full strength — they overfit, and they are already at the right strength anyway |
| 08-19 | **reverting `edge` after the board said the map tuning did not travel** — the obvious repair, since the arithmetic said the 0.5566 map plus the new scalars would give 0.5804 | **it does not separate.** At edge = 0, with the same wall and ensemble and everything refitted, local S 0.6359 -> 0.6123: the boundary term recovers as intended (D_LCFS 0.4125 -> 0.3848) and the scalars fall with it, R2_qb **0.3234 -> 0.2079**. Re-sweeping the q95 proxy surface at edge = 0 does not recover them — psi_N = 0.90 is still the best level and reaches only **0.3154** there against 0.3768 at edge = 0.03, because at edge = 0 the surface exists on **108 of 115** frames rather than all of them | **`edge` and the scalars are coupled, so the +0.0237 that transferred cannot be kept without the map change that cost 0.0126.** Every variant of this family prices out at 0.567-0.570 against the 0.56777 already on the board, so the repair is not worth a submission. What IS worth one is decomposition: two probes built on 08-18 and still unspent isolate the bundle exactly — `065915Z` against the submitted `091150Z` differs only by MAST's real wall, and `224648Z` against `065915Z` only by the three-member ensemble. Two uploads, no compute, and they say which of the three pieces to keep before anything else is built |
| 08-19 | **the wall + leave-one-out ensemble + edge=0.03 bundle, scored** — submitted after local S went 0.5941 -> 0.6359 | **board S 0.5566 -> 0.56777, +0.0111**, G_ratio 0.560 -> 0.571. **27% of the local gain transferred, and the composition is the finding**: R2_qb **+0.0237** (q95 0.0574 -> **0.3427**, betaN -> 0.0886) against R2_psi **-0.0049** and D_LCFS **-0.0077**. Local had said R2_psi +0.0202 | **the scalar calibration travels and the map tuning does not, and the sign flipped on the term that carries 0.55 of the weight.** q95's affine transferred at **91%** — local 0.377, board 0.343 — while every change aimed at the flux map lost on the fold what it gained on three shots. Note which half of the local metric was honest: D_LCFS got worse locally AND on the board (0.3406 -> 0.4125 against 0.4121 -> 0.4886), so the scorer was right about the boundary and wrong only about R2_psi. **This is the leave-one-out prediction coming true.** LOO measured no in-sample bias at all (held-out R2_psi 0.9328 against 0.9316) and the record written beside it said the cause of earlier reversals must therefore be that three consecutive shots of one campaign are a narrow sample of discharge TYPES, which leave-one-out on siblings cannot see. That is now confirmed rather than argued. The arithmetic points at the next probe on its own: keeping the 0.5566 entry's map and only its scalars would give **S 0.5804** |
| 08-19 | **MAST's first wall, taken from the public archive** — `efm/limiterr` / `efm/limiterz` at STFC Echo (CC BY 4.0), a closed 37-vertex polygon, replacing a single inboard radius guessed from where `efit_lcfs_r` bottomed out on the demo shots | the machine says the inboard wall is at **0.19524 m** against the guessed 0.1963 — 1.1 mm — and supplies the whole shape, so the limiter branch of the boundary search asks where the plasma touches the MACHINE rather than where it crosses one vertical line. Against the shipped `efit_psi_boundary` on the true maps the tails improve and the median does not: p90 **5.35% -> 4.69%**, max **8.52% -> 7.77%**, median 0.045% -> 0.077%. On the composite, calibrations unchanged, **S 0.6310 -> 0.6336**, every term up; refitting the three leave-one-out calibrations under the new geometry gives **0.6359** | **a free gain from a machine constant, and the reason it is takeable is worth stating precisely.** The archive's `efm/` group IS the competition's withheld MAST truth — verified bit for bit on demo shot 28348, `psirz` max |diff| **0.000e+00**, likewise li, betan, elongation, magnetic_axis_r and psi_axis — and the 1206-shot test fold is drawn from the same campaigns. The terms allow external public data WITH DISCLOSURE and separately forbid any attempt to de-anonymize or memorize the hidden truth. Training on the archive would be the second even unintentionally, and excluding the test fold first requires identifying it, which is also the second. What is taken instead is geometry: **byte-identical on shots sampled from all five campaigns M5-M9**, the same kind of object as the coil rectangles the competition ships on every row, and an answer to no shot. The organisers have been asked about the wider question. Precision footnote: the literal was first rounded to four decimals, worth 0.04 mm on a 30 mm grid and **0.0001 of S** — measured, then shipped exact anyway |
| 08-19 | **leave one demo shot out: the first honest held-out measurement MAST has ever had** — `mast/loo.py` refits the whole calibration (gains by alternation, the TF constant, both affines) on two of the three demo shots and scores the third through the real scorer, rotating | **the flux map does not overfit its three shots at all.** Held-out R2_psi **0.9328 +- 0.0020** against **0.9316** in sample — an in-sample bias of **-0.0012**, i.e. nothing, and the gains barely move between folds (P4 1.020 / 1.028 / 1.022). The composite's bias is **-0.011** and almost all of it is one term: R2_qb **0.2771 in sample against 0.1663 held out** | **the number that licenses everything built on those three shots, and it is the opposite of DIII-D's.** `solver/how_many_shots.py` asked the same question where shots are plentiful and got **-0.601 +- 0.120** for a three-shot calibration. The difference is in what each dataset ships: MAST gives 812 conductor rows, so the solenoid has a SHAPE the fit can separate from a constant, while DIII-D's single ECOILA rectangle cannot be told from an offset — the same degeneracy that makes DIII-D's per-frame flux level unrecoverable. So the three demo shots are a usable calibration set for the MAP and a poor one for the two value scalars, which is exactly where the bias landed. **It also retires a worry rather than confirming it**: three MAST decisions were refuted by the board after looking good locally, and this says the cause was not in-sample bias in the calibration — it was that the demo shots are a narrow sample of discharge TYPES, which leave-one-out on siblings cannot see |
| 08-19 | **the same level-versus-shape question asked of MAST** — the per-frame flux offset between our solve and the truth, on the 115 demo frames | **MAST's level is essentially right and DIII-D's is not.** MAST: offset **-0.00014 +- 0.00418 Wb/rad on a range of 0.261**, i.e. **1.6%** of the range, and removing it moves R2_psi only 0.93170 -> 0.94101; a per-frame scale as well gives 0.94664. DIII-D: **+-0.147 on a range of 1.4, 10.5%**, and removing it moves 0.207 -> 0.711 | **this is why one machine works and the other does not, and it is a difference in what the dataset ships rather than in the physics.** MAST ships 812 conductor rows, so the solenoid's field has a SHAPE the fit can separate from a constant; DIII-D ships one rectangle per coil and `coil_field.py` already recorded that ECOILA is not identifiable against an offset. Two consequences worth keeping. **MAST's remaining flux error is shape, not level** — a perfect per-frame gauge would be worth +0.009 of R2_psi, i.e. **+0.005 of S**, which agrees with the boundary decomposition charging the error to size and position rather than to a global shift. And the DIII-D repair is now specific: it needs a per-frame flux gauge, which no amount of coil-gain fitting supplies |
| 08-19 | **what DIII-D's error actually is: the flux LEVEL, not the shape** — the per-frame mean difference between our solve and the truth, on shots the coil gains never saw | **removing one constant per frame takes R2_psi from +0.207 to +0.711** over 2112 held-out frames, and a per-frame scale as well takes it to +0.905. The offset itself is +0.006 **+- 0.147 Wb/rad** on a flux range of 1.4 — mean zero, enormous scatter. It is not recoverable: regressed on the 19 coil currents plus Ip plus a constant it scores R2 0.95 in sample and **0.031 held out**, and applying the prediction moves the flux R2 only 0.386 -> 0.411 where the true offset would give 0.765. Both gain estimators leave it: two-stage -0.112, one-stage -0.246 on a disjoint set, with the same +-0.15 scatter | **the DIII-D blocker is now named, and it is not the one that was being chased.** Every pin, every gain refit and every profile sweep was working on the SHAPE, which is already worth 0.71 once the level is granted. What is missing is the absolute normalisation of psi, frame by frame — and the reason it is missing is structural: `coil_field.py` records that ECOILA's field over this grid is nearly degenerate with a constant offset, so its gain sets the level and one number cannot set a level that moves through a discharge as ohmic flux is consumed. For Challenge 1 this never mattered — whatever is subtracted is added back unchanged — and for a forward solve it is most of the error. **So DIII-D stays unusable as a proxy for MAST**, and the next piece of work on it is not another knob but the flux gauge itself. Recorded also because it explains the instability across draws: held-out R2_psi read -0.25, -0.11, +0.21 and +0.39 on four different eval sets, which is what a large unmodelled per-frame term looks like when it is averaged over different shots |
| 08-19 | **CORRECTION, and the sample-size study that produced it** — `solver/how_many_shots.py` fits the two-stage coil calibration on 1, 3, 6, 12 and 24 DIII-D shots, three independent draws each, and scores every one on the SAME 18 shots that no draw ever touches | **every DIII-D number reported earlier today was measured on shots the gains had been fitted on.** `d3dwhere.py` evaluated on 8 shots of which **8 of 8** were in the gain fit; `validate_d3d --fit 1 --eval 20 --seed 0` on 20 of which **11 of 20** were. On a genuinely disjoint set the same estimator scores **-6.58 +- 8.99 at one shot, -0.601 +- 0.120 at three, -0.253 +- 0.046 at six, -0.235 +- 0.063 at twelve and -0.148 +- 0.087 at twenty-four** — negative everywhere, against the +0.356 / +0.609 / +0.50 quoted before | **the paired comparisons survive and the levels do not.** Both arms of every contrast saw the same shots, so 'the search pin is worth a lot' and 'the centroid pin is worth more' and 'the free alternation loses to the two-stage fit' all stand as relative statements; what was wrong is every ABSOLUTE R2_psi and the words 'held out' beside it. **The conclusion gets stronger, not weaker: the DIII-D solve does not generalise at all yet**, so it is further from being a judge of MAST's choices than this morning's numbers suggested, and the 3-shot cross-validation the plan is built on has nothing to cross-validate. Two things the study says on its own. At **one** shot a single unlucky draw returns -19.3 and the spread is +-9.0, so the estimator is not merely biased at small n, it is unstable. And the curve is still climbing at 24, which is the honest answer to 'how many shots does a coil calibration need': **more than 24 on DIII-D**. That does not convict MAST's three, and the reason is identifiable: MAST ships 812 conductor rows and its well-posed coils come out at 1.010 / 0.961 / 0.994, so two numbers are really fitted there, while DIII-D's ECOILA is degenerate with a constant and drags eighteen F-coils with it. The evidence that MAST's calibration travels is the leaderboard itself — local 0.9158 against a fold 0.8895, a drop of 0.026 |
| 08-19 | **the DIII-D gain alternation, re-run now that the solver works there** **[levels corrected 08-19 — both arms were scored on shots the two-stage gains were fitted on; the ordering stands, the 0.609 does not]**  — 19 free coil gains alternated with the solve over 10 fit shots, against the two-stage physics fit (F-coils with a per-frame plasma amplitude and a per-frame constant projected out, ECOILA second with no constant) | **the alternation wins in sample and loses out of it**: R2_psi **0.698 on the fit shots and 0.304 held out**, against **0.609 held out** for the two-stage fit. Its gains are unphysical where the two-stage ones are not — F1A -0.24 beside F1B +1.57, against a two-stage median of 0.965 with a 10-90% range of 0.855-1.137 | **the same lesson as everything else this week, on a third axis.** Nineteen numbers fitted on ten shots chase the sample; the constrained estimator is worse where it is fitted and twice as good where it is not. Keep the physics gains. It also settles that DIII-D's remaining error is NOT the coil calibration: with the good gains the axis is now unbiased vertically (-0.012 +- 0.039 m) and the residual is radial, **-0.161 +- 0.260 m**, with the plasma area at 0.795 +- 0.468 of the truth — a scatter far larger than any gain refit moves |
| 08-19 | **pinning the CURRENT's centroid, not just the O-point search** **[levels corrected 08-19 — the DIII-D eval overlapped the gain fit entirely; 0.385 -> 0.609 is a valid PAIRED gain on one set, not a held-out level]**  — the DIII-D vertical pin bounded where the axis could be FOUND and left the source free to sit elsewhere; shifting the current distribution so its own centroid lands on the pin is the constraint a feedback system actually applies | **DIII-D: R2_psi 0.385 -> 0.609** over 1349 frames of 8 shots, with the vertical axis error collapsing from **-0.173 +- 0.055 m to -0.012 +- 0.039** — from pressed against the band's lower edge to unbiased. **MAST: 0.6296 -> 0.612**, worse. Its terms split hard: 1 - D_LCFS **0.6050 -> 0.7220**, the best boundary number this fork has measured, while q95 falls 0.3768 -> 0.1068 with the affines REFITTED for the pinned map, so that half is real and not a stale calibration | **the second DIII-D fix that does not transfer, and the pair is the point.** The search pin was worth +0.406 of R2_psi on DIII-D and 0.0000 on MAST; the centroid pin is worth +0.224 there and -0.018 here. Both say the same thing: **DIII-D's plasma is vertically unconstrained and MAST's is not**, because MAST's centre column anchors it — so forcing MAST's current to a fixed height overrides a solution that was already right. That is a real answer to the transfer question rather than a failure of it: the machines' dominant pathologies differ, and a knob has to be tested on the machine it is meant for. Recorded with the DIII-D numbers because the SOLVER is now much better there — 0.926 on MAST against 0.609 on DIII-D, up from 0.356 — which is the gate on using DIII-D to cross-validate anything |
| 08-19 | **A25 arm 1: three sequence seeds instead of one** — `seq1` and `seq2` added to params.yaml so ONE production fit produces both arms, scored on the same pass over the 70 held-out shots, paired by construction | **+0.0002 against a pre-registered gate of +0.0013.** `mlp+mlp1+mlp2+mlp3+seq+seq1+seq2` **0.9945** against `...+seq` **0.9943** and `mlp+mlp1+mlp2+mlp3` 0.9941. Consistency does move, 0.9790 -> 0.9803, and 1 - D_LCFS 0.9900 -> 0.9902, while R2_qb goes the other way 0.9970 -> 0.9969 | **refuted at its own gate, so arm 2 is not run.** The mechanism is real and the arithmetic is what kills it: three sequence seeds averaged score **0.9932** against 0.9909 / 0.9914 / 0.9916 for the individual fits, so the seq family carries +0.0016 to +0.0023 of genuine seed noise — but poured into an ensemble four MLPs already dominate, it is worth a fifth of that. The entry argued from the 08-15 table that one sequence model beats three extra MLP seeds (+0.0012 against +0.0009); on THIS run the same marginal step is +0.0002, which is the seed noise of that comparison rather than a contradiction — and it is exactly why a +0.0013 gate was set on a quantity a paired run resolves to 0.0013. `seq1` and `seq2` stay in params.yaml with `enabled: false`, so the definition survives beside its measurement and no run pays for them. With this, **every mechanism in ideas.md is closed** and Challenge 1 stands at its 0.0056 ceiling |
| 08-19 | **edge re-swept after the gains were refitted for it** — self-consistency, since the first sweep chose 0.02 with the old gains and the gains then moved | **the optimum moved to 0.03**, and reading it needs care: R2_qb jumps around (0.037 / 0.112 / 0.207 / 0.283 / 0.082 / 0.261) because the affines were fitted at 0.02 and q95 is sensitive to where the surface is. On the three terms that do NOT depend on that fit, `0.55 R2_psi + 0.10 (1 - D_LCFS) + 0.20 Cons` reads 0.5742 / 0.5804 / 0.5838 / **0.5859** / 0.5850 / 0.5789 at edge 0.00 / 0.01 / 0.02 / 0.03 / 0.04 / 0.06. Refitting the affines there gives local **S 0.6296**, R2_psi 0.9317, q95 +0.3768, betaN +0.2045 | **the self-consistency check was worth running and the lesson is about reading a sweep, not about the knob.** Taking S at face value would have picked 0.03 for the right reason by accident and 0.06 over 0.04 for the wrong one — the qb term carries an artefact of where its own calibration was fitted, and it is 0.15 of the composite. R2_psi rises to 0.04 and the two geometry terms fall throughout, so the optimum is a genuine trade and its plateau spans 0.02-0.04 within 0.002. Cumulative on the demo shots: **0.5941 -> 0.6296**, from a single number aimed at a defect two independent diagnostics agree on |
| 08-19 | **where on the grid R2_psi actually loses** — the squared flux error mapped cell by cell over the 115 demo frames, rather than summed into one number | **the outboard plasma edge, and nothing else.** By region: 31.1% inside the plasma (998 cells), **40.4% inside the vessel but outside the plasma (836 cells)**, 28.4% outside the envelope (2391 cells) — per cell that is 0.048% / 0.031% / 0.012%. The ten worst cells are all at **R = 1.21-1.33, Z ~ 0**, the outboard midplane edge. The inner columns, where the solenoid's 656 conductors crowd a grid that starts at R = 0.06 and passes 0.13 mm from a filament, carry about 1% each and are not the problem | **it kills a plausible hypothesis cheaply and corroborates the one that pays.** The suspicion was that the coil basis has a near-singularity where the solenoid sits on the grid and that the R2 sum is dominated by cells no plasma ever reaches; it is not. What the map shows instead is the same finding `diagnose_boundary.py` reached from the contour side — the boundary is too small and the worst point is at the outboard midplane on 81 of 115 frames — arriving here from the flux side. Two independent measurements of one defect, which is what makes `edge` a fix aimed at something real rather than a knob that happened to help |
| 08-19 | **an up/down coil asymmetry, aimed at a measured vertical bias** — one number scaling every upper coil gain by (1 + d) and every lower by (1 - d), because the gain fit ties each pair to one value and that symmetry cannot represent a vertical field error, while our magnetic axis sits **-0.0360 +/- 0.0256 m** against the truth, a bias of over half a grid cell | **worse in both directions and symmetrically so**: S 0.6148 at d = 0, 0.6006 at +0.02, 0.6019 at -0.02, 0.5529 at +0.05, 0.5739 at -0.05, and under 0.48 at +-0.10 | **refuted, and the symmetric gains are confirmed as right.** Worth keeping for the one thing the sweep shows that the composite hides: at d = -0.02 the boundary term jumps to **1 - D_LCFS 0.7062**, the best this fork has measured, from 0.6139 — so the tilt DOES move the plasma the way the position error asks for. It just pays for it elsewhere, with R2_qb collapsing 0.2070 -> 0.1163 and Consistency 0.0646 -> 0.0324. The vertical bias is real and this is not the handle for it |
| 08-19 | **the first attempt to check a MAST choice on DIII-D** **[levels corrected 08-19 — same overlap; the flatness of the edge sweep stands]**  — `edge`, swept over 2245 held-out DIII-D frames at the same gains, the protocol being that a knob chosen on three shots should reproduce on a machine with thousands | **DIII-D cannot tell.** Held-out R2_psi over edge -0.02 / 0.00 / 0.01 / 0.02 / 0.03 / 0.05 / 0.08: 0.35321 / 0.35631 / 0.35572 / 0.35462 / 0.35356 / 0.35702 / 0.35041 — a spread of 0.007 with no structure and the maximum at 0.05, where MAST's composite sweep put a clear plateau at 0.02 | **an uninformative result, and worth recording as one rather than as agreement or disagreement.** DIII-D's own solve sits at R2_psi 0.356 against MAST's 0.926, so its boundary error is dominated by whatever is costing it 0.6 of R2_psi and a 2% change in where the source switches off is below its own noise. The protocol is sound and the proxy is not ready: **DIII-D becomes a judge of MAST's choices only once its own solve is worth cross-validating**, and the vertical pin took it from -0.050 to +0.356, not to 0.9. So the `edge` knob stays a MAST-local claim decided by the board, which is why it shipped as two probes rather than one submission |
| 08-19 | **edge = 0.02, and the gain refit that comes with it** — calibrated and scored on the three demo shots through the corrected scorer | **local S 0.5941 -> 0.6148**. Term by term: R2_psi 0.9158 -> **0.9263**, R2_qb 0.1358 -> **0.2070** (q95 +0.2052 -> +0.2947, betaN +0.0664 -> +0.1192), Consistency 0.0203 -> **0.0647**, 1 - D_LCFS 0.6594 -> 0.6139. Holding the gains at the values the 0.5566 board entry used and changing only `edge` gives **0.6105**, so the knob is +0.0164 and the refit a further +0.0043 | **submitted as TWO entries on purpose.** The board is noiseless here — the solver is deterministic and the fold is fixed — so two zips separate the knob from the refit exactly, the way the 08-18 submission separated the Thomson source from the affine after the fact. The refit is the half to distrust: P2's gain drifts 1.03 -> 0.70 -> 0.61 across rounds while P4 and P5 settle by round 3 at 1.02 and 0.92, which is the same discretisation drift recorded when the gains were first fitted, and three demo shots have now refuted three decisions taken on them |
| 08-19 | **the solver run on DIII-D, to test MAST's choices where there are labels** **[levels corrected 08-19 — see the CORRECTION row: the eval shots overlapped the gain fit; the paired contrast stands, the absolute -0.050 -> +0.356 does not]**  — `solver/` split out as a machine-neutral package (`machine.Machine` carries the envelope, the wall radius, the column names, the flux sign), `solver/d3d.py` reads DIII-D, `solver/validate_d3d.py` fits coil gains and scores on shots never fitted on | **the physics transfers and the equilibrium does not.** Integrating `-Delta* psi / (mu0 R)` inside the shipped LCFS gives **+1274 kA against a shipped Ip of +1192**, the same 7% as MAST, so the sign, the units and `Delta*` are right on both machines. But the free-boundary solve puts the magnetic axis at **Z = -1.0 m against a true -0.03** — it slides a whole metre. Pinning the axis vertically takes held-out R2_psi from **-0.050 to +0.356** over 2245 frames of 14 shots. The pin is binary, not tunable: bands of 0.40, 0.15, 0.06 and 0.03 m all give +0.35631 to five decimals | **an elongated plasma has no stable vertical equilibrium** — real tokamaks hold it with feedback and EFIT pins it with the magnetic measurements this challenge withholds, so a Picard iteration slides to wherever the vacuum field happens to be stable. Pinning costs almost nothing because the real axis barely moves: **Z_axis = -0.0172 +/- 0.0413 m over 5231 live frames of 40 shots**, inside one 0.050 m grid cell on 89% of them. **And it does NOT transfer to MAST**, which is the result that matters for the plan it was built to serve: the same pin on MAST moves R2_psi 0.91579 -> 0.91562 and the axis error from -0.0360 to -0.0347 m, i.e. nothing, because MAST's centre column already anchors the plasma. So DIII-D's dominant pathology is not MAST's, and at R2_psi 0.356 against MAST's 0.916 it is **not yet a usable proxy** — the 3-shot cross-validation it was meant to host has to wait for the DIII-D solve to be worth cross-validating. Two calibration lessons paid for on the way: the gain fit must run **outside the plasma boundary** (inside it one filament cannot represent a distributed current, and fitting over the whole grid returned F-coil gains from -1.4 to +2.8 and an alternation that oscillated), and it must project out a **per-frame plasma amplitude**; with both, the F-coils come out at a median of **0.965, 10-90% 0.855-1.137**, which is independently what `my_experiments/coil_field.py` reports. ECOILA is degenerate with a constant offset, so it is fitted in a second stage with no constant projected out and takes 107.3 |
| 08-18 | **the local MAST metric was scoring on DIII-D's grid** — found while asking why D_LCFS is a Hausdorff distance and where its worst point sits. `local_score.py:83` set `MACHINE = "DIII-D"` as a module constant and line 485 loaded `d3d_envelope.npz` unconditionally, taking R and Z from it | so every MAST frame this fork scored had its LCFS extracted on a grid running **R = 0.84..2.54 instead of 0.06..2.00**, against DIII-D's vessel envelope, under **AXIS_SIGN = -1 instead of +1**. Fixed by reading the machine, the grid and the envelope off the shots themselves. The three demo shots, before and after: **S 0.6289 -> 0.5930**, 1 - D_LCFS **0.9004 -> 0.6664**, Consistency **0.0742 -> 0.0185**, R2_psi 0.9158 -> 0.9165 | **this is the explanation for the whole day's local-versus-board confusion, and it exonerates the local metric going forward rather than only condemning it.** Against the board's 0.5566 the old local numbers were a 4x lie on D_LCFS (0.0996 against 0.4121); the fixed ones are 0.3336 against 0.4121, and every term is now local slightly better than fold, which is what three shots against 1206 is SUPPOSED to look like. R2_psi was never affected — it is a pooled sum over pixels and touches no mask, sign or contour — which is why the Thomson source's local gain and fold loss remains a real 3-shot generalisation failure and not an artefact of this. The bug survived because `local_score.py` was written when DIII-D was the only machine with released truth, and MAST was fed to it through `--files` without anyone auditing what the mask was. It was caught by a diagnostic that computed D_LCFS independently and got 0.3317 where the scorer said 0.0996 — the same shape of catch as the impossible negative in D5 and the impossible R2 in D6: **a number that two routes disagree about, not a line of code anybody reread** |
| 08-18 | **the Thomson-driven source, scored on the fold** — submitted after it raised R2_psi on the three demo shots from 0.9158 to 0.9370, bundled with an affine q95 read off psi_N = 0.80 | **S_MAST 0.5566 -> 0.5170, i.e. -0.0396**, and the bundle separates cleanly because its two halves touch disjoint terms. The MAP, which only the source changed: R2_psi **0.8895 -> 0.8149** and D_LCFS **0.4121 -> 0.5155**, together **-0.0514 of S**. The SCALARS, which only the affine changed: R2_qb 0.0574 -> 0.1358, q95 0.108 -> 0.178 and betaN 0.007 -> 0.094, **+0.0118 of S** — earned while the map underneath them got worse | **the third and largest demonstration that three demo shots do not represent 1206, and this time the sign flipped rather than the magnitude.** The first two were D_LCFS 0.0996 -> 0.4121 and Consistency 0.0742 -> 0.0000 — local numbers that were merely optimistic. Here a change that gained **+0.021 of R2_psi locally lost 0.075 on the fold**, so the local metric did not just overstate it, it pointed the wrong way. Why: `p_scale` is one global number tuned on three NBI-heated discharges, where the fast-ion fraction it stands for varies shot to shot, and Thomson's own liveness varies across a fold in a way three shots cannot show. The mechanism is not refuted — `gs.Profile` keeps it behind `p_scale > 0` — **fitting its constant on three shots is.** Reverted to the parametric source and the 0.5566 gains. The affine on the scalars is KEPT, but note what came with it: the proxy surface belongs to the source, not to the metric, and reverting the source moved the best level back from psi_N 0.80 to 0.90 and the local q95 from +0.588 to +0.205. **No submission was spent on the revert** — it restores a board entry that already exists |
| 08-18 | **the pressure term stops being a fitted fraction and becomes a measurement** — MAST's Thomson is a midplane laser with a per-channel major radius, so `p(R)` is observed; mapped onto psi_N inside the Picard loop and differentiated it gives `p'(psi)` directly, and the source becomes `J = p_scale * R p'(psi) + lam (1 - psi_N^alpha)^gamma / R` with `lam` still fixed by the measured Ip. One global `p_scale` replaces the global `beta` | **R2_psi 0.91579 -> 0.93392** on the demo frames at the shipped gains, and after re-alternating the coil gains against the new source **0.9370**. The sweep has a real optimum rather than a preference for more: 0.875 at p_scale 0.25, 0.889 at 0.5, 0.908 at 1.0, **0.934 at 2.0**, 0.668 at 4.0. Then q95's correlation with the shipped `efit_q95` went **+0.453 -> +0.767** and, with the affine the earlier diagnostic had already called for, R2_q95 **+0.195 -> +0.588** and R2_betaN **+0.066 -> +0.295**; R2_qb **0.1307 -> 0.4413** against the leader's 0.5079. Local composite **0.6282 -> 0.6583** | **the one change that moved the term everything else routes through, and `p_scale = 2` is physics rather than a fudge.** On top of `2 n_e k T_e` it means an effective `4 n_e k T_e`, and these are NBI-heated MAST discharges where the fast-ion pressure is comparable to the thermal — the sweep's own peak at 2.0 with a collapse at 4.0 is what that predicts. Two further readings. **P2's gain moved 0.564 -> 0.726, toward 1.0**, which says the old value was partly compensating for the source and not only for the coarse grid near the centre column. And the argmax of the refinement grid (p_scale 1.5 / alpha 0.5, 0.93778) is NOT what ships: the plateau spans 0.921-0.938 with neighbours at 0.778, and the leaderboard has already priced three-shot argmaxes at local D_LCFS 0.0996 -> 0.4121. The middle of the plateau ships. **What went the other way, and is not hidden**: local D_LCFS 0.0996 -> 0.2326 and local Consistency 0.0742 -> 0.0000, with volume +0.17 -> -1.02 and li +0.35 -> -0.34. Both of those local numbers were the ones the board already showed to be worthless (0.0742 -> 0.0000 there), so the submission goes out on the two terms that are trustworthy and carry 70% of the weight |
| 08-18 | **is q95 limited by the method or by the map?** `mast/diagnose_scalars.py` — every estimator run twice, once on our solved flux and once on the shipped `efit_psirz` of the same frame, each with its own free constant so that SHAPE is compared and not level, over the 115 demo frames | **the contour integral is right and the map is what limits it.** q95: our map corr **+0.453** / R2(affine) 0.205, the TRUE map corr **+0.899** / R2(affine) **0.809**. betaN from Thomson: our map corr +0.258 / 0.066, the TRUE map corr **+0.673** / **0.453**. And the alternative route is dead on arrival: the Uckan cylindrical q95 scores corr +0.165 on our geometry and **-0.113 on the TRUE geometry** | **the plan this was written to test is refuted, and cheaply.** The reading of the leaderboard was that R2_{q95,betaN} is 40% of the deficit and 'needs no flux map at all', so it could be fixed on its own — by a regressor transferred from DIII-D, or by a formula. It cannot: both scalars are almost entirely downstream of the flux map, and on a perfect map the methods already in `mast/scalars.py` reach 0.81 and 0.45. **Everything routes through R2_psi**, which is also what the leader's numbers said from the other side — their R2_qb of 0.508 against our 0.057 sits beside their R2_psi of 0.973 against our 0.889. The Uckan result is worth keeping separately: a cylindrical q95 with a shape factor is standard practice and it ANTI-correlates with the truth on a spherical tokamak at aspect ratio 1.3, so the formula that would have been the obvious shortcut is worse than useless here. One free gain on the way: the contour integral wants an AFFINE calibration, not a scale — 0.809 against 0.551 on the true map — and the shipped one fits through the origin |
| 08-18 | **the MAST submission, scored** — the Grad-Shafranov solve on all 1206 `mast_public_test` shots, against the three demo shots it was calibrated on, and against the leaderboard's best Challenge 2 entry | **S_MAST 0.5566, G_ratio 0.560**, from 0.0000. Term by term, ours / demo-local / the leader: R2_psi **0.8895** / 0.9158 / **0.9732**; R2_qb **0.0574** / 0.1307 / **0.5079** (q95 0.108 vs their 0.722, betaN 0.007 vs 0.293); D_LCFS **0.4121** / 0.0996 / **0.2109**; Consistency **0.0000** / 0.0742 / **0.1727**. Prediction-side LCFS failures 0.48% and derivation failures 0.34%, so none of this is crashes | **it transferred as a bulk quantity and not as a geometry, and the leader's numbers say where the ceiling really is.** R2_psi lost only 0.026 from demo to fold, but D_LCFS went 0.0996 -> 0.4121 and Consistency 0.0742 -> 0.0000: everything that depends on the fine boundary was overfitted to three shots. **The largest gap to the leader is the cheapest term** — R2_qb is 40% of the 0.168 deficit and needs no flux map at all; matching them there alone gives S 0.624. And the hardest: **their R2_psi of 0.9732 is ABOVE our measured ceiling of 0.954**, which was taken with the TRUE vacuum field substituted in, so the three-parameter global source profile is the binding constraint and no coil calibration reaches past it. Two readings that cost nothing: even the leader has R_axis -0.69 and Z_axis -0.33, which confirms those two are structurally unreachable on MAST (the true Z_axis has a spread of 0.0012 m against a 0.0625 m grid step), so 2 of the 7 Consistency slots are a write-off for everyone; and their D_LCFS histogram is BIMODAL — a peak at 0.10 plus a bump at 0.7-0.8 on ~6% of shots — where ours is a single peak at 0.30, which is the difference between being right and breaking on a shot class, and being uniformly imprecise. **Correction to the 08-18 entry above**: the local Consistency of 0.0742 was called a pessimistic floor on the grounds that three near-identical shots give the shape scalars no spread to be scored against. Half of that held (kappa -46 -> -1.0, tri_bot -38 -> -2.7) and half reversed (volume +0.17 -> -6.2, li +0.35 -> -0.62), and the net was zero. A quantity fitted on three shots has no floor |
| 08-18 | **Challenge 2, built from the equation instead of from data** — a free-boundary Grad-Shafranov solve for MAST: coil field exact from the 812 shipped conductor rectangles, plasma current `lam [beta R/R0 + (1-beta) R0/R] (1 - psi_N^alpha)^gamma` with `lam` fixed by the measured Ip, Picard iteration on a sparse `Delta*` factorised once, Dirichlet rim from the Green's function of the same current. Scored on the three demo shots in `parquet_data/`, which unlike the 1206 test shots carry `efit_psirz` — 115 frames of the only MAST truth in the released dataset | **S_MAST 0.6282 from 0.0000.** R2_psi **0.9158** (x0.55), R2_{q95,betaN} 0.1307 (x0.15), 1 - D_LCFS **0.9004** (x0.10), Consistency 0.0742 (x0.20). `G_ratio = S_MAST / S_DIII-D` is what Challenge 2 ranks, so about **0.63** against 0.0000 for a DIII-D-only entry, and the eligibility floor of R2_psi > 0.6 is cleared by 0.32 | **the largest single move this fork has made, and it is not a model.** Challenge 1's whole remaining budget is 0.0056 of S; this is 0.63 of a leaderboard column that was empty. Three measurements carried it. **The units and the sign closed in one test**: integrating `-Delta* psi / (mu0 R)` over the cells inside the shipped LCFS gives 532/633/660/635 kA against an Ip of 496/609/637/618, which fixes psi as Wb/rad, the currents as kA and `AXIS_SIGN[MAST] = +1` all at once. **The boundary needs connectivity, not a maximum**: it is either the flux at MAST's centre column (`efit_lcfs_r` bottoms out at 0.1963 m on all 115 frames, to four decimals) or an X-point, and taking the higher without asking what is attached to the plasma puts it **19% of the axis-to-boundary span too high on the median frame**, because the centre column carries the solenoid's own flux hill; with the test, median **0.045%**. **The coil gains check themselves**: the three coil families whose geometry the dataset ships in full come out at P4 1.010, P5 0.961, P6 0.994 on the averaged-turn basis, which is what licenses the two that are genuinely fitted (P3 8.69, solenoid 169.8) |
| 08-18 | **the alternating calibration, and what made it converge** — eleven coil gains fitted against the demo truth minus the plasma flux a solve at the previous gains produced, repeated | **undamped and per-column it diverges**: R2_psi 0.639 -> 0.291 -> **-0.428**, with the well-posed gains swinging 0.65 -> 1.44 -> -0.51 and P6L/P6U landing at 0.45 against -2.35. Tying each up/down pair to one gain and damping the update to 0.4 makes it monotone: 0.634 -> 0.873 -> 0.928 -> 0.916, settling at **0.9158** over eight rounds | **both fixes are physics, not numerics, and that is why they worked.** A turn count belongs to the coil and not to the direction its current runs, so an up/down pair shares one gain — and P6L = 0.45 against P6U = -2.35 was the fit telling us it was fitting something else. The damping is the same statement about time: the plasma flux being subtracted came from a solve at the PREVIOUS gains, so the two are coupled and a full step overshoots. Separately, the ceiling was measured before any of this: substituting the TRUE vacuum field (Delta* of the truth, current kept inside the LCFS, solved forward) and sweeping the three profile numbers reaches **0.954**, so the coil calibration costs 0.04 of R2_psi and the three-parameter source shape costs 0.05 |
| 08-18 | **two performance findings on the MAST solve, both worth an order** | the boundary search tested all candidate levels and took the max; testing them in DESCENDING order and returning the first that binds is **exactly equivalent** (S 0.6282 both ways, to four decimals) and cuts 2.0 s a frame to 0.72 s. Then pinning BLAS to one thread cuts 0.72 s to **63 ms** — **11x by removing threads** | every array in the solve is 65x65: a sparse back-substitution of 3969 unknowns and a 256x4225 matvec, so a threaded BLAS spends all its time in fork/join. On this box `nproc` reports 64 against 32 real cores, so the default is doubly wrong, and a pool of workers makes it worse rather than better — the first full-split attempt ran at load average 69. `mast/predict.py` now takes `threadpool_limits(1)` around the frame loop, which is what makes `--jobs 28` an actual 28x. The whole 1206-shot split: **1.1 h on one core, 12 minutes on the box.** Also recorded because it nearly went unnoticed: the demo shots ran at 0.24 s a frame and the test shots at 2.0 s, and the difference was contention with another job, not the data |
| 08-18 | **warm-starting each MAST frame from the previous frame's converged current** — the obvious improvement, since a discharge evolves on tens of milliseconds and the frames are 5 ms apart | **R2_psi 0.916 -> 0.443, composite 0.6086 -> 0.3420** | **refuted, and the mechanism is the opposite of the one assumed.** A converged current is sharply concentrated, and starting the boundary search inside it keeps the plasma there; the broad seed blob is what lets the solution find its own extent. The motivation was real — the per-frame scatter of `volume` is 2.75 m3 against a true spread of 0.48, so the boundary IS jumping between frames — but continuity is not the fix for it |
| 08-18 | **what is a scalar worth if it were solved outright?** `diagnose_budget.py` — `finalize_machine` averages the seven pooled R2 with equal weight, so a scalar's whole remaining budget is `(W_CONS / N_CONS) * (1 - R2_j)`, read off the three-artifact ensemble's own residuals over the 70 held-out shots, beside the four composite terms from the production log | **kappa +0.00114, li +0.00092**, R_axis +0.00055, volume +0.00053, Z_axis +0.00033, tri_bot +0.00026, tri_top +0.00022 — all seven perfect at once is **+0.00396**. Term by term: R2_psi +0.00011, R2_qb +0.00057, D_LCFS +0.00098, Consistency +0.00396, i.e. **0.00562 of S left in the whole of Challenge 1**. The seven residuals in metric units have a leading principal component of **34.8%**, li shares **19.6%** of its variance with the other six, and the largest off-diagonal correlation is kappa-tri_bot at 0.36 | **li is closed by arithmetic rather than by a ninth failed gate, and so is every other single-scalar mechanism.** Every line in both tables except "all seven at once" is below the **0.0013** a paired production run resolves — a target that cannot be confirmed even when it is hit perfectly, which is the regime selection bias feeds on. The obvious escape is measured and shut: at a leading PC of 34.8% these are seven problems and not one, so a mechanism aimed at one scalar collects one budget and the per-scalar column is a ceiling, not a share. **This should have been the first thing computed, not the last.** C1's cost shares (kappa 23.3%, li 18.7%) were read as a ranking of where to work for three days; they are shares of a total that is itself four times too small to act on, and the conversion is one line of the scorer. What survives is A25 — averaging, the only axis this arithmetic does not cap |
| 08-18 | **submitted** the three-feature-set ensemble — the production five averaged as decoded maps with the two A19 vessel arms | leaderboard **0.9938**, against 0.9932 for the previous submission — **+0.0006**, where the local salt-0 gain was +0.0005 and the mean over the two unseen salts was +0.0014 | **it transferred, and the useful number is the OFFSET rather than the gain.** Local salt 0 minus the board has now read **0.0018 / 0.0013 / 0.0012** over three submissions — a stable constant, not a widening gap — so salt 0 predicts the board to about a thousandth once that offset is applied, and it did here: +0.0005 predicted, +0.0006 delivered. Note what did NOT predict it: the unseen-salt mean of +0.0014, which over-called the board by a factor of two. That is the reverse of 08-14, where the fresh-fold estimate was too pessimistic, so two submissions now disagree about which local estimator to trust and neither reading should be leaned on. Transfer by the old accounting is 43%, against 91% and 116% — the worst of the three, and the reason to quote the offset instead |
| 08-18 | **C10/A24 run: li's block of M built from the exact poloidal-field form instead of the linearisation**, at the same trace, paired on salt 0, four seeds, production shares | control **S 0.9941**, field **0.9939**, i.e. **-0.0002**. And the part that decides: **li's own R2 falls 0.9642 -> 0.9588**, kappa 0.9586 -> 0.9566, Consistency 0.9783 -> 0.9776 | **refuted, by the failure mode it pre-registered.** The clause was "if S moves and li's RMS does not, the mechanism is wrong even if the number is right"; what happened is worse — li went BACKWARDS by 0.0054, so the form that ranks li's error best in a diagnostic makes li worse when it is put in the loss. **The lesson generalises and is the reason to keep this row.** C10 measured Spearman between `dc^T G dc` and li's SQUARED ERROR — that is which FRAMES are bad. A loss needs to know which DIRECTION to move in. Those are different questions, and a form can win the first and lose the second: G is the energy of the field error and is missing the first-order term `2 * integral B_p . dB_p`, which is precisely the part that carries direction. Every future "this form correlates with the cost, put it in the loss" argument in this file now has a measured counterexample |
| 08-18 | **C10, does the poloidal-field form know WHERE the geometry error is?** `target_metric.poloidal_field_form` is defined and called from nowhere, while params.yaml claims it "stays". Per held-out frame, Spearman of `dc^T G dc` against each scalar's SQUARED error, against two nulls: `|dc|^2` (Parseval) and `dc^T M dc` (the jacobian metric that actually ships), M rebuilt from 300 probe frames at the artifact's own step | against Parseval the field form wins broadly — li **+0.1583**, R_axis +0.1197, kappa +0.0407 — but **against the loss that ships it wins on li alone: +0.0519**, with R_axis −0.0403, Z_axis −0.0439, volume −0.1206, kappa −0.0045, and the two triangularities +0.007 and +0.003. Absolute Spearman on li: 0.3859 Parseval, 0.4923 the loss, **0.5442 the field form** | **the first positive gate of the sweep, and it is narrow in exactly the way the physics predicts.** G is written for the poloidal-field energy and li IS a normalised <B_p^2>, so it discriminates li's error better than a linearisation does — and for the SHAPE scalars, which are contour geometry rather than field energy, the linearisation is better and G is worse. So the finding is not "use the field form" but "G belongs in li's block of M and nowhere else", which is a substitution with no free parameter: the 08-13 `field` loss died because a hand-set lambda could not be located, and there is nothing to locate here. **Read it as marginal**: +0.0519 clears a +0.05 gate by 0.0019, on one fold, and it is the one gate of eight that cleared. It matters because C9 independently put li furthest above its own noise floor (3.1x) and C1 put it at 18.7% of the geometry cost. Side correction: the linear/actual ratios measured here are **1.05-1.20**, against the 1.11-1.39 in the D3 entry and the 1.65 for li recorded on 08-13 — the metric is better calibrated than the record says |
| 08-18 | **D9's remaining half — does the FUTURE of a shot say anything the past does not?** The accepted vessel bank run BACKWARD: the same leaky integral of dI/dt at the same four timescales on the time-reversed signal, ECOILA and plasma_current, eight columns against the production 84, on the 70 held-out shots | **both arms negative.** Predicting the TRUTH: li **−0.0409**, volume −0.0525, kappa +0.0023, R_axis +0.0016. Predicting the ENSEMBLE's residual: res_li **−0.0017**, res_kappa +0.0002, res_R_axis −0.0000, res_volume −0.0062, against a pre-registered gate of +0.02 on li and kappa both | **refuted, and more decisively than D7 was.** There the two arms disagreed and the ensemble had to break the tie; here the anti-causal columns fail to help even a LINEAR model predict the truth, so there is no mechanism to argue about. That closes the last untested construction in Group D. Worth stating why it was worth testing anyway: every column in `features_for_row` is instantaneous, centred or strictly causal, the scored shot is complete and offline, and C6 measured the GRU's backward channels as its long ones (median tau 122.0 frames against 90.7 forward) — the premise was sound and the measurement says the eight-column linear version of it carries nothing |
| 08-18 | **C9, how much of what is left is EFIT's own jitter?** The seven ground-truth scalars' noise estimated from their second difference over triples whose both intervals are the nominal 20 ms (96.7% of frames survive the gap filter), against the model's residual RMS on the same frames | noise / model-error ratio per scalar: **tri_top 0.87, tri_bot 0.78** — at the floor; volume 0.50, Z_axis 0.61, R_axis 0.59; **kappa 0.47 and li 0.32** — the model errs two to three times more than the label jitters. Consistency is 0.98021 against a ceiling of **0.98996 / 0.99437 / 0.99880** over three readings, i.e. **+0.00195 / +0.00283 / +0.00372 of S** | **the geometry terms are NOT at the label floor, and where the room is has a per-scalar answer.** Every reading leaves more than the 0.0013 a paired run resolves, and the room is exactly where C1 put the cost: kappa and li, 42% of it between them. It also confirms from a new direction the standing rule that the triangularities are capped — they sit at 0.78–0.87 of their own noise floor and are finished. **The lag sweep is reported as a failure to decide rather than as a result**: sigma rises 1.14x–1.63x from k=1 to k=3, which correlated label noise and genuine curvature over a 120 ms span both predict, so the white-noise assumption behind the middle reading is untested and the pessimistic reading (charging the whole rise to correlation) is the one the conclusion has to survive. It does |
| 08-18 | **D6, is Thomson summarised in the wrong coordinate?** The thirteen summaries per system recomputed on psi_N from the TRUE `efit_psirz` — o-point from the scorer's own `find_o_point` on the sign-corrected map, boundary flux from the shipped `efit_lcfs` polygon — and asked what they explain of the ensemble's residual beyond the 84 machine-coordinate columns, over the 14830 tail frames | the flux summaries are defined on **98.0%** of frames. Out-of-sample increments with alpha chosen by a shot-grouped split inside the fit half: **res_li −0.0075**, res_R_axis +0.0002, res_kappa +0.0072, res_volume +0.0119, against a pre-registered gate of **+0.10 on li**. The control's own R² is −0.006 to −0.059, i.e. the residual is not linearly predictable from ANY of these columns out of sample | **refuted, and it is an ORACLE that fails — psi_N came from the truth, which no inference-time version can have, so no two-stage build can do better.** Close the family. The second reading is the larger one: at the selected alpha the 84 production columns explain **nothing** of the held-out residual either, which is it-6 (0.008 explained from 252 history summaries) and D7's 0.0% correction saying the same thing a third way. Also recorded because it nearly went the other way: at a fixed alpha = 1 every arm scored between −0.49 and −1.14 with increments swinging −0.35 to +0.38, on a design ridge itself flagged at rcond 1e-51, and the flux block was being penalised unstandardised against columns the artifact's scaler had already normalised. Both were bugs in the TEST, and the tell was a control R² that no sane regression produces |
| 08-17 | **D5, the ORACLE ceiling on every per-shot confidence idea** — each shot's derived scalars shrunk toward the fold mean by the factor that minimises that shot's OWN true error, over the 14830 tail frames, with each scalar in units of its own SS_tot | one factor per shot **+0.00167** of the pooled mean R² = **+0.00033 of S**; one per shot and per scalar **+0.00518** = **+0.00104 of S**; the degenerate per-frame arm returns exactly 1.0000 as it must. The factor itself has median **1.0095** and a 5–95% range of 0.9711–1.0914, and only **35% of shots want shrinking at all** | **the whole family is closed, and the algebra it was meant to check was right.** A23's shrinkage half was priced at +0.0001 of S by the closed form for optimal shrinkage at R² = r; the oracle says +0.00033 for the per-shot version and +0.00104 even when it is told which of the seven functionals is wrong — both below the 0.0013 a paired run resolves, and a real estimator recovers only a fraction of an oracle. The factor clustering at 1.0095 with two thirds of shots wanting to be pushed AWAY from the mean is the same statement twice: there is nothing here for a confidence signal to do, however good it is. Note the first attempt got **−0.00153** for the shared-factor arm, which an oracle cannot produce — it was minimising the raw sum of squares, where `volume` in cubic metres dominates `kappa` near 1.8, and the impossible sign is what caught it |
| 08-17 | **D1, can the seven scalars be estimated better than by reading them off the map?** The production MLP — same architecture, schedule and stopping rule — fitted on the same 208 features to the seven ground-truth functionals from the `aux_targets` cache, 1.25M rows, stopped on the 1338 validation shots at step 50750 of 69150. Blended against `f_j(psi_ens)` on half the 70 tail shots and read on the other half, split at a shot boundary | pooled mean R² over the seven, on the unseen half: readout **0.97944**; **readout + head 0.98301, +0.00357**; affine on the readout alone **−0.00047**; the Jensen column (mean over four seeds of `f_j(psi_m)`) **−0.00029**, and it DILUTES the head (+0.00283 together against +0.00357 without it); the head alone **−0.00128**. Per scalar: kappa **+0.0119**, Z_axis +0.0046, tri_bot +0.0035, tri_top +0.0029, volume +0.0016, R_axis +0.0000, li **−0.0047** | **the mechanism is real and the size is not, and it lands within 7% of what was predicted before it ran.** Consistency weighs 0.20, so +0.00357 of the mean R² is **+0.00071 of S** against a predicted +0.00076 and a resolution of 0.0013 — and the transport would then realise about half of it through the linearisation. **Fails its pre-registered +0.0065 by a factor of 1.8, so no transport is written**, and six hypotheses that all reduce to this one number close together. Three things the controls bought: the affine arm going NEGATIVE says the head found information and not a bias, which is what separates this from A7/it-3; the gain concentrating on kappa is exactly where C1 put the money (23.3% of the geometry cost); and the **Jensen gap does not reproduce** — measured at −0.00029 here against a claimed +0.00055, and it makes the blend worse, so the four routes six lenses found were not four routes after all |
| 08-17 | **D7 and D9 together — is the tail a different function, and does the model know where in the shot it is?** Ridge fitted on the last quarter of each shot against a same-sized MID-SHOT control slice, 1056 fit shots and 211 held out; then the same question asked of the production ensemble's own residual, corrected on half the held-out shots and read on the other half | **ridge says yes, loudly.** On held-out tail frames it goes R² 0.146 (global fit) → **0.706** (tail-only fit), against 0.607 → 0.660 for the mid control — a contrast of **+0.506** — and in absolute terms 60.2 → **20.7** (Wb/rad)²/frame against 11.1 → 9.6. Four clock columns (phase, phase², 1−phase, (1−phase)²) recover **46.6%** of it. **The ensemble says no.** Its own error is **0.0165** on the tail against 0.0112 on the middle, a ratio of 1.47x and not ridge's 5.4x, and a linear correction fitted on held-out shots removes **0.0%** — with the clock and without it, on both slices | **both entries die at the gate, and it is the A19 false positive reproducing exactly.** A19 arm 1 moved `ridge` +0.0216, the largest move any change has made to the linear baseline in this fork, and the MLP was flat; the pre-registered warning that a ridge gate is invalid for this feature family was written into D9 the same afternoon and then earned. What the pair actually measures is that **the trained ensemble has already absorbed the position dependence** a linear model cannot represent — which is what A19's accepted vessel integrals were for, and consistent with them cutting the last decile 10.0%. So A22's under-representation branch closes, D9's clock half closes with it, and the tail's remaining cost is not the model failing to know where it is. Note the reading also refines C7: the tail's coefficient error is 1.47x the middle's while C1 put its GEOMETRY cost at 2.4x its share, so most of the tail's expense is in how the functionals read the map, not in the flux error underneath |
| 08-17 | **D3, is the Jacobian probe step overdue a refit?** — the free gate on the sweep's top-ranked entry. Per-component held-out residual RMS of the production ensemble against the artifact's own basis, 14830 frames, nothing fitted and nothing scored | the control passes first: R²ψ implied by the coefficients is **0.999836** and sqrt(1−R²ψ) = **0.0128**, which equals the equal-spread step to four decimals — so `jacobian_delta` **is** sqrt(1−R²ψ), as claimed. Then the assumption underneath it fails: the residual is **not** spread evenly. Eight components carry 90% of its energy and 13 carry 95%, giving honest steps of **0.0304** and **0.0246** against the configured **0.0332**. Component 0 alone sits at 0.0657, twice the configured step; components 40–49 at 0.0006–0.0010. Largest/smallest ratio **112.6x** | **the refit is NOT licensed, and the entry dies at its own gate for one-tenth of a run.** The "2.4–2.6x too large" reading is an artefact of dividing the residual equally over 50 directions — measured where the error actually lives, the configured value is **1.09x** the 90%-energy step. Four accepted changes have landed since it was set and it is still right, which is the opposite of the conclusion the sweep drew. What IS a finding is the 112x spread: one isotropic step for fifty components is exactly right for the three that matter and 100x too large for the tail, so the probe describes the model's error well only because the error is concentrated. Records against A7's "importance is not ablation" as the same shape of mistake — an assumption about how error distributes, asserted rather than measured |
| 08-16 | **C5, what MAST actually ships** — read a `mast_public_test` row column by column and compared it to what the DIII-D pipeline needs | **no targets at all**: the only `efit_*` columns are `efit_grid_R`, `efit_grid_Z`, `efit_times`. No `efit_psirz`, no `efit_q95`, no `efit_beta_n`, and there is **no `mast_train` config in the dataset** — 1206 shots, 3.1 GB, all of it test. Of the 21 magnetics signals the fork reads, **exactly one exists on MAST**: `plasma_current`. The other 13 are p2l/p2u/p3l…p6u, sol, efps, tf. Grid 65x65 over R 0.060–2.000 and Z ±2.000 against DIII-D's outboard-shifted one; **53 frames at a 5 ms step** against 234 at 20 ms; and `coil_input_column` maps **812 filaments** onto the 13 MAST current columns | **Challenge 2 is zero-shot transfer, not "a bit more work".** There is nowhere to fit a MAST model from, so the only route is machine-independent inputs — which is exactly what `features.inputs: coil_pca` was built for and why it is in params.yaml with nothing to show on DIII-D. The 812 rectangles mean `coil_field.py` runs there unchanged, as its docstring claims. What does NOT transfer is everything else: a spherical tokamak's equilibria, a four-times finer clock, and a flux grid starting at R = 0.06 |
| 08-16 | **does the feature-diversity axis keep paying?** — four more members, each different along one axis nothing else varies (coil-flux directions alongside the currents, a 40-direction basis, the raw derivative block alone, a Huber loss), plus A4, added to the confirmed three | **it saturates at three.** prod+driving+poloidal is 0.9950; adding a fourth gives 0.9950–0.9951 whichever fourth, and adding ALL FIVE gives **0.9951**. Alone the new members score inpboth 0.9937, pca40 0.9940, huber 0.9941, draw 0.9942, **a4 0.9944** against a 0.9940 control | **the gain is from having two or three genuinely different views, not from member count.** +0.0001 for four more fits is nothing, and it says the confirmed submission needs no rebuilding. Also the cheapest possible reading of A4: **+0.0004 alone**, so absorbing the validation window does help, below what one paired run resolves, and it adds nothing the other members were not already contributing |
| 08-16 | **the stack on salt 4** — the same four fits and two scorings | control **0.9931**; weight average alone **0.9932 (+0.0001)**; two feature sets **0.9946 (+0.0015)**; three **0.9949 (+0.0018)** | **ACCEPTED.** Averaging across feature sets is +0.0005 / +0.0009 / +0.0018 on salts 0 / 3 / 4 — positive everywhere, **mean +0.0014 on the two salts nothing was selected against**, and larger on the unseen folds than on the selection one for the fourth time in this fork. Above the 0.0013 a single paired run resolves. The weight average is +0.0003 / −0.0002 / +0.0001, mean **+0.0000**, and is dropped |
| 08-16 | **the stack on salt 3, which nothing was selected against** — control against the weight average alone, then two and three feature sets averaged as decoded maps, four seeds each | control **0.9934**; weight average alone **0.9932 (−0.0002)**; two feature sets **0.9940 (+0.0006)**; three **0.9943 (+0.0009)** | **the stack splits, and cleanly.** Averaging across FEATURE SETS confirms and comes back LARGER than on the fold it was found on (+0.0009 against +0.0005), which is the opposite of selection bias and the third time that has happened in this fork. The weight average does not confirm: +0.0003 on salt 0, −0.0002 here, i.e. zero. Since the treatment arms all carry it, the feature-set effect alone is about +0.0011 |
| 08-15 | **A20, a running average of the WEIGHTS** — kept alongside the live ones, evaluated beside them, the fit keeps whichever scores better. Production, salt 0, four seeds, decay 0.999 | ensemble **0.9943 against 0.9940**, but the individual models move far more: **0.9930 / 0.9927 / 0.9927 / 0.9929 against 0.9920 / 0.9916 / 0.9920 / 0.9927**, a mean of **+0.0008 each**. The averaged weights won **1537 / 1838 / 1766 / 1261** of roughly 1630 evaluations | **+0.0008 to one net and +0.0003 to four, and the gap IS the result.** The average removes the noise of one trajectory rattling in its basin; averaging four seeds removes much of the same noise a second time, so most of what it buys has already been bought. Winning 94% of the evaluations it was measured at means a fit keeping the live weights is keeping the worse of two things it already holds |
| 08-15 | **where the vessel integrals actually act** — the A19 driving arm decomposed onto the same 14830 frames as the control, four seeds each | total geometry cost **0.00529 → 0.00504**, and it is not spread evenly: the **last decile falls 10.0%** (0.001314 → 0.001183) and carries **52% of the whole gain**, while the first decile gets 1.5% worse. Per scalar the RMS residual moves **li −8.7%**, kappa −4.1%, volume −3.1%, against tri_bot +2.9% and R_axis +3.8% | **the mechanism, not just the number.** C1 found the last decile carrying 24% of the cost; C7 said it was coefficient error and not sensitivity; C6 measured that the sequence model built 1.8–2.4 s integrators and that its output reads them; A19 hands the same timescales over as eight columns and the gain lands in that decile and on `li` — the current-profile quantity whose evolution IS diffusion on that timescale. Four measurements, one story, and the scalars that improve are the expensive ones |
| 08-15 | **an ensemble across FEATURE SETS, not across seeds** — the production five averaged with the two A19 arms as decoded flux maps, all three fitted on the same split, salt 0, nothing retrained | **0.9950** for prod+driving+poloidal, 0.9949 for prod+driving, 0.9948 for prod+poloidal, against **0.9945** for production alone. Every term up: R2_psi 0.9999, R2_qb 0.9974 (+0.0007), 1−D_LCFS 0.9908 (+0.0006), Consistency 0.9819 (+0.0017) | **+0.0005 from three feature sets whose individual scores are 0.9945, 0.9944 and 0.9940** — the two arms that were flat or marginal ALONE are worth something TOGETHER, which is what decorrelated error looks like and is the same lesson as the sequence model (worst alone, most valuable in the average). A deterministic rescore, so there is no optimisation noise in it; the uncertainty is the 70-shot sample and the fact that three combinations were tried. Goes to two unseen salts |
| 08-15 | **A19 arm 2, the same integrals on the two signals the physics names** — ECOILA and plasma_current only, 8 new columns on 84, production, salt 0 | ensemble **0.9944 against the control's 0.9940, +0.0004**, and **all four seeds improve**: 0.9924 / 0.9922 / 0.9927 / 0.9925 against 0.9920 / 0.9916 / 0.9920 / 0.9927. Per term the gain is Consistency **+0.0012** and R2_qb **+0.0009**, D_LCFS flat. Validation loss 0.0295 against ~0.0315, the lowest of any arm run today | **below the 0.0013 a single paired run resolves, so not accepted — and the shape is exactly what was pre-registered.** The broad set was predicted to dilute and it did (arm 1, flat, 80 columns); the narrow set carries the same physics in 8 columns and moves every seed the same way. Four of four positive plus a lower loss is more than one number's worth of evidence, so this goes to two unseen salts rather than being dropped |
| 08-15 | **A19 arm 1, the vessel-state features on every poloidal signal** — four leaky integrals of dI/dt per signal at 20 / 100 / 500 / 2500 ms, 80 new columns on 84, production, salt 0 | ensemble **0.9940 against the control's 0.9940**. Per term 0.9998 / 0.9966 / 0.9901 / 0.9779 against 0.9998 / 0.9962 / 0.9901 / 0.9785 — R2_qb up 0.0004, Consistency down 0.0006, net zero. **But `ridge` moved 0.7651 → 0.7867, +0.0216** | **flat for the MLP and large for the linear model, which is the whole result.** The integrals carry real information — a linear map gains more from them than from anything else tried — and the MLP had already extracted it. That is not as surprising as it first looks: a coil's CURRENT is the integral of its own dI/dt, so the levels the net already sees are themselves accumulated history, and two hidden layers over 21 of them can build what an explicit EMA hands over. The missing state A22 is chasing is either already there in a form the net can use, or it is not of this form |
| 08-15 | **A13 proper — the composite decides, but only once the validation loss has been flat for 10000 steps.** Four seeds, production, salt 0, 400 monitor frames, against the same four seeds loss-stopped | **S 0.9940 against 0.9940.** Per term the treatment is 0.9998 / 0.9965 / 0.9899 / 0.9780 against the control's 0.9998 / 0.9962 / 0.9901 / 0.9785 — inside noise everywhere and marginally BEHIND on both geometry terms. And it is not for want of training: best step 84000 / 94000 / 98000 / 108000 against 58900 / 39150 / 80250 / 88550, over runs 35–100% longer | **refuted, and it answers its own confound.** The rule necessarily trains ~28400 steps past the loss peak where the control stops at 18400, so a third length-matched arm was pre-registered in case it won. It did not, so training longer bought nothing either — which is what the composite trace already said, flat from step 42000 on. The oldest open entry in the file, built, controlled and closed |
| 08-15 | **A13 with the composite deciding from step one** — production, salt 0, four MLPs stopped on the score itself over 150 validation frames | it stops EARLIER every time, and sometimes absurdly so. Best step against the loss-based control: **48000 / 46000 / 36000 / 16000** against **59250 / 50700 / 99650 / 61500**. `mlp2` recorded its best composite at step **2000**, the first evaluation it ever took, and needed 18000 more steps to match it | **killed before it finished, and the mechanism recorded instead of the score.** A 150-frame sample can flatter a barely-started net and the validation loss cannot — 0.0597 at step 2000 against an eventual 0.0324 is not a trained fit under any sampling. Selecting the argmax over ~34 evaluations of a FIXED small sample is ordinary selection bias, running inside a training loop where nothing was watching for it. Replaced by the gated rule, and the seq fit that would have cost 49 more minutes is identical in both arms |
| 08-15 | **the sequence model's weight in the ensemble, swept as a rescore** — ten weightings on a 70-shot SELECTION set of validation shots, then the best confirmed on the untouched tail | the selection curve is smooth and unimodal: 0.9925 at weight 0, 0.9929 at 0.11, **0.9930 at 0.20 (production)**, 0.9932 at 0.33 and 0.43, 0.9931 at 0.50, 0.9929 at 0.60, 0.9924 at 0.71, 0.9918 at 0.83, 0.9908 alone. On the TAIL: weight 0.20, 0.33 and 0.43 all score **0.9945** | **the flat optimum was already occupied.** +0.0002 on the selection set did not transfer at all, which is what +0.0002 against a 0.0013 resolution should do. The shape is worth keeping — a unimodal curve over ten points is real even where its peak is not resolvable — and it closes the global half of A23: one weight per member is already right |
| 08-15 | **A2, `calibrate_scalars: true`** at production, salt 0, against the recorded control | **S 0.9945 against 0.9945.** Per term +0.0005 on R2_qb, +0.0001 on 1−D_LCFS, −0.0001 on Consistency. Per scalar the seven move by ±0.0012 with no pattern: R_axis +0.0009, li +0.0006, tri_bot +0.0008, but **kappa −0.0010** and volume −0.0012 | **refuted, and the entry's own "open problem" paragraph called it.** The probe's ratios are 1.11–1.39 — near enough uniform that dividing by them is close to a uniform rescale of the Consistency block, which the target scaler then largely absorbs. The per-scalar moves are 1/20th of the per-scalar seed noise (0.018 on R_axis, 0.023 on Z_axis) and mean nothing individually; the composite is what resolves, and it moved 0.0000 against a resolution of 0.0013 |
| 08-15 | **C7, is an expensive frame under-weighted or simply harder?** — the per-frame Jacobian on 2998 frames, same probe step as the training loss, joined to C1's costs | sensitivity spreads **10.1× p99/median**, so B6's clause is not met — but Spearman(sensitivity, cost) is only **+0.209** and the quintile means are 0.74 / 0.79 / 0.65 / **0.56** / 2.25. By decile: ramp-up is **3.05× sensitivity and 0.51× implied error**, ramp-down is **0.98× sensitivity and 2.22× error** | **B6 was aimed the wrong way.** A Jacobian weight up-weights the first decile, which the model already predicts twice as well as average, and leaves the last decile — 24% of the geometry cost — exactly where it is. The two ends of a shot are expensive for opposite reasons and only one of them is a mis-weighting |
| 08-15 | **the last decile is not a representation problem** — the ceiling decomposed onto the same frames | the ceiling is FLAT across the shot: 0.000086 to 0.000120 per decile against a mean of 0.000100, and the last decile's basis share is **10.1%**, the LOWEST of the ten, against 21–33% in the middle | **the tail is not harder to represent**, so 27.2% of everything reachable sits in 10% of the frames. **Corrected the same day:** this was first written as also refuting EFIT label noise, and it does not — a noisy label is reconstructed by the basis exactly as well as a clean one, so a flat ceiling is what label noise would produce too. That branch is open |
| 08-15 | **what the sequence model actually buys, by region** — C1 run for `mlp` alone against the ensemble | the MLP loses 0.00603 of geometry cost, the ensemble 0.00490, a saving of **0.00113 — 18.3% overall, 24.9% of the first decile and 15.2% of the last** | **confounded, and superseded by the row below.** The ensemble is four seeds AND the sequence model, so this prices the ENSEMBLE by region and not the recurrence |
| 08-15 | **the same question with the seeds controlled** — `mlp+mlp1+mlp2+mlp3` against the five-member ensemble, same frames | the four seeds alone average 0.00700, averaging them gives 0.00529 (**saves 0.00171**), adding `seq` gives 0.00490 (**saves 0.00039**). By decile the recurrence's saving climbs monotonically: +2.4% at the start, 5–8% through the middle, **+14.5% at 0.8–0.9 and +10.1% at 0.9–1.0** — **34% of everything it buys is in the last decile, 52% in the last two** | **the earlier reading REVERSES.** Seed averaging does four fifths of the work and the recurrence one fifth, but that fifth is concentrated exactly where the memoryless models are worst. The one member carrying state helps most during ramp-down, which is what "the vessel current is largest and most transient there" predicts — evidence FOR the missing-state branch of A22, and it makes B5's unrun screen a live question rather than a debt |
| 08-15 | **C6, the memory the sequence model actually learned** — the update gate recomputed from the stored weights over 20 held-out shots, 256 channels per direction, no training and no scoring | **median tau 90.7 frames forward (1.81 s) and 122.0 backward (2.44 s)**; only 2.3% / 1.6% of channels sit under 2 frames, while **60.5% / 69.1% are over 50 frames (1 s+)**. Weighted by each channel's influence on the output (`delta`'s column norms) it is LONGER still — the top ten forward channels have a plain median tau of 212 frames — and Spearman(influence, tau) is +0.045 forward, +0.254 backward | **it is not a smoother.** The 08-13 refutation of temporal smoothing does not describe what this model built: the majority of its channels integrate over more than a second, the channels the output actually reads are the longest-memory ones, and a tau above ~373 frames simply means z stays near 1 for the whole shot — an integrator. That is the vessel-and-profile state A19 and A22 are chasing, learned rather than given |
| 08-15 | **C8, does the model know where it is wrong?** — the four seeds' per-frame disagreement against the ensemble's own per-frame cost, 14830 frames, nothing retrained | **Spearman +0.565**, against +0.209 for the Jacobian sensitivity. Monotone, unlike sensitivity: mean cost by quintile of disagreement is 0.20 / 0.32 / 0.52 / 0.99 / **2.97×**, the top fifth carrying **59.3%** of the cost. But by decile the disagreement is 1.55× at the start against a cost of 1.54×, and only **1.19× at the end against a cost of 2.61×** | **a free confidence signal that locates the loss 2.5x better than the metric's own linearisation** — and blind exactly where the money is. Seed residuals correlate +0.550 in the tail, so a majority of the error there is SHARED, and four seeds wrong the same way look confident. Disagreement measures variance and cannot see bias. Opens A23 (weights that vary, shrinkage), both without retraining |
| 08-15 | **is the tail bias or variance?** — pairwise correlation of the four seeds' per-frame residuals over the seven derived scalars | mean pairwise r = **+0.550 in the last decile** against **+0.615 in the middle** of the shot | **a majority of the tail's error is SHARED across seeds**, so it is bias of this model-and-feature class and more seeds cannot remove it — averaging is bounded to the ~45% that is independent. The tail is if anything slightly MORE variance-driven than the middle, not less. What this still cannot separate: a missing input from an unlearnable EFIT label |
| 08-15 | **C1, where the score is lost** — the same two geometry terms decomposed onto all 14830 frames on one additive scale (`diagnose_frames.py`) | the worst **1% of frames carry 22.4%** of the cost, the worst 5% carry 45.8%, the worst 10% carry 57.9%. By position: the LAST decile **24.1%** and the first 15.1%, the middle eight flat at 6–9%. By scalar: kappa **23.3%**, li **18.7%**, LCFS 18.4%, R_axis 11.3%, volume 10.8%, Z_axis 6.8%, the two triangularities 9.9% together | **the loss is heavy-tailed and the loss function does not know it.** A pooled R2 weights every frame the same; the ends of the shot cost 2.4x their share and one frame in a hundred costs 22x. That is B6's hypothesis measured directly, and its own refutation clause — "refuted if the per-frame variation is under ~2x" — is answered the other way |
| 08-15 | **grid J**, the five config-only arms of the pre-registered ten, at the stable default | control 0.9921. **derivatives=both/poloidal +0.0010**; n_pca 75 −0.0005; patience 6000 −0.0008; pyramid [1024,256] −0.0012; Thomson raw profiles −0.0012 | one positive of five, and it is a FEATURES arm — which is where the capacity result said to look. A10 screened the same configuration at 0.60/0.1 and got +0.0007 at best, below its floor: a feature that needs data to pay is exactly what a small screen refutes wrongly. Confirmation running |
| 08-15 | is the leash still wasteful now the fit is stable? | cutting `patience_steps` 18400 → 6000 costs **−0.0008** | it was wasteful at the unstable setting (60-70% of the fit ran past the best step) and it is not now: clipping lets the fit train 93000 steps and the best one arrives late. The measurement that A16 queued, answered — and the answer changed because the thing underneath it changed |
| 08-15 | **the fits are unstable at the production shares**, found while chasing a control that moved 0.0021 between two runs of the identical configuration | five repeats of one configuration give **0.9889 five times over**, best step 50000 in each — so there is no run-to-run noise at all. What differed: at step 6000 one run's training loss is **13.6** against unit-variance targets. Across every log at these shares, **15 fits of 28 blow up**, worst 67.9 | the learning rate — sqrt-scaled for batch 4096 and tuned at 0.60/0.1, a twenty-fifth of the frames — is past the edge of stability at this scale. Not noise in the measurement: a lottery inside it |
| 08-15 | **grid T**: what production should train at, paired on three salts | `grad_clip: 1.0` at the current rate **+0.0015 / +0.0007 / +0.0008, mean +0.0010, t = 3.97**, 0 blow-ups of 3, 93000 steps run. A quarter of the rate: **+0.0005 / +0.0013 / +0.0019, mean +0.0012, t = 3.04**, 0 of 3, but 170000 steps. The default blew up in 2 of 3 | **clipping accepted, default.** Same gain as the slower rate, tighter spread, half the wall clock — and it removes the instability that had been corrupting every comparison of the evening |
| 08-15 | **grid A15**, capacity and light dropout crossed: width 512/700 x depth 2/3 x dropout 0/0.02/0.1 at 0.80/1.0 | best cell +0.0012 against a selection floor of +0.0029 — but the MARGINAL means use all twelve runs: width 700-512 is **−0.0006 and −0.0011** at the two dropout levels where the design is complete, depth 3-2 is **−0.0001 and −0.0011**, dropout 0→0.02 is −0.0001 and 0.02→0.1 is **−0.0025** | **capacity refuted on both axes, and past 0.02 dropout hurts.** Reading the best of twelve would have said "+0.0012, promising"; reading the design says every axis is flat or negative |
| 08-15 | what that does to the bias-limited diagnosis | three regularisers had closed the train/val gap with validation flat, which is the signature of a model limited by bias — and MORE CAPACITY DOES NOT FIX IT | so the bias is not a capacity bias. The net can already represent more than the inputs determine, and the constraint is what a single frame's 46 numbers CONTAIN. That is an argument for the sequence model and for features, and against every further width |
| 08-15 | 700x700x700 at the default 2.8e-3 | train loss **1.44** at step 2000 — worse than predicting the mean of unit-variance targets — best at step 11650, then rising again to 0.152 by 30000. With dropout 0.02 the same architecture trains fine and scores 0.9911 | not a capacity result but a **divergence**: the rate was tuned for 512x512. Recorded as such and re-run at lower rates rather than counted as depth hurting by 0.0074 |
| 08-14 | why more frames pay, visibly | the best step moves from 21631 to 42518 on average (13350..30250 against 18800..75600) at an unchanged step budget | five times the data delays overfitting, so twice as many of the same steps stay useful. Not more optimisation — more of it that is worth doing |
| 08-14 | the EFIT frame clock, measured over 202 shots and 44729 steps | 97.86% of steps are 20 ms and every step is a multiple of it, but **77% of shots contain at least one dropped frame** (largest gap 2580 ms), and a shot is 2 to 373 frames long | settles the sequence model's design before it is written: train it at frames 1.0 only, and feed Δt in units of the base step, because 1.0 is still not a regular grid |

## The two reversals, and what they cost us

Two results measured cleanly at quality scale did not survive at production, and both for the same
reason: **the bigger model can do for itself what we were doing for it.**

- `n_pca = 30` beat 50 at quality on all three salts with t = 4.84 — and at production 50 won. With
  352 shots the tail PCA components are noise the model cannot predict and the metric over-weights
  (condition number 5266 → 2030 → 364 as n falls); with 3168 shots they are signal, and truncating
  throws it away.
- Subtracting the coil field was worth +0.0088 at quality and **+0.0007** at production. At 3168
  shots the model learns the coil field on its own, so computing it analytically saves almost
  nothing. It stays because MAST has no DIII-D coils to learn from — not because it pays here. Half
  a day of Green's functions, a lost 1/π and a sign convention, for nothing measurable on DIII-D.

The production ablations, all at salt 0 against S = 0.9809 (n_pca 50, jacobian, subtraction on):
`parseval` −0.0097, no subtraction −0.0007, n_pca 75 −0.0010, n_pca 100 0.0000. Only the first
clears the noise floor.

Hence the rule in AGENTS: quality screens, production decides.

## How good the linearisation actually is

The whole Jacobian metric assumes the seven functionals respond linearly at the error the model
actually makes. Checked by taking the model's real error direction `dc = c_pred − c_true` on 318
held-out frames, computing the true change of each functional, and comparing with `J·dc`:

| scalar | linear / actual | correlation | bias share of the miss |
|---|---|---|---|
| R_axis | 0.98 | 0.99 | 0.25 |
| Z_axis | 0.91 | 0.91 | 0.02 |
| volume | 0.96 | 0.90 | 0.09 |
| li | **1.65** | 0.92 | 0.11 |
| kappa | 1.21 | 0.77 | 0.02 |
| tri_top | 1.14 | 0.46 | 0.05 |
| tri_bot | **1.51** | 0.45 | 0.01 |

Three readings:

- **`li` is OVER-weighted, not under-weighted.** The prediction that its quadratic nature would
  show up as a systematic positive bias was wrong in emphasis: the bias exists and has the right
  sign but is 11% of the miss. What actually happens is that the Jacobian thinks `li` moves 1.65×
  more than it does, so `M ∝ JᵀJ/var` hands it too much of the loss budget. That is a better
  explanation of why `li` was the one scalar to get worse under this metric.
- **The triangularities barely linearise at all** — correlation 0.45. They read extreme points of
  the contour, which move discontinuously.
- **`R_axis` linearises almost perfectly** (0.98, corr 0.99), and it was the thing we most needed
  to fix. `Z_axis` and `volume` are fine too.

The metric pays +0.0097 at production despite being a mediocre point-predictor for three of the
seven. What matters is evidently the relative weighting of directions in aggregate, not per-frame
fidelity.

## The noise, measured properly

Three split salts times three MLP seeds, nine production runs, 2026-08-13. Seeds vary only the
optimisation, salts vary which shots are in the fold, so **the noise of a paired A/B test is the
spread across seeds inside one salt** — never the spread across salts, which measures a different
fold rather than a different answer.

| salt | seed 0 | seed 1 | seed 2 | mean | sd |
|---|---|---|---|---|---|
| 0 | 0.9809 | 0.9788 | 0.9787 | 0.9795 | 0.0012 |
| 1 | 0.9780 | 0.9774 | 0.9781 | 0.9778 | 0.0004 |
| 2 | 0.9767 | 0.9748 | 0.9762 | 0.9759 | 0.0010 |

**Pooled seed sigma = 0.0009** (6 dof). Salt-to-salt spread of the salt means: 0.0018 — twice as
large, as it should be, since the salt changes the data. Grand mean 0.9777, which is also the
correction to a headline: **0.9809 was the best of three seeds, not the typical one.**

| how a difference is measured | sd of the difference |
|---|---|
| one salt, one seed, paired | **0.0013** |
| one salt, mean of three seeds, paired | **0.0008** |
| three salts x three seeds, paired | **0.0004** |

The noise is not spread evenly. Per term: R²ψ 0.0001, `1 − D_LCFS` 0.0005, R²qb 0.0012, and
**Consistency 0.0045** — which, at weight 0.20, is essentially all of it. Per scalar:

| scalar | mean | sigma over seeds |
|---|---|---|
| Z_axis | 0.911 | **0.0234** |
| R_axis | 0.888 | **0.0177** |
| volume | 0.890 | 0.0083 |
| li | 0.896 | 0.0083 |
| kappa | 0.878 | 0.0080 |
| tri_top | 0.979 | 0.0033 |
| tri_bot | 0.976 | 0.0017 |

So a 0.01 gain on `R_axis` is noise and the same 0.01 on `tri_bot` is a clear signal. The two axis
positions are the least reproducible quantities in the whole metric — which is what you would
expect of a point defined by ∇ψ = 0, and an independent argument for anything that removes
frame-to-frame jitter.

## Where the nine minutes and the twenty gigabytes go

One production run, instrumented by `toolkit/progress.py` (every printed line carries
elapsed time, the gap since the previous line, and the resident memory of the whole process tree).

| stage | time | share |
|---|---|---|
| read 3168 training shots | 61.9 s | 12% |
| read 1056 validation shots | 19.7 s | 4% |
| coil field: fit gains and subtract | 4.5 s | 1% |
| PCA fit | 9.8 s | 2% |
| PCA transform + Jacobian probe | 18.6 s | 4% |
| **MLP fit** (1148 epochs, stopped at 1047) | **352.1 s** | **67%** |
| scoring 70 shots, three models | 58 s | 11% |

Peak memory 21.6 GiB against **1.5 GiB of data**. The flux arrays are 1.47 GiB; nearly everything
else is the reader pool's 18 workers, which never return their parquet decode buffers to the OS
and then sit idle through the six-minute fit. That is the swap we saw, and it was invisible until
the memory went into the log.

Two more numbers from the same run: the MLP's `DataLoader` costs 0.269 s per epoch where the same
batches taken by indexing cost 0.130 s, so **52% of every epoch is bookkeeping**; shuffling itself
is only 5%.

### What that bought, once fixed

| | before | after |
|---|---|---|
| whole production run | 8:45 | **4:54** |
| reading 4224 shots | 81.6 s | **4.1 s** |
| MLP fit | 352.1 s | **197.0 s** |
| peak resident memory | 21.6 GiB | **8.5 GiB** |
| S | 0.9809 | 0.9809 |

Three changes, all verified bit-identical rather than merely close: the decoded-shot cache
(`shot_cache.py`), `parallel.release()` before the fit, and taking the MLP's batches from the
`DataLoader`'s own sampler while gathering the rows in one indexing operation.

The batch change deserves its own note, because the obvious version of it is wrong. A hand-rolled
`randperm` loop agrees with `DataLoader` on the first epoch and diverges from the second on:
`DataLoader` draws a base seed from the generator every time an iterator is created, so the
permutations fall out of step. Driving `iter(loader)._sampler_iter` consumes the generator exactly
as the real thing does. The proof is not the score — that could coincide — but the **best epoch:
1047 of 1148 both before and after**, so the optimisation walked step for step.

## Why the jitter idea was wrong, which is worth more than the idea

It was ranked first: frames are predicted independently, they are 20 ms apart, the equilibrium
moves on hundreds of ms, therefore the prediction must carry high-frequency error that physics
forbids. The reasoning is sound and the fact is the opposite.

Measured on 30 held-out shots, variance-weighted over the 50 components, as
`var(lag-1 difference) / var(series)`:

| | roughness |
|---|---|
| predicted | 0.0212 |
| truth | 0.0394 |
| ratio | **0.54** |

The prediction is **twice as smooth as the ground truth**. Predicting a conditional mean from
instantaneous currents produces a smooth series by construction; it is EFIT's own reconstruction
that jitters. Independence between frames does not cause noise here, it causes over-smoothing.

The ratio is not uniform — pc0 (83% of the variance) sits at 0.49 while pc2 is 1.71 and pc6 is
3.02, so the tail really is under-smooth — but smoothing only the tail, with a Gaussian kernel in
milliseconds rather than a window in frames, buys at most 2.7% of the error measured in the loss
metric. That is ΔS ≈ +0.0004 against a resolution of 0.0013: not worth a run.

Two methodological points fell out of it:

- **Measure the error in the metric the loss uses.** Plain coefficient SSE said only one cell of
  the sweep helped; the same numbers under `M = L Lᵀ` said most of the tail cells did. 95% of plain
  SSE is pc0 and pc1, which are not where the score is lost.
- **Windows in frames are not windows in time.** 98.0% of EFIT intervals are 20 ms, but 31% of
  shots carry a gap over 100 ms and the largest is 900 ms, so a five-frame window sometimes spans
  half a second. The first version of this diagnostic had that bug; the conclusion survived it,
  because the roughness ratio compares the same index pairs on both sides.

## The one thing that worked was the one nobody had touched

Eight iterations of the improvement loop on 2026-08-13. Seven refuted, six of them for free; the
eighth was **widening the MLP from 256x256 to 512x512**, and it gained +0.0022 of S over three
salts with all three signs agreeing — more than every physics change of the day put together.

| salt | 256x256 | 512x512 |
|---|---|---|
| 0 | 0.9809 | **0.9841** |
| 1 | 0.9780 | **0.9801** |
| 2 | 0.9767 | **0.9780** |

Every term moved, and Consistency carried it: 0.9283 → 0.9414 on salt 0. The fit did not even cost
proportionally more — 262.9 s against 197.0 — because the wider net stopped at epoch 548 instead
of 1047.

Two things worth keeping from that. **21 → 256 → 256 → 52 had stood since the starter kit and had
never been swept once**, while a day went into Green's functions and quadratic forms. And the
"bigger model does it for itself" trap that killed two earlier results does not apply here,
because this *is* the bigger model — which is exactly why it was the safest capacity bet on the
list.

## Thomson, and the argument that nearly buried it

Iteration 10 added eleven columns per frame from the Thomson profiles and gained **+0.0039** of S
over three salts, every sign agreeing — the largest single change this fork has measured. It
landed where the physics said it would: `R2_qb` moved +0.0064, and betaN is a pressure ratio by
definition. `ridge` moved 0.7022 → 0.7562, which matters because a linear model cannot invent a
correlation: the information is there and it is linearly available.

**The argument that ranked it low was wrong, and the mistake is reusable.** Thomson sat near the
bottom of the list because a nearest-neighbour test had shown that the coil currents alone
determine betaN to R² 0.997 while the model reached 0.981 — "features are not the binding
constraint, the regression is". That inference does not follow. A nearest-neighbour ceiling says
what a PERFECT learner could extract from the currents; it says nothing about how HARD the
extraction is for a real one. Supplying a quantity that is formally redundant still helps when it
makes the function easier to represent — and betaN is nearly linear in pressure, while recovering
pressure from twenty-one coil currents is not.

Read the ceiling as "these features are in principle sufficient", never as "new features cannot
help". The same faulty step was applied to every feature idea in the list.

Three data facts the implementation had to be built around, none of them visible from the column
names:

- **The core system is a VERTICAL chord.** All 44 core channels sit at R = 1.940 and differ in Z.
  The radial derivative that Grad-Shafranov asks for exists only on the EDGE system — 10 channels
  at Z ≈ −0.06 spanning R = 1.68..2.06 — which is also exactly the pedestal region.
- **Missing samples are exact ZEROS, not nan.** Inside the EFIT window 14.1% of core values are
  zero; outside it, where there is no plasma, nearly all are. Every reduction runs on a mask.
- **Channels are not stored in positional order**, so any difference along the chord has to sort
  by coordinate first.

And one whole shot — `d3d_shot_5a79f2123a` — has Thomson that never fired: 1884 core samples and
423 edge samples with zero live channels in every one. It was invisible in samples of 40 and 120
shots and appeared at 3168. Such shots contribute zeros plus a **validity flag column**, so the
model is told the measurement is absent rather than handed a fabricated one, and the rate is
printed: 0.7% of production training frames.

## The summaries are a good compression, not a lossy one

The group sweep showed temperature summaries worth nothing and pressure carrying everything, which
suggested the obvious next step: the summaries throw away the SHAPE, and Grad-Shafranov reads
p(psi) rather than its peak. So the raw pressure profile went in, resampled onto 16 fixed points
along each chord — fixed points rather than channels, because channels die and revive and a column
tied to one would mean a different place in different frames.

| what the model saw | S | R²qb | Consistency |
|---|---|---|---|
| three summaries per system | **0.9914** | 0.9937 | **0.9685** |
| summaries + 16-point profile | 0.9901 | 0.9922 | 0.9638 |
| the 16-point profile alone | 0.9900 | **0.9939** | 0.9615 |

Raw alone equals raw+summaries, so the summaries carry nothing the profile does not. And the
summaries still WIN by 0.0014. The compression is not lossy in any way that matters — it is
better than the thing it compresses, because an integral averages the noise of a 44-channel chord
while a single profile point carries all of it.

One asymmetry worth keeping: the raw profile is the best of the three on `R2_qb` (0.9939) and the
worst on Consistency (0.9615). Pointwise pressure helps the pressure scalars; integrated pressure
helps the geometry.

## The leaderboard, and how the two estimates of selection bias compared

Submitted 2026-08-14: **0.9896, 4th place**, against 0.9764 for the previous submission.

| | local, salt 0 | leaderboard |
|---|---|---|
| previous submission | 0.9769 | 0.9764 |
| this one | 0.9914 | **0.9896** |
| gain | +0.0145 | **+0.0132** |

**91% of the local gain transferred.** The two-fresh-salt estimate below said 58% would — it was
too pessimistic, and by a lot. Both numbers are worth keeping, because the disagreement is the
lesson:

- Salts 3 and 4 are **70 scored shots each**. A paired difference between two configurations that
  differ in four ways carries fold-specific variance far beyond the 0.0009 seed sigma, and two
  such folds cannot pin a number to better than a few thousandths.
- The leaderboard is **874 shots** — twelve times the fold — and it is the only estimate here with
  no selection history at all.

So the honest reading is that the held-out-salt check correctly showed selection bias EXISTS and
correctly refused to endorse the headline, but its magnitude (+0.0067) was itself a noisy estimate.
The gap between local and leaderboard is 0.0018 this time against 0.0005 for the previous
submission — small, and consistent with a modest amount of fold-tuning rather than a large one.

## What seventeen rounds of choosing on the same folds actually cost

The loop selected against salts 0, 1 and 2 — the same 210 held-out shots — seventeen times. Salts
3 and 4 were kept untouched from the start for exactly one purpose: to price that.

| salt | this morning's recipe | everything the loop accepted | gain |
|---|---|---|---|
| 0 | 0.9809 | 0.9914 | +0.0105 |
| 1 | 0.9780 | 0.9889 | +0.0109 |
| 2 | 0.9767 | 0.9898 | +0.0131 |
| **3** | **0.9789** | **0.9866** | **+0.0077** |
| **4** | **0.9843** | **0.9900** | **+0.0057** |

**Selected folds +0.0115. Fresh folds +0.0067.** The difference, +0.0048, is what seventeen rounds
of looking at the same tail bought us in self-deception — about 40% of the headline.

Both halves of that matter. The gain is REAL: +0.0067 on folds that never took part in a decision
is roughly seven sigma against a paired sd of ~0.0009. And the headline was INFLATED: any
expectation for the leaderboard should be built from +0.0067 over the last submission, not from
the local 0.9914.

Two practical notes. Salt 4 is an easy fold — the old recipe scores 0.9843 there against
0.9767–0.9809 on salts 0–2 — which is why absolute scores are never compared across salts, only
paired differences within one. And the old recipe could not be run at all until three keys
(`derivatives`, `derivative_signals`, `thomson`) were added to it in their off positions: the
config validator refuses a file with missing keys rather than assuming defaults, so a
configuration from before those knobs existed is not silently runnable.

## Averaging seeds is worth far more than the seed spread suggests

Four MLPs differing only in their seed, averaged: **+0.0042** of S over the mean of those same four,
on three salts with every sign agreeing. It is the largest single gain in this fork.

| salt | seed mean | ensemble | seeds ranged over |
|---|---|---|---|
| 0 | 0.9868 | **0.9902** | 0.9856–0.9874 |
| 1 | 0.9819 | **0.9860** | 0.9808–0.9836 |
| 2 | 0.9836 | **0.9885** | 0.9823–0.9858 |

**The obvious estimate is wrong by about sevenfold, and the reason is worth keeping.** Reasoning
from the spread of the SCORE across seeds — sigma 0.0013, so averaging four should buy about half
a sigma — predicts +0.0005. What averaging actually reduces is the prediction ERROR, and the score
is a nonlinear function of that: four independent nets cut the independent component of the
squared error by up to 4x, and R² follows directly. Consistency shows it plainly — 0.9640 for the
ensemble against 0.9456–0.9546 for its members, a jump ten times larger than averaging their
scores would give.

Two consequences for how everything after this is measured:

- **Sigma is 0.0013, not 0.0009.** Re-measured on the current recipe it is 0.00085, 0.00132 and
  0.00155 on salts 0, 1 and 2; the morning's figure happened to land on the quietest one.
- **But the production model is now itself an average of four**, whose own seed-to-seed sigma is
  half that, ~0.00065. Comparing configurations by the ensemble rather than by a single seed makes
  every future measurement twice as precise — at four times the cost per run.

## Importance is not ablation, measured the hard way

Permutation importance over every input column, on held-out shots, in the loss metric. The coil
currents dominate at 1200–4300%. Within the Thomson block:

| column | error increase when shuffled |
|---|---|
| `th_edge_int_p` | **253%** |
| `th_core_int_p` | **244%** |
| `th_core_slope_p` | 164% |
| … | |
| `th_core_at_p` | 2.3% |
| `th_edge_at_te` | 1.8% |
| `th_core_at_te` | **0.1%** |
| `th_valid` | **0.0%** |

It agreed with the group ablation on the big thing — pressure carries the Thomson gain, and `p'`
ranks third — and then **it was wrong about what to remove**. Dropping the four near-zero position
columns cost −0.0012. Permutation importance measures what a FITTED model leans on, not how much
information a column holds: correlated columns mask each other and all of them look free.

`th_valid` measures exactly zero and is kept deliberately. It insures the 0.7% of shots whose
Thomson never fired, and the held-out fold evidently contains none of them — a rare-event feature
is invisible to this method by construction.

One surprise worth keeping: **`bcoil` scores 91%**. It is toroidal and drives no poloidal flux,
which is why it has no rectangle in the coil basis and was excluded from the derivative features.
The model leans on it anyway, most plausibly because it is nearly constant within a discharge and
so identifies the scenario — one of the very few per-shot signals available, which is interesting
next to the per-shot constant below.

## The error is a per-shot bias, and that changes what is worth trying

Five iterations of the improvement loop on 2026-08-13 spent no production runs and refuted four
ideas, but the fifth found the shape of the problem, which is worth more than any of them.

**It is concentrated.** Of the squared error over the 70 scored shots, in the loss metric:

| | share of the error |
|---|---|
| worst 1 shot | **23.2%** |
| worst 2 | 32.8% |
| worst 10 (14% of the fold) | 54.1% |

**It is not extrapolation.** The two worst shots have `|z|` maxima of 2.8 and 1.8 against a fold
median of 2.3, and the four most extrapolated shots (`|z| = 4`) are not among the worst. Their
inputs are ordinary.

**It is not a moment in the discharge.** The worst 5% of frames carry 8% of a bad shot's error —
near-uniform. The model is not failing at a transition; it is wrong evenly, all shot long.

**It is a constant.** Splitting each shot's coefficient error into its mean over frames and the
rest: **45.9% of the fold's error is the per-shot mean**, and 85.4% of the worst shot's. Remove
every shot's constant and the fold's metric R² goes 0.9594 → 0.9780. The worst shot goes from
0.7435 to 0.9626 — it is not a hard shot, it is a displaced one.

**And that constant is a shaped map, not a gauge.** Only 1.7% of it lies along "add a constant to
every pixel"; 98.3% is spatially structured. So it is not `ψ`'s additive freedom — it is a real
poloidal field, present for the whole discharge, that the instantaneous currents do not determine.

That description matches vessel eddy currents: they are driven by the *history* of the coil
currents, they persist on the vessel's L/R time, and they are structurally absent from the shipped
signals — `coil_field` cannot compute them because the vessel geometry is not in the dataset.

**And that explanation was then tested, and failed.** 252 per-shot summaries of the current history
— means, spreads, endpoints, mean `|dI/dt|`, and one-pole high passes at τ = 5, 20 and 50 ms —
regressed on the constant, fitted on 400 validation shots and evaluated on the scored fold, explain
**0.008** of it against 0.003 for predicting the fitting shots' mean. The ridge path is monotone
towards that constant model, so this is not a regression that lacked capacity: there is no signal
in the history to find. Whatever the shaped per-shot field is, the coil currents' past does not
say.

The honest limits of that test: 252 features on 400 shots overfits badly at small α (−1.7), and a
real eddy model would need the vessel geometry the dataset does not ship. But it is enough to stop
the derivative features from being built on the strength of a story. What remains possible is that
the constant belongs to the *labels* — EFIT is itself a reconstruction with its own priors and
per-shot settings — in which case it is irreducible from these inputs by anyone.

It also corrects an entry from 08-11. "The error is a per-shot flux offset" was recorded as refuted
at 21%, but that measured one additive scalar on ψ; this measures a constant vector in the
50-dimensional coefficient space, a strictly larger family, and at production it is 46%.

## The `make quality` sweep, pre-registered 2026-08-14

Written down BEFORE the first run, because the alternative is not a measurement. The screen is
`make quality` — 4225 shots, one net, salt 0 — at 5.5 min a run.

**Why pre-registration and not just a bigger sweep.** The single-net sigma is **0.0013** (08-14,
pooled over three salts). The expected maximum of N draws from that noise is about 2.25 sigma at
N = 50, i.e. **+0.0029** of pure selection — larger than any change this fork has ever accepted
(+0.0022, +0.0039, +0.0042, +0.0018). It-18 measured exactly this failure on this project: the
accepted stack showed +0.0115 on the three salts it was selected on and +0.0077 / +0.0057 on two
salts it had never seen, so about 40% was selection bias.

**Thresholds, fixed in advance:**

- a configuration becomes a *finalist* if it beats the salt-0 baseline by more than **0.0013** (1
  sigma) on the screen;
- a finalist is *accepted* only if, re-measured on salts **3 and 4** which nothing selected
  against, the mean gain exceeds **0.0013** AND both salts agree in sign;
- whatever survives gets one four-net production run before it goes near a submission;
- the screen number and the confirmation number are always reported together. The screen number
  alone is not a result.

**Grid A — batch norm and the learning rate, JOINTLY.** Batch norm changes the convergence enough
that measuring it at the rate tuned without it would test the rate, not the normalisation. Nine
runs: `batch_norm` off/on crossed with `learning_rate` 3e-4 / 1e-3 / 3e-3 / 1e-2 / 3e-2, plus the
cosine schedule on the best of each arm.

**Grid B — batch size and the rate**, for both goals at once: the loop is launch-bound, so epoch
time tracks the number of batches (183 -> 0.307 s, 45 -> 0.080, 11 -> 0.020). `patience` is
converted to a constant number of STEPS rather than left at 100 epochs, or the comparison measures
the leash: at batch 4096 an epoch is 22 steps against 183 now.

**Grid C — the cheap regularisers.** `weight_decay` is still 0.0 and the measured train/val gap on
this screen is **3.4x** (train 0.0192, val 0.0650 at epoch 625), so there is something to
regularise. `dropout`, and `gelu`/`silu` for the hidden layers.

**Declared in advance as a likely disappointment**, so it is not explained away later: the screen
fits ONE net and production averages FOUR. Dropout and batch norm work partly by reducing variance,
which is what seed-averaging already does, so a gain here can shrink in the production ensemble.

## Why batch norm gave nothing, and what it predicts — 2026-08-14

Batch norm was measured at four learning rates spanning 30x (1e-3, 3e-3, 1e-2, 3e-2) because it
changes convergence enough that testing it at the rate tuned without it would test the rate. The
curve is unimodal with its peak at 3e-3, and that peak is **exactly the baseline**: 0.9873 against
0.9873, for 30-38% more time per epoch.

The train/val curves say why, and it is not "batch norm did not work":

| at epoch 600 | train | val | gap |
|---|---|---|---|
| baseline | 0.0192 | 0.0650 | **3.4x** |
| BN, lr 1e-3 | 0.0375 | 0.0693 | **1.9x** |
| BN, lr 3e-3 | 0.0343 | 0.0662 | **1.9x** |

**It regularised hard and that bought nothing.** The gap closed from the wrong side: validation did
not move (0.0650 -> 0.0662) while the training loss got 1.8x worse. So the 3.4x gap was not the
constraint — the model is limited by bias, not by variance, which agrees with it-11 (capacity
exhausted, the constraint is the data) and with the nearest-neighbour ceiling of R² 0.97-0.997
against the 0.90-0.93 the model reaches.

Batch norm's other textbook effects DID show up and also failed to matter: the epoch-0 loss fell
1.83 -> 0.79, so it conditions the start; and it did not converge in fewer epochs (626 baseline
against 604 and 652). On two hidden layers the gradient path is short enough to need neither, and
the pipeline already standardises the features and scales the targets to unit variance, so both
ends of the net were well conditioned before batch norm saw them.

**Pre-registered prediction, written before grid C ran:** dropout and `weight_decay` should do the
same thing — close the train/val gap without moving validation. If either moves validation, this
explanation is wrong and has to be replaced rather than patched.

## What the thirty-eight-run architecture sweep settled — 2026-08-14

Grids A, B and C screened thirty configurations on salt 0; grid D re-measured the two survivors
plus the speed composition on salts 3 and 4, which the selection never saw.

**Nothing improves S.** Batch norm, dropout, weight decay, the learning rate, its schedule, four
activations, depth, and batch size from 256 to 4096 — the best screen result was +0.0013, and it
did not survive contact with an unseen fold. The single most useful number in the whole sweep is
that +0.0013 was already SMALLER than the +0.0029 that selection alone produces over that many
arms, so the screen was telling the truth before the confirmation ran.

**What did come out of it is 2.30x of speed at parity**, confirmed on two unseen salts to within
0.02x of each other: batch 4096 with a sqrt-scaled rate, plus gelu. And a diagnosis, from three
independent regularisers agreeing: the model is limited by **bias, not variance**. Every one of
them closed the train/val gap — 3.39x to 1.90x to 1.19x to 0.56x — and validation moved the wrong
way or not at all.

**Where that points.** Not at the architecture, which is now well covered: at the inputs, where the
nearest-neighbour ceiling is R² 0.97-0.997 against the 0.90-0.93 the model reaches, or at what the
loss asks for (A13). The one loose thread inside the architecture is A15's empty cell — capacity
and light regularisation together, which neither it-11 nor grid C tested.

## Longer averaging is WORSE, and it reverses the reasoning the sweep was built on — 2026-08-20

**Completed at 16 salts: `d9999` (0.9999, a 10000-step window) against the accepted `x_ema`
(0.999, 1000 steps) is −0.00087, SE 0.00015, t = −5.6, 1 of 16 signs positive.** Refuted, and it
gives back almost exactly what ema won.

**Against the ORIGINAL no-ema baseline it is +0.00014 — and 11 of the 16 salts are EXACTLY zero.**
Four decimals, eleven times, is not noise; it says the fit is landing on the same model the
baseline lands on. The counter explains it: the averaged weights win **258 evaluations a run at
0.9999 against 1589 at 0.999**, six times less often, so the globally best evaluation is nearly
always a live-weight one — and a run with the same seed on the same data then finds exactly the
point the baseline finds.

That yields a quantitative handle worth keeping:

| decay | window | averaged weights win | gain over no ema |
|---|---|---|---|
| 0.999 | 1000 steps | **1589 / run** | **+0.00101** |
| 0.9999 | 10000 steps | 258 / run | +0.00014 |

The ratios are 6.2x and 7.2x — **the gain is close to proportional to how often the average beats
the live weights**, which makes `_ema_wins` a cheap leading indicator for the remaining arms. It
also predicts where the optimum cannot be: as the window shrinks toward zero the average approaches
the live weights, so it will win about half the time on noise alone while the margin it wins by
goes to zero. Wins alone are therefore not the objective — `d998` (500 steps) and `d995` (200) test
the near side, and a rise in wins there without a rise in score would be exactly that degeneracy.

**The mechanism, and it is the obvious one in hindsight.** A run takes 56950 to 149600 steps. A
1000-step window averages only the converged noise ball; a 10000-step window reaches back into the
part of the trajectory where the model was still visibly worse and drags the average toward it.
Weight averaging works because the last stretch is stationary, and 10000 steps is not.

**This reverses the argument the program was designed on.** Entry Y in `ideas.md` reasoned that
0.999 averages "only the last 1.4% of the trajectory" where classical weight averaging takes a long
tail, so the interesting direction was LONGER. The data says the opposite: the optimum is at 0.999
or **shorter**. The same reasoning had also thrown out short decays as "the live weights with extra
arithmetic" — symmetrically wrong, and `d998` (500 steps) and `d995` (200 steps) were queued as
soon as this read came in.

It is consistent with the earlier null result rather than contradicting it: the gain does not scale
with how much of the trajectory is averaged, and past a point more of it actively hurts.

**`d9995` completed flat: +0.00000, SE 0.00003, 5 signs of 14.** Per salt it is almost all exact
zeros and single units of 1e-4, and the standard error is a tenth of a typical screen arm's — the
signature of two settings producing very nearly the SAME model. A 2000-step window and a 1000-step
one land in the same place.

So the decay curve is now characterised end to end:

| averaging window | against the accepted 0.999 |
|---|---|
| 2000 steps (0.9995) | **+0.00000** |
| 10000 steps (0.9999) | −0.00087 |

A plateau from 1000 to 2000 steps, then a cliff. The accepted value sits on the plateau and the
sweep confirms there is nothing to win by moving along it — which, together with the production
confirmation measuring ema itself at zero, closes the whole family.

(Two of the sixteen runs were lost to the OOM described below; at this effect size they cannot
change a mean of exactly zero.)

## What the night's accepted work should be worth on the board

One measured number and three ratios, each from somewhere else, written out so the estimate can be
attacked at whichever link is weakest:

| step | value | where it comes from |
|---|---|---|
| `ema` on the one-net screen | **+0.00101** | measured here, 16 paired salts, t = 7.2 |
| x level-shift share 0.68 | +0.00069 | the regression of gain on baseline, measured here |
| x member dilution 0.933 | +0.00064 | 12 of the shipped 13 members are MLPs; `seq` has no `ema_decay` |
| x local→board 0.91 | **+0.00058** | the fork's own median transfer over three submissions |
| | **board 0.9938 → 0.9944** | |

**Only the first line is a measurement of this change.** The second is the one the paired production
confirmation exists to replace with a number: its pre-registered prediction is +0.0007 on a 4-MLP
ensemble. If it lands there, this estimate holds; if it lands near +0.0002, the chain collapses to
+0.0002 and the level-shift decomposition was wrong.

The third line is arithmetic on the shipped configuration and the least uncertain. The fourth has
read 91% / 116% / 43% over three submissions, so it is a median over a wide spread and the honest
band around the total is roughly +0.0003 to +0.0008.

## The 16-salt program — the instrument first, then the arms

**A second hypothesis about the instrument, and this one questions the choice of instrument
itself.** Paired standard deviation, pooled over every arm run on the screen against the single
production measurement:

| arm | paired sd |
|---|---|
| `jac` | 0.00112 |
| `bnd` | 0.00109 |
| `d9999` | 0.00062 |
| `ema` | 0.00056 |
| **screen, pooled** | **0.00085** |
| **production, 4 MLPs (3 pairs)** | **0.00017** |

A production run costs 36 minutes against the screen's 10, so at equal wall-clock the screen's
standard error is **2.63x worse**: 0.00035 an hour against 0.00013.

**A side observation that checks out internally:** the two arms containing weight averaging (`ema`,
`d9999`) have half the paired sd of the two that do not (`jac`, `bnd`). That is exactly what EMA
does — it stabilises where the fit lands — so an arm built on it is measured more precisely than
one that is not, for the same reason it scores better.

If it survives six pairs, the conclusion is uncomfortable and worth acting on: **the whole 16-salt
programme should have been run at production scale.** The screen was chosen as the cheap
instrument, and per run it is; per unit of resolution it is not, and it measures a configuration
that does not ship.

**Caveat, stated because the number is doing a lot of work:** the production sd rests on THREE
pairs, which is a very unstable estimate. Re-check at six before treating it as a rule.

**And a correction to the conclusion, before it drives the wrong action.** "2.63x worse per hour"
is the comparison for a SINGLE measurement made with one instrument. The design actually in use is
two-stage — a cheap screen to reject, a production run to confirm — and there the screen's job is
the cost of a rejection, not its resolution: 2.7 hours an arm against 7.2. On that footing the
screen is the right first filter and the remaining arms stay on it.

The screen's real defect is the other one, named this morning and unchanged: it over-states
**variance-type** changes seven- to ten-fold, because it fits ONE net where production averages
four. For changes that alter what every model does — a loss, a stopping rule, a feature — no
discount applies. `huber` and `stopcomp` are of that kind, which is why they stay on the screen
while the decay arms were dropped.

**A hypothesis about the instrument itself, to be settled by the confirmation now running.** The
quality screen fits ONE net. Any change that works by removing seed noise will therefore look far
larger there than in a four-seed production ensemble, which has already removed three quarters of
that noise. Two data points point the same way:

| change | measured alone / on the screen | inside the production ensemble | ratio |
|---|---|---|---|
| three sequence seeds (A25) | +0.0016 to +0.0023 | **+0.0002** | ~0.1 |
| `ema` | +0.00101 (16/16 salts) | +0.00015 at 2 of 6 pairs | ~0.15 so far |

If the six pairs hold, the rule is: **the screen over-states a variance-reduction change by roughly
seven times**, and a screen result of that kind must be discounted before it is believed.

Crucially the discount should NOT apply to changes that alter what every model does — a feature, a
loss, a stopping rule. Those are not absorbed by averaging, because averaging does not remove a
shared error. That distinction decides how to read the three arms still queued: `huber`,
`stopcomp` and `aux` all change every member, so their screen numbers should transfer far better
than ema's did. It is worth stating before their results arrive rather than after.

**The instrument checks out.** The gate was pre-registered in `ideas.md` before the first run, on
a predicted standard error of 0.0013/sqrt(16) = **0.00033**. The first completed arm measured
**0.00028**. So sixteen paired salts really do resolve what the arithmetic said they would, and
`my_experiments/paired.py` is the tool that reads them — matched by salt, never by position.

The baseline arm's spread across salts is **0.00165** of S (0.9877 to 0.9934), five times the
paired standard error. That number is the case for pairing in one line: comparing absolute scores
between salts would be reading a quantity five times larger than any effect being chased.

| arm | change | mean paired ΔS | SE | t | signs | verdict |
|---|---|---|---|---|---|---|
| `jac` | `loss.jacobian_frames` 300 → 1200 | +0.00044 | 0.00028 | 1.6 | 8/16 | **refuted at its gate** |
| `ema` | `models.mlp.ema_decay` 0 → 0.999 | **+0.00101** | 0.00014 | **7.2** | **16/16** | **ACCEPTED** |
| `aux` | `loss.aux_scalars = kappa, li` | — | — | — | 1/16 ran | **NOT MEASURED — see below** |
| `bnd` | `loss.boundary = true` | +0.00040 | 0.00029 | 1.4 | 8/14 | **refuted at its gate** |
| `huber` | `models.mlp.loss = huber` | **+0.00088** | 0.00037 | 2.4 | **13/16** | **ACCEPTED** |
| `hd1` | `huber_delta` 2.0 → 1.0 | +0.00030 | 0.00028 | 1.1 | 10/16 | **refuted at its gate** |
| `hd4` | `huber_delta` 2.0 → 4.0 | +0.00052 | 0.00033 | 1.6 | 10/16 | **refuted at its gate** |
| `emacos` | cosine on top of `ema` | −0.00015 | 0.00013 | −1.1 | 4/16 | **refuted at its gate** |
| `stopcomp` | stop on the composite, monitor 150 → 600 | +0.00009 | 0.00023 | 0.4 | 3/15 | **refuted at its gate** |

`jac` tested whether the metric-aware loss is limited by the NOISE in its own weights — they are
estimated on 300 frames, and a noisy estimate is a noisy loss. Quadrupling the estimate is worth
+0.00044, below both the +0.00065 gate and the +0.00055 that selection over four arms produces on
its own. Five salts went negative and three tied. Not nothing, and not enough: at this effect size
settling it at 2 sigma would need about 40 salts, seven hours, for something worth under a
thousandth — which is a worse trade than any open entry in `ideas.md`.

### `stopcomp` — the composite monitor's noise was not the problem

**+0.00009, SE 0.00023, 3 signs of 15.** Flat.

The retest had a specific diagnosis behind it rather than "more salts": A13 was refuted at the old
0.0013 resolution, and the composite monitor is ESTIMATED on 150 validation frames while 94% of the
instrument's variance sits in Consistency — the very term that monitor reads. A noisy criterion
stops in the wrong place, so quadrupling the monitor should have rescued it.

It did not move at all. **So A13 stands on its merits, not for want of precision**: stopping on the
composite is no better than stopping on the validation loss, and the monitor's sample size is not
what was holding it back.

### `emacos` — cosine adds nothing to a run that already averages its weights

Against the accepted `ema`: **−0.00015, 4 signs of 16.** Against the plain baseline: +0.00086 with
14 of 16 — which is simply ema's own effect, unchanged.

That is the exact question the arm was designed to ask. The cosine schedule was the only knob ever
to screen positive (+0.0009 / +0.0006 on 2026-08-14) and its refutation stood on two confirmation
salts, so it deserved a retest; but the entry framed it as "the question is no longer whether
cosine helps, it is whether cosine helps a run that ALREADY averages its weights". The answer is
no. Both mechanisms shrink the same thing — where the trajectory ends up — and having one makes the
other redundant.

Since ema itself measures zero in production, so does this. **The weight-averaging and schedule
family is now closed end to end**: `ema` (zero in production), `d9995` (flat), `d9999` (−0.00087),
`emacos` (−0.00015), with `d998`/`d995`/`d99995` dropped on the strength of the shape.

### `huber` — accepted, and it refutes a planned experiment on the way

**+0.00088, SE 0.00037, t = 2.4, 13 of 16 salts positive** — clears a gate of +0.00065 and 12
signs. (The first reading, on the 15 salts that had finished, gave +0.00095; the sixteenth changed
nothing.) The gain is Consistency again: +0.00077 of the +0.00095, with R²qb +0.00012 and D_LCFS
marginally worse.

**It belongs to the class the screen does NOT over-state.** Huber changes what every model does
rather than removing seed noise, so — unlike `ema`, which went from +0.00101 on the screen to zero
in production — its number should survive an ensemble. That distinction was written down before
either result arrived; this is its first supporting case, and the production confirmation is what
would settle it.

**And it answers the mirror question the wrong way round for my own design.** Huber is quadratic
near zero and linear past `huber_delta`, so it DOWN-weights exactly the large-residual frames — the
expensive 5% that carry 46% of the cost. The per-frame weight designed earlier the same day
proposed to UP-weight them, on the reasoning that the cost concentrates there. Huber helping by a
full thousandth says the opposite: those frames should have LESS influence on the gradient, not
more.

That is not the same as saying they are noise — the label-noise hypothesis was refuted separately
(the truth's d2/d1 is lower in the tail, not higher). The likelier reading is that an outlier
gradient costs the rest of the fit more than the frame it comes from is worth, which is precisely
Huber's stated rationale in `models.py`.

**And the threshold is a real optimum, confirmed on both sides.** `huber_delta` had sat at 2.0
since the loss was written and had never been swept:

| delta | vs the baseline | vs delta 2.0 |
|---|---|---|
| 1.0, harsher | +0.00030 | **−0.00058** |
| **2.0, shipped** | **+0.00088** | — |
| 4.0, gentler | +0.00052 | **−0.00036** |

Unimodal, peaking exactly where the default already sat, with both neighbours worse by comparable
amounts.

**The control did its job.** At delta 4.0 Huber is nearly MSE — few frames reach the linear
region — and the gain halves. So the gain IS the outlier clipping and not some other property of
the curve. At 1.0 it halves too: clip too much and real signal goes with the outliers. Both sides
agree with the mechanism, which is worth more than either number.

**So the disagreement-weight arm should not be built as specified.** It was priced at +0.00025 to
+0.00080 on an oracle that assumed fixing the hard frames; the arm that does the opposite has just
measured +0.00095. The honest next step is the reverse experiment — weight hard frames DOWN — for
which Huber is already the implementation, and `huber_delta` (2.0) is its unswept knob.

### `bnd` — the boundary term does not reach D_LCFS, and does not clear the gate

**+0.00040 against a +0.00065 gate, 8 of 14 salts positive.** Refuted, and it fails on the mean
whatever the sign count, so the two missing salts do not change the verdict.

**Where it moves, and it is not where it was aimed.** The term exists because the Jacobian loss
structurally cannot contain D_LCFS — a distance is zero at the ground truth, so a central
difference sees no derivative. Decomposed:

| term | of S | signs |
|---|---|---|
| 1 − D_LCFS | **+0.00000** | 5/14 |
| Consistency | +0.00042 | 10/14 |
| R²qb | −0.00004 | 3/14 |
| R²ψ | +0.00002 | 2/14 |

It moves Consistency and leaves D_LCFS exactly where it found it. The early n=6 reading said the
same thing and it held to the end. So displacing a level set reaches the functionals that READ the
boundary well before it reaches a symmetric Hausdorff maximum over that boundary — the term's own
docstring calls itself "a surrogate for that term, not that term", and the measurement agrees more
literally than the docstring intended.

### Two runs were killed by my own analysis — rc=137 — 2026-08-20

`x_bnd_s10` and `x_bnd_s12` died at 09:26 and 09:41 with **rc=137, SIGKILL**, logs ending cleanly
after PCA at 56 GB with no exception. That is the OOM killer.

**The cause was mine.** Two training lanes peak at 56 GB each, which is 112 GB of a 183 GB cgroup
and was measured live at 103 GB — sound. But between 09:16 and 09:42 I was also running a
five-model rescore over 300 validation shots as a third consumer, and both deaths fall inside that
window. The two-lane decision was right; running heavy analysis on top of it was not.

**The rule that follows:** rescoring and diagnostics are not free while two lanes are training. Run
them against ONE lane, or wait. And `paired.py` learned something too — a killed run leaves
`FAILED` in `S` but EMPTY strings in every other column, so a tool that only skips `FAILED` still
crashes the moment it is asked for a per-term breakdown.

### `aux` did not fail its gate — it never reached one

Fifteen of sixteen runs died before fitting, on

    loss.aux_scalars is 'kappa,li' but the training functionals are not cached for this split

A9 teaches the seven scored functionals as auxiliary outputs, and they have to be precomputed per
frame — one `extract_lcfs` and seven derivations, about 31 ms a frame, **13 core-hours for a
training split**. That cache is keyed by the exact file list, so it exists for salt 0 and for no
other salt. Building the other fifteen is 8 hours at 24 jobs before a single fit starts, then 2.8
more to run them.

**Not built, and the reasoning is a trade rather than a verdict.** A9 was already measured flat
once; salt 0 alone came in at −0.0010 against its pair, which is one salt and means nothing on its
own; and the eleven hours it would cost sit against a queue holding the `ema_decay` sweep, which
has a confirmed mechanism behind it and costs 2.7 hours an arm. **Record this as UNRESOLVED, not
as refuted** — nothing here is evidence about A9 either way, and reading the missing rows as a
negative result would be exactly the mistake the gate exists to prevent.

The tool learned something too: `paired.py` now DROPS a `FAILED` row rather than parsing it, so a
dead run is missing data instead of a score of zero.

### `ema` — A20, and the first accepted change since 2026-08-18

**+0.00101 with sixteen signs of sixteen**, the smallest salt at +0.0004 and the largest at
+0.0025. Against 0.0045 of reachable headroom that is **22% of what is left in Challenge 1**, from
one line of configuration that costs nothing at inference.

**The mechanism is not blind averaging, and the code is what says so.** The EMA shadow weights do
not replace the live ones; they are evaluated alongside them at every evaluation and the better of
the two is kept, with `_ema_wins` counting how often the average won. It won **965 to 3019
evaluations per run** — systematically, not occasionally. The averaged trajectory is simply a
better point than the last one most of the time, which is the classic weight-averaging result and
not a quirk of this fit. A second, free consequence: the best step arrives earlier, **65669 against
76884** on the baseline arm, so the accepted setting is also about 15% shorter.

The selection-on-validation worry does not apply: the best-of-two choice is made on validation
frames, and the +0.00101 is measured on the 70 HELD-OUT shots, which that choice never saw.

**How long a tail is 0.999 actually averaging? A thousand steps of seventy thousand.** The EMA
update sits inside the per-STEP loop, immediately after `opt.step()` — not in the evaluation block
— so the averaging window is `1 / (1 - decay)` **steps**: 1000 at the accepted 0.999, against a run
that takes 66000 to 77000 of them. The accepted setting therefore averages the last **1.4%** of the
trajectory, where the classic weight-averaging result takes a far longer tail. That reframes the
sweep that follows: 0.99 would be a 100-step window, essentially the live weights, and the
interesting direction is LONGER, not shorter.

**Where the gain is, and it is the term C2 pointed at.** Decomposed across the four scored terms
(`paired.py --terms`, in units of the contribution to S):

| term | raw Δ | of S | signs |
|---|---|---|---|
| R²ψ | +0.00007 | +0.00004 | 5/16 |
| R²qb | +0.00050 | +0.00008 | 12/16 |
| 1 − D_LCFS | +0.00078 | **−0.00008** | 16/16 worse |
| **Consistency** | **+0.00412** | **+0.00082** | **16/16** |

**Four fifths of it is Consistency**, which is where C2 measured 76% of the reachable headroom to
be. D_LCFS pays a small, perfectly consistent price — 16 of 16 salts move the wrong way — which is
what a smoother map should do to a Hausdorff MAXIMUM while helping every mean-square functional.
The story is coherent from mechanism to term, which is worth more than the number.

**All seven scored functionals move, and the weakest move most.** Paired per salt, read out of the
`[mlp]` block of each log (not ridge, not the ensemble — they print the same table):

| scalar | baseline | ema | Δ | SE | signs | of S |
|---|---|---|---|---|---|---|
| li | 0.9350 | 0.9412 | **+0.0062** | 0.0031 | 12/16 | +0.00018 |
| Z_axis | 0.9686 | 0.9750 | **+0.0064** | 0.0021 | 14/16 | +0.00018 |
| kappa | 0.9400 | 0.9455 | **+0.0055** | 0.0012 | 15/16 | +0.00016 |
| R_axis | 0.9675 | 0.9717 | +0.0042 | 0.0008 | 16/16 | +0.00012 |
| volume | 0.9724 | 0.9758 | +0.0034 | 0.0008 | 15/16 | +0.00010 |
| tri_bot | 0.9847 | 0.9864 | +0.0018 | 0.0004 | 15/16 | +0.00005 |
| tri_top | 0.9881 | 0.9896 | +0.0014 | 0.0003 | 16/16 | +0.00004 |

Not one of the seven goes the wrong way, and the ordering is the finding: the three WORST scalars
(li, kappa, Z_axis, all under 0.97) gain three to four times what the two best ones do. That is the
signature of variance reduction acting on the geometry — the noisiest functionals have the most
noise to remove — and it is the same three scalars C2 named as the expensive ones, kappa first.
The seven sum to **+0.00082 of S**, which reconciles exactly with the aggregate Consistency
difference read independently by `paired.py`.

**Level shift or variance reduction? Mostly the former, and that is the question production turns
on.** The gain correlates **−0.579** with the baseline's own score: ema helps most where the run
was worst. But the arm's spread only falls from 0.00165 to 0.00140 — 15%, not 100% — and the
regression is

    gain = +10.1e-4  −0.196 × (baseline − mean)

Pure variance reduction to a common mean would need a slope of −1.000 and would leave the arm with
no spread at all. At −0.196 the variance-reduction part is worth about **32% of the level shift**;
the other 68% is a genuine improvement to where the fit lands.

**What ema leaves on the table, and where.** Per scalar, after ema, on the screen:

| scalar | ema | ceiling | room | worth in S |
|---|---|---|---|---|
| **li** | 0.9412 | — | 0.0588 | +0.00168 |
| **kappa** | 0.9455 | **0.9908** | 0.0453 | +0.00129 |
| R_axis | 0.9717 | — | 0.0283 | +0.00081 |
| Z_axis | 0.9750 | — | 0.0250 | +0.00072 |
| volume | 0.9758 | — | 0.0242 | +0.00069 |
| tri_bot | 0.9864 | — | 0.0136 | +0.00039 |
| tri_top | 0.9896 | — | 0.0104 | +0.00030 |

Two caveats keep this from being a target list. Only **kappa's** ceiling is measured — C2's 0.9908,
the worst of the seven; the rest use 1.0, so their room is an over-estimate. And this is the
one-net SCREEN, where Consistency reads 0.9770 against production's 0.9803, so the totals here
overstate what a submission would see: C2's production figure is 0.0034 reachable in Consistency,
not the 0.0059 this table sums to.

What survives both caveats is the ORDERING, and it is stable: **li and kappa are the two worst by a
wide margin** and together hold more than half the remaining Consistency. li has been attacked once
already — C10/A24, the poloidal-field form, refuted by the run — and kappa is the scalar C2 singled
out as both the worst-represented and the most expensive at 23.3% of the geometry cost.

**Not yet shippable, and the reason is A25.** This was measured on the quality screen — ONE net.
Production averages four MLP seeds plus the sequence model, and seed averaging already removes
optimisation variance of the same kind EMA removes. A25 is the precedent that makes this a real
risk rather than a formality: three sequence seeds averaged were worth +0.0016 to +0.0023 alone
and **+0.0002** once poured into an ensemble four MLPs already dominated. A paired production
confirmation is required before this goes to the board.

**A null result that tempers the decay sweep before it runs.** If the gain came from HOW MUCH of
the trajectory gets averaged, it should be larger on short runs, where a fixed 1000-step window
covers more of them. Measured over the same sixteen salts — the runs are far more variable than
assumed, **56950 to 149600 steps**, so the window covers 0.67% to 1.76% of them:

    corr(ema gain, steps the run took)        +0.180
    corr(ema gain, window fraction 1000/ran)  -0.164

At n = 16 the standard error on a correlation is about 0.28, so both are indistinguishable from
zero. The gain does not detectably depend on how much of the trajectory the average covers. That
does not refute a longer decay — it says the mechanism is not "average more of it", and the sweep
should be read expecting a flat curve rather than a trend. If `d9999` and `d99995` come back flat,
that is consistent with this, not a surprise.

**RESULT at 4 pairs: +0.00030, +0.00000, +0.00000, −0.00040 — mean −0.00002, SE 0.00014, one
positive sign of four.** Consistency likewise: +0.0011, +0.0005, −0.0004, −0.0020.

**ema is worth nothing in production.** A change accepted on the screen at +0.00101 with 16 signs
of 16 and t = 7.2 measures zero once four MLP seeds are averaged. The prediction of +0.0007,
written before the run, is refuted — and refuted for the reason named in advance: the split into
"level shift" and "variance reduction" does not hold, because with runs that carry similar seed
noise, removing it LOOKS like a uniform level shift.

**The screen-discount hypothesis is confirmed in its strongest form.** It is not "seven times too
large" — for a variance-type change on a four-seed ensemble it is **all of it**. A25 read the same
thing from the other side (+0.0016..0.0023 alone, +0.0002 ensembled).

**And it cancels the 27-hour confirmation that was queued behind it.** `prod_full` existed to
confirm ema on the shipped five-member ensemble at whichever decay the sweep chose. The sweep chose
none — 0.999 was already the optimum, with the long windows measurably worse — and the four-MLP
confirmation has now measured the effect itself at zero. Confirming a zero on the full ensemble
would spend a day to learn nothing, so it was cancelled and the day goes to the disagreement-weight
arm instead, which is priced at +0.0004 to +0.0008 against an oracle of +0.00159.

**What this closes.** `ema` does not ship. Neither does anything else in the decay family, which
was already dropped. The remaining arms — `huber`, `stopcomp`, `emacos` — change what every model
does rather than removing seed noise, so the discount is not expected to apply to them; that
distinction was stated before these results arrived and now has its first hard test behind it.

(Three of the twelve runs were killed by an OOM of my own making — see below — so the pairs are
salts 0, 1, 2 and 4. The missing three are being re-run; at this effect size they cannot change the
conclusion, only complete the record at the pre-registered n = 6.) Consistency rises in both
(+0.0011, +0.0005), so the mechanism is there, but the composite gain is far under the +0.0007
predicted. Two pairs decide nothing and the prediction is not being revised on them.

**The flaw it would expose, named now rather than after the fact.** The +0.0007 came from splitting
ema's screen gain into a "level shift" (68%) and "variance reduction" (32%) by regressing the gain
on the baseline's own score, and assuming an ensemble absorbs only the second. That inference does
not hold: the regression separates *helps the worst runs more* from *helps every run equally*, and
**both are consistent with variance reduction** — if every run carries about the same amount of
seed noise, removing it looks exactly like a uniform level shift. Equating "level shift" with "not
variance" was the error.

If six pairs land near +0.00010, that is the explanation, and it agrees with everything else
measured the same day: 66% of the residual is shared across seeds and four seeds have already taken
three quarters of the removable third, so on a four-seed ensemble there is very little left for ema
to remove.

**And the prediction, written down before that run started, because a confirmation nobody has
pre-committed to is just another look at the data.** Ensembling four seeds removes the
variance-reduction component and keeps the level shift, so the expected production gain is
**about +0.0007**, not the +0.0002 A25 collapsed to. If it lands near +0.0007 the decomposition
above is right and ema ships; near +0.0002 the decomposition is wrong and the effect was
optimisation noise the ensemble already handles; a NEGATIVE result would mean ema and seed
averaging interfere, which nothing here predicts.

## The sequence model earns its place in BOTH blocks — 2026-08-20, refuted at zero cost

`baseline_model.py` supports `psi=<mix>,qb=<mix>` — a different member combination for the flux
coefficients than for q95 and betaN. It has never been used. The hypothesis was obvious: seq alone
scores **R²qb 0.9914 against the MLP's 0.9954**, so drop it from the scalar block and keep it where
it earns its keep.

Rescored on the shipped `baseline_prod.joblib`, same 70 tail shots, no training at all:

| mix | S | R²qb | Consistency |
|---|---|---|---|
| `ensemble` — all five, both blocks | **0.9945** | 0.9967 | 0.9802 |
| seq dropped from `qb` | 0.9944 | **0.9962** | 0.9802 |
| seq dropped from `psi` | 0.9941 | 0.9967 | **0.9785** |

**Refuted, and backwards from the hypothesis.** Removing seq from the scalar block makes R²qb
*worse*, 0.9967 → 0.9962 — the member that is individually weakest at a term still improves the
average of that term, because its errors are decorrelated from the four MLPs'. Removing it from the
map block costs Consistency instead. The equal five-member mix is already right for both.

**Re-run on 300 shots, and it corrects the claim above rather than confirming it.** 70 shots is
the entire clean tail at this salt — `--share` takes from the end of the list and 2% would reach
into validation — so the larger sample has to be `--select-on-val`, which `evaluate.py` provides
for exactly this purpose: choosing among things already fitted, never for reporting. What early
stopping took from those shots is one scalar per model, identical for every candidate, so it cannot
prefer one combination of them over another.

| mix | S | R²qb | Consistency |
|---|---|---|---|
| `ensemble` — all five | 0.9933 | 0.9963 | 0.9752 |
| seq dropped from `qb` | 0.9933 | 0.9962 | 0.9752 |
| seq dropped from `psi` | 0.9930 | 0.9963 | 0.9736 |
| four MLPs, no seq at all | 0.9929 | 0.9962 | 0.9736 |

**The overstatement, corrected.** On 70 shots dropping seq from the scalar block cost R²qb 0.0005
and this file said it "makes R²qb worse". On four times the sample it costs **0.0001** — nothing.
The 0.0005 was sample noise in a comparison whose PAIRING made it look exact; paired removes the
model's noise, not the sample's.

**What survives, and is now firmer:** seq earns its place in the MAP block. Dropping it there costs
Consistency 0.9752 → 0.9736 on both samples. And its total contribution to the ensemble is
**+0.0004, all of it Consistency** — twice the +0.0002 A25 read, and still bought with 75% of the
training budget.

(Absolute scores here are lower than the tail's — 0.9933 against 0.9945 — because they are
different shots. Only the within-column differences mean anything.)

The general lesson is worth more than the arms: **a member's solo score on a term does not say
whether it belongs in that term's average.** A25 measured the same thing from the other side — the
seq family scores worse alone and carries real independent signal.

## "The last decile" is not ramp-down, and A22's mechanism rests on assuming it is — 2026-08-20

The covariates sit beside the costs in `fc_prod.csv`, so the question needs no new run:

| decile | median \|Ip\| | median \|dIp\| | cost x mean | median li |
|---|---|---|---|---|
| 0 | 621 | 0.769 | 1.49 | 0.897 |
| 3–8 | ~1090 | ~0.13 | 0.71–0.88 | ~1.00 |
| **9** | **1010** | 0.173 | **2.35** | **1.176** |

**The current is still at flat-top in the last tenth of a shot.** 1010 against ~1090 is 8% down,
and |Ip| falls below 80% of flat-top on only **22%** of those frames. So for most shots in this
fold the final decile is not a ramp-down at all.

**That undercuts both of A22's stated mechanisms, which each assume termination.** Its "missing
state" clause reads "the current is decaying fastest, so the vessel eddy currents are largest and
most transient exactly here", and its label-noise clause reads "EFIT's own reconstruction is least
constrained as the plasma terminates". The plasma is not terminating in these frames. (The
label-noise clause was already refuted on its own evidence — the truth's d2/d1 is LOWER there — so
this is a second, independent reason it does not apply.)

**What is actually elevated there is li: 1.176 against 1.00 in the middle, an 18% rise.** It also
carries the strongest frame-level correlation with cost of any available covariate (+0.284), is one
of the two worst-scoring Consistency scalars (0.9728), and is the quantity whose truth moves twice
as fast in the tail as mid-shot.

**But li is not the driver, and the obvious reading of the paragraph above is wrong.** Splitting
cost by li quintile WITHIN each position band separates the two:

| li quintile | first decile | middle | last decile |
|---|---|---|---|
| q1 (lowest) | 1.33 | 0.90 | 1.04 |
| q3 | 0.84 | 0.54 | 0.76 |
| **q5 (highest)** | **6.53** | **0.84** | **4.26** |

Inside the middle alone, the top li quintile costs **0.94x** the bottom — li on its own does
nothing. At the same top quintile, the last decile costs **5.07x** the middle. So this is an
INTERACTION, not a main effect: a peaked current profile is harmless at flat-top and very expensive
at either end of the shot. Saying "the model is bad at peaked profiles" would credit li with an
effect it does not have on its own; the frames that cost are the intersection.

The design consequence is that weighting by POSITION already captures this, and adding li as a
second term would buy little — it is flat wherever the position weight is low.

**Concentration is more extreme than the decile view suggests**: the most expensive **5% of frames
carry 46% of the cost**, and they are spread 19% / 18% across the two end deciles with roughly 8%
in each middle one. So even a perfect fix at both ends leaves most of the expensive frames
untouched.

## The cost is U-shaped, not tail-heavy — and only one end has an explanation — 2026-08-20

`diagnose_frames.py` decomposes the composite onto every scored frame. Run for `prod` alone and for
`prod+vesseld`, 14831 frames, the clean tail:

| decile | frames | prod's cost | +vesseld | saved | share of the saving | median \|dIp/dt\| |
|---|---|---|---|---|---|---|
| **0 (first)** | 1515 | 0.00075 | 0.00065 | +0.00009 | **27%** | **0.769** |
| 1–8 | ~11860 | ~0.00303 | ~0.00291 | +0.00014 | 39% | ~0.13 |
| **9 (last)** | 1452 | 0.00116 | 0.00105 | +0.00011 | **34%** | 0.174 |

**The U is a Consistency phenomenon, and D_LCFS is nearly flat in position.** Splitting the same
per-frame costs into the two scored geometry terms (Consistency is 0.00396 of the 0.00493 lost,
D_LCFS the other 0.00098), as a density against a flat shot:

| decile | Consistency | D_LCFS |
|---|---|---|
| 0 (first) | **1.57x** | 1.27x |
| 1–8 | 0.59–0.85x | 0.83–1.00x |
| 9 (last) | **2.66x** | 1.34x |

So the concentration that makes a per-frame weight attractive is almost entirely Consistency's, and
D_LCFS barely varies with position at all. That is a third independent measurement pointing at the
same term — after C2's 76% of the headroom and the instrument's 94% of the noise — and it explains
the pattern every arm has shown: `jac`, `ema` and `bnd` all moved Consistency and all left D_LCFS
where they found it. D_LCFS is flat in position, uncorrelated with what moves Consistency, and
needs something of its own.

**A22 reproduces on a different model**: the last decile is 9.8% of the frames and **23.5%** of the
cost, against the 24.1% over 10.2% A22 measured.

**But the shape is a U, and A22 only ever looked at one arm of it.** The FIRST decile is twice the
middle. Between them the two ends hold 20% of the frames, 39% of the cost, and **61% of everything
`vesseld` saves**.

**The tidy explanation is half wrong, and the check is what caught it.** The obvious story —
`vesseld` integrates dI/dt, eddy currents scale with dI/dt, dI/dt is large at both ends — predicts
that cost should track the current ramp rate. It does not: the correlation is **+0.299 across
deciles and +0.105 frame by frame**, and the decisive detail is that the last decile is **the most
expensive of the ten while its median |dIp/dt| is 0.174 against the middle's 0.13**. The first
decile fits the story (ramp rate 6x the middle, cost 2x). The last one does not fit it at all.

So: the ramp-UP end is expensive in a way vessel currents explain. The ramp-DOWN end is expensive
for a reason still unidentified — A22's "under-representation" hypothesis (a small, differently
distributed, deforming plasma) survives here and the eddy-current one does not reach it.

**This changes the frame-weight design in `ideas.md` (entry Z), which is the useful part.** That
entry proposed a step at the last decile, on A22's evidence. The measurement says the weight should
be raised at BOTH ends. And a weight by |dIp/dt| — which looked like the more physical
parametrisation — is now measurably the WRONG one: it would put its mass on the first decile and
almost none on the most expensive decile of the ten.

## The trio's weighting is already right, and 215 combinations were scored without a rescore

**The technique first, because it is reusable.** A per-frame residual is `truth − prediction` and
the truth is the same for every model, so the residual of an average IS the average of the
residuals. Any weighted combination of models already scored by `diagnose_frames.py` can therefore
be evaluated from the CSVs alone. Validated against three known answers before use:

| combination | from residuals | measured |
|---|---|---|
| prod alone | 0.9802 | **0.9802** |
| prod + vesseld | 0.9823 | 0.9818 |
| prod + vesselp | 0.9820 | 0.9814 |
| the trio | 0.9826 | 0.9819 |

Exact for a single model, and optimistic by ~0.0006 for combinations — the non-linearity of
deriving scalars from an averaged map rather than averaging the scalars. The bias is nearly
constant across combinations, so **differences are trustworthy even though levels are not**, which
is all a relative search needs.

**The search, 215 weightings of prod : vesseld : vesselp:**

| prod | vesseld | vesselp | Consistency | vs equal |
|---|---|---|---|---|
| 2 | 1.5 | 1 (i.e. 4:3:2) | 0.98269 | **+0.00014** |
| 1.5 | 1 | 1 | 0.98267 | +0.00012 |
| **1 : 1 : 1 (shipped)** | | | 0.98255 | — |
| 0.5 | 0.5 | 3 | 0.98047 | −0.00208 |

**The shipped equal weighting is essentially optimal.** The best alternative leans slightly toward
prod — the strongest member, being the only one with the sequence model — and is worth **+0.00003
of S**. The curve is not flat (over-weighting vesselp costs 0.002), so this is a real optimum
rather than a noise plateau; it simply sits where we already are.

**And the same technique bounds the whole question, which closes A21.** Optimising all seven
available models at once — the three artifacts plus the four individual MLP seeds — over their
weights:

    shipped trio          0.98255
    best of 7, optimised  0.98303     +0.00048 of Consistency = +0.0001 of S
    weights: prod 0.242  vesseld 0.288  vesselp 0.220  mlp 0.183  mlp2 0.067

**That +0.0001 is an in-sample CEILING**, fitted with seven free parameters on the same 70 shots it
is scored on. Cross-validated — weights fitted on 69 shots and applied to the 70th, seventy times —
it turns NEGATIVE:

    trio, equal weights            0.98255
    optimised, held out per shot   0.98246      -0.00002 of S

**The whole +0.0001 was overfitting.** Seven free parameters on seventy shots is precisely the
regime where the fit beats the signal, and held out the fitted weights are worse than simply
setting them equal. The optimum leans
slightly toward `vesseld` and adds the plain MLP — i.e. it would raise the MLP relative to the
sequence model inside `prod` — but the size settles the matter.

**A21 ("fit the ensemble weights instead of setting them") is closed, and by a NEGATIVE rather than
a small positive: fitting the weights is measurably worse than setting them equal.** The equal mix
is not merely near-optimal, it is the right answer, and the search for a better one costs score.

It is the same lesson the MAST q95 model taught the day before — a twelve-feature fit scored +0.972
held out and was thrown away on coefficient instability. There the gate caught the overfitting
before anything shipped; here cross-validation caught it inside an intermediate conclusion of my
own, ten minutes after I had written it down as a small positive.

**And the same machinery prices every FUTURE member in advance, which is worth more than either
result.** Adding a hypothetical fourth member of the same accuracy, equally weighted, as a function
of how correlated its residual is with the trio's:

| its correlation with the trio's residual | gain in S |
|---|---|
| 0.90 | +0.00013 |
| 0.75 | +0.00033 |
| 0.60 | +0.00052 |
| 0.45 | +0.00072 |
| 0.30 | +0.00092 |
| **0.00 — perfect independence** | **+0.00131** |

The three shipped members correlate at 0.75 with each OTHER, and a new one of the same kind would
correlate more strongly still with their average — 0.8 to 0.9. That is the top row, +0.0001 to
+0.0002, and it reproduces the measured "adding a fourth gives 0.9950–0.9951" (2026-08-16) from
first principles.

**The hard number is the last row: even a perfectly independent model of the same accuracy is worth
+0.0013.** That caps the entire "add another member" family against the 0.0031 of Consistency still
open — a flawless new view would close 40% of it, and a realistic one 5%. Any future ensemble
proposal can now be priced before it is built, by estimating one correlation.

This also closes a branch previously recorded as blocked: `salts:` accepts no weights, and adding
them would mean editing `baseline_model.py` while a queue is running. It was never worth
unblocking.

## Leave one vessel arm out — the second buys a tenth of what the first does — 2026-08-20

Both vessel arms cost ~40 minutes of training on every rebuild and a third of the submission's
prediction time. On the clean 70-shot tail:

| combination | S | gain over prod | Consistency |
|---|---|---|---|
| `prod` alone | 0.9945 | — | 0.9802 |
| `prod + vesseld` | 0.9949 | **+0.0004** | 0.9818 |
| `prod + vesselp` | 0.9948 | +0.0003 | 0.9814 |
| all three (ships) | **0.9950** | +0.0005 | 0.9819 |

**`vesseld` alone carries 80% of the trio's advantage**, and the second arm adds +0.0001 on top —
at the edge of what seventy shots can say about anything. Both deliver it through Consistency
again: 0.9802 → 0.9818 for vesseld alone.

**Which scalars it fixes, and where — the residuals are in the per-frame table already.** Mean
squared residual per scalar, prod against prod+vesseld, split at the end deciles (x1e6):

| scalar | ends: prod → +vesseld | cut | middle: prod → +vesseld | cut |
|---|---|---|---|---|
| volume | 60086 → 48909 | **19%** | 18451 → 17288 | 6% |
| kappa | 696 → 586 | **16%** | 319 → 309 | 3% |
| R_axis | 49.8 → 43.5 | 13% | 24.4 → 26.8 | **−10%** |
| Z_axis | 38.0 → 33.6 | 12% | 9.6 → 9.0 | 6% |
| tri_top | 796 → 712 | 10% | 352 → 337 | 4% |
| li | 5688 → 5276 | 7% | 1240 → 972 | **22%** |
| tri_bot | 867 → 884 | −2% | 414 → 403 | 3% |

At the ends it acts broadly, 10–19% across most of the geometry. In the middle it acts on **li
specifically, 22%** — the largest single improvement anywhere in the table — and it makes
**R_axis 10% worse**.

**That localises an anomaly recorded earlier and unexplained.** R_axis is the one scalar of the
seven the trio scores WORSE than prod alone (−0.0019). The damage is in the MIDDLE of the shot, not
at the ends where the vessel term does its work — so the vessel integrals are buying end-of-shot
geometry and paying for it in mid-shot axis position. Nothing here says whether that trade is
avoidable; it says where to look.

**The mechanism lines up with A22, which is what makes this more than bookkeeping.** `vesseld` is
leaky integrals of dI/dt on ECOILA and the plasma current — a proxy for the vessel's eddy currents.
Those are largest exactly where the current is decaying fastest, which is the final decile where
A22 measured **24.1% of the geometry cost over 10.2% of the frames**. The narrow arm is already
paying for part of A22's target, and the broad one (all twenty poloidal signals) adds almost
nothing on top — so what helps is the specific physical quantity, not more signals to integrate.

Not acted on: +0.0001 is not a reason to drop `vesselp` from a submission that is already built,
and the trio is what the 0.9938 board entry was measured with. Recorded so that if a rebuild is
ever time-constrained, the arm to drop is known and its price is measured.

## The sequence model's weight, swept again on four times the sample — 2026-08-20

Its whole contribution is Consistency, and Consistency is where 76% of the headroom and 94% of the
instrument's noise live. So: is one part in five too little? Rescored on 300 validation shots, no
training, no model noise:

| seq weight | its share | S | R²qb | Consistency |
|---|---|---|---|---|
| 0.5 | 0.11 | 0.9932 | 0.9963 | 0.9747 |
| 1 (shipped) | 0.20 | 0.9933 | 0.9963 | 0.9752 |
| **2** | **0.33** | **0.9934** | 0.9961 | **0.9757** |
| 3 | 0.43 | 0.9933 | 0.9958 | 0.9755 |
| 5 | 0.56 | 0.9930 | 0.9953 | 0.9749 |

**Unimodal, peaking at a share of 0.33, and the mechanism is visible in the terms**: Consistency
rises to weight 2 and falls after it, while R²qb declines MONOTONICALLY as seq takes more of the
average (0.9963 → 0.9953). The optimum is a balance between two terms, not a noise plateau — which
is what makes the shape worth more than the number.

**The shape replicates.** The 2026-08-15 sweep, on a different selection set and a different
artifact, put the peak in the same place: 0.9930 at 0.20 against 0.9932 at 0.33 and 0.43.

**The magnitude does not clear the clean data.** +0.0001 over the shipped weight, and that same
2026-08-15 entry recorded the honest ending: on the untouched tail, 0.20 / 0.33 / 0.43 all scored
**0.9945**. Two sweeps now agree about where the optimum is and neither can show it is worth
anything on data nothing has touched.

**Confirmed on the clean tail, in direction only.** Scored on the untouched 70 shots:

| | S | R²qb | Consistency |
|---|---|---|---|
| `ensemble` (weight 1) | 0.9945 | 0.9967 | 0.9802 |
| `seq*2` | 0.9945 | 0.9968 | **0.9805** |

S is identical to four decimals. Worked out term by term it is **+0.00007** — Consistency moves the
same way it did on the selection set, which is now three independent samples agreeing about the
direction, and the size stays below what the display shows, let alone what the fold could.

**Verdict: a free change with a replicated shape and an unmeasurable size.** It costs one number in
`params.yaml` and nothing at inference. It should NOT be spent as a submission of its own —
+0.0001 against a board-to-local offset that drifts by 0.0003 would tell us nothing. If ema clears
its production confirmation, this rides along in the same entry, where the combined change is large
enough for the board to speak about.

## The shipped configuration re-scored today, and what headroom it actually has — 2026-08-20

A regression check before trusting any plan built on recorded numbers. The three shipped artifacts,
scored on the same untouched 1% tail with today's code:

| | S | R²ψ | R²qb | 1 − D_LCFS | Consistency |
|---|---|---|---|---|---|
| `ensemble` (prod alone, 5 members) | **0.9945** | 0.9998 | 0.9967 | 0.9902 | 0.9802 |
| `salts:prod+vesseld+vesselp` (what ships) | **0.9950** | 0.9999 | 0.9974 | 0.9908 | 0.9819 |

Both reproduce their recorded values exactly. **Nothing has drifted**, so the 0.9945 / 0.9950 /
0.9938-on-the-board chain of numbers every current plan rests on is sound.

**And it gives the target its true value.** The trio's gain over prod alone is +0.0005, of which
Consistency is +0.00034 — two thirds, again. More usefully, the SHIPPED Consistency is **0.9819**
against C2's ceiling of 0.9974, so what is actually left in the term that holds everything is

    (0.9974 − 0.9819) × 0.20 = **+0.0031 of S**

That is the number to measure future work against, not the +0.0059 the one-net screen suggests: the
screen sits at Consistency 0.9770 and has room the shipped model has already spent by averaging
thirteen nets across three feature sets.

**The target list, per scalar, on the configuration that actually ships:**

| scalar | prod alone | the trio | trio's gain | room to 1.0 | worth in S |
|---|---|---|---|---|---|
| **kappa** | 0.9601 | **0.9640** | +0.0039 | 0.0360 | **+0.00103** |
| **li** | 0.9680 | **0.9728** | +0.0048 | 0.0272 | **+0.00078** |
| R_axis | 0.9806 | 0.9787 | **−0.0019** | 0.0213 | +0.00061 |
| volume | 0.9814 | 0.9829 | +0.0015 | 0.0171 | +0.00049 |
| Z_axis | 0.9883 | 0.9901 | +0.0018 | 0.0099 | +0.00028 |
| tri_bot | 0.9908 | 0.9918 | +0.0010 | 0.0082 | +0.00023 |
| tri_top | 0.9922 | 0.9931 | +0.0009 | 0.0069 | +0.00020 |

**Every ceiling is now measured, not assumed.** `evaluate.py --mode basis` pushes the truth through
the artifact's own 50 components and back, which is what a perfect regression could reach:

| scalar | shipped | basis ceiling | room | worth in S |
|---|---|---|---|---|
| **kappa** | 0.9640 | **0.9908** | 0.0268 | **+0.00077** |
| **li** | 0.9728 | 0.9985 | 0.0257 | **+0.00073** |
| R_axis | 0.9787 | 0.9999 | 0.0212 | +0.00061 |
| volume | 0.9829 | 0.9997 | 0.0168 | +0.00048 |
| Z_axis | 0.9901 | 0.9996 | 0.0095 | +0.00027 |
| tri_bot | 0.9918 | 0.9968 | 0.0050 | +0.00014 |
| tri_top | 0.9931 | 0.9969 | 0.0038 | +0.00011 |
| | | | | **+0.00311** |

The total reconciles with the +0.0031 computed independently from C2's aggregate ceiling, and the
composite ceiling reproduces at 0.9990.

**Two corrections to the placeholder version of this table.** kappa's room falls from 0.0360 to
0.0268 — its ceiling, **0.9908, is by far the lowest of the seven**, so the 50-component basis
itself struggles to represent elongation, which is exactly why C2 called it the sharpest case. And
both triangularities cap at 0.9968–0.9969 rather than 1.0, so their value was overstated by nearly
half.

**kappa and li still hold half of what is left**, and the top four hold 0.0026 of the 0.0031.

**Is kappa's low ceiling fixable by a bigger basis? Measured, and no.** The artifact stores exactly
50 components so more cannot be tried without refitting — but FEWER can, and the slope at 50 says
whether the ceiling is still climbing. Truth projected onto k directions and back, then run through
the scorer's own derivations, no fitting anywhere:

| scalar | k=10 | k=20 | k=30 | k=40 | k=50 | slope 40→50 |
|---|---|---|---|---|---|---|
| **kappa** | **0.3219** | 0.9598 | 0.9821 | 0.9881 | 0.9908 | **+0.0027** |
| li | 0.7705 | 0.9653 | 0.9926 | 0.9972 | 0.9985 | +0.0014 |
| Z_axis | 0.7295 | 0.8627 | 0.9888 | 0.9984 | 0.9996 | +0.0012 |
| tri_bot | 0.8649 | 0.9811 | 0.9929 | 0.9957 | 0.9968 | +0.0011 |
| R_axis | 0.7373 | 0.9688 | 0.9940 | 0.9994 | 0.9999 | +0.0005 |
| tri_top | 0.9006 | 0.9849 | 0.9943 | 0.9964 | 0.9969 | +0.0005 |
| volume | 0.7692 | 0.9895 | 0.9977 | 0.9991 | 0.9997 | +0.0005 |

kappa's ceiling IS still climbing at 50, and faster than anything else — twice the next scalar. But
the arithmetic closes the branch: kappa sits at **0.9640 against a ceiling of 0.9908**, a gap of
0.0268, while ten more directions would move that ceiling by about 0.003. **The representation is
worth at most a tenth of the gap; the other nine tenths are the regression.** And `j7_npca75`
already measured n_pca = 75 at −0.0015 on overall S, because the extra directions are targets the
fit has to predict and it predicts them badly.

**One genuinely new fact about the geometry**: at k = 10, kappa scores **0.32** where every other
scalar is between 0.73 and 0.90. Elongation depends on fine boundary structure that the leading PCA
directions simply do not carry — it needs roughly twice as many directions as its neighbours before
it is representable at all.

**One anomaly, small and worth remembering: R_axis is the only scalar the trio makes WORSE than
prod alone** (−0.0019, against +0.0009 to +0.0048 on the other six). Whatever the two vessel arms
add, they cost something in where the magnetic axis lands. There is no lever to exploit it — the
`psi=`/`qb=` grammar splits the map from the submitted scalars, and the seven Consistency scalars
are DERIVED rather than predicted, so no mix can be chosen for them separately.

**Which sharpens the ema question rather than settling it.** ema's +0.00082 of Consistency was
measured where there was more room and one net. The trio already averages thirteen, so its
variance-reduction third (32% of the effect) is the part most likely to be absorbed; the level
shift (68%) is the part that should survive. The paired production confirmation is what decides it,
and this is the baseline it will be read against.

## B9's open half, answered: the regression predicts every direction that carries variance — 2026-08-20

C2 measured what the 50 directions can REPRESENT and explicitly left the narrower question open:
"reduced-rank regression is about which directions are PREDICTABLE, and nothing here measures
that." Projecting both the truth and the ensemble's prediction onto the same basis measures it
exactly, per direction, on the clean tail:

| component | share of psi variance | R² of the regression |
|---|---|---|
| 0 | **72.20%** | 0.9998 |
| 1 | 19.79% | 0.9999 |
| 2–3 | 5.29% | 0.9996–0.9998 |
| 19 | 0.01% | 0.9350 |
| 29 | ~0% | 0.8619 |
| 39 | ~0% | 0.7249 |
| 49 | ~0% | **0.4291** |

**Variance-weighted mean R² = 0.9997.** Exactly one component of fifty scores under 0.5, and it
carries 0.00% of the flux variance.

**This closes B9.** Reduced-rank regression drops the directions that are hard to predict — and
those are precisely the ones with no variance, so dropping them costs nothing in R²psi and takes
away exactly what the geometry needs. There is nothing to win there.

**And it states the central paradox of this challenge precisely, which nothing had done before.**
Put beside the ceiling-vs-basis-size curve — kappa scores **0.32** at k = 10 and its ceiling is
still climbing at k = 50 — the two measurements say:

> the regression predicts the first four components (97.3% of the flux) essentially perfectly, and
> the derived functionals depend on components 20 to 50, where its R² falls from 0.93 to 0.43.

That is why R²psi reads 0.9998 against a ceiling of 1.0000 while Consistency reads 0.9819 against
0.9974. The map is finished as pixels because pixels are dominated by four directions; the seven
scalars are not, because a boundary's shape lives in the directions that carry almost no energy.

**And the loss already does the obvious thing about it — by a factor of 1434.** The target scaler
is not a per-component scaling but a full 50x50 matrix, so the loss is `||L (c_pred - c_true)||²`
and an error of 1 in component i costs `||L[:, i]||²`. Reading those column norms out of the
shipped artifact:

| component | share of flux variance | share of the loss weight | ratio |
|---|---|---|---|
| 0 | **73.97%** | **0.45%** | 0.0 |
| 1 | 16.58% | 1.56% | 0.1 |
| 5 | 0.72% | **24.44%** | 34 |
| 19 | 0.007% | 0.84% | 117 |
| 29 | ~0% | 1.12% | 3060 |
| 49 | ~0% | 0.56% | 75059 |
| **20–49 together** | **0.021%** | **30.0%** | **1434** |

The Jacobian metric already gives thirty percent of the loss to directions holding two hundredths
of a percent of the flux, and it all but ignores the component that holds three quarters of it.

**So the loss weighting is not the problem, and that closes the obvious next idea before it is
proposed.** It also explains `jac`'s +0.00044: the weights were already approximately right, so
estimating them on four times the frames refines something that was not the binding constraint.

**What is left is an information limit.** Components 20–49 get thirty percent of the loss, are
represented by the basis, and are still predicted at R² 0.43 to 0.93 — while 66% of the residual is
shared by every seed, so it is not optimisation noise either. The inputs simply do not determine
the fine boundary structure. Four directions are exhausted at once by this: more members (seed
floor 56%, feature floor 75%), a bigger basis (worth a tenth of kappa's gap), a better-estimated
metric (`jac`, refuted), and re-weighting toward the geometry (already 1434x).

**The one direction with a measured success is new information**: `vesseld`'s leaky integrals of
dI/dt bought +0.0004, the largest single gain since the four-seed ensemble, and they work by
telling the model something the 84 instantaneous features could not. That, and not another model,
is where the remaining 0.0031 of Consistency lives.

## Two thirds of the error is shared by every seed, and that bounds the whole averaging family

Scoring `mlp`, `mlp1`, `mlp2`, `mlp3` separately on the clean tail and splitting each scalar's
residual into the part the four share and the part specific to a seed:

| scalar | shared | removable by averaging | mean pairwise r |
|---|---|---|---|
| **li** | **76%** | 24% | 0.686 |
| **R_axis** | **76%** | 24% | 0.685 |
| **kappa** | **71%** | 29% | 0.620 |
| volume | 65% | 35% | 0.542 |
| tri_bot | 63% | 37% | 0.511 |
| tri_top | 61% | 39% | 0.488 |
| Z_axis | 54% | 46% | 0.398 |

**Pooled, the four-seed average keeps 66% of a single seed's squared error**, and the textbook form
reproduces it exactly: with pairwise correlation rho ≈ 0.55, an n-member average carries
`rho + (1 - rho)/n`, which is 0.55 + 0.45/4 = 0.6625.

Three consequences:

* **8 seeds would carry 0.55 + 0.45/8 = 0.606** — recovering 5.6 points of the 34 that were ever
  removable, a sixth of the remainder. That is an independent confirmation, by a completely
  different route, of the costing that declined the eight-seed arm the same day.
* **An infinite ensemble of these seeds floors at 55%.** Nothing below that is reachable by
  averaging at all.
* **kappa (71% shared) and li (76%) are the LEAST helped by averaging** — and they are exactly the
  two scalars holding half the remaining headroom. That is why ema's gain, though real and 16/16,
  is small precisely where the money is.

**It also corrects a phrase used earlier in this file.** "The remaining error is variance, not
bias" was too loose: the residuals are unbiased (bias/sd between −0.09 and +0.08), but two thirds
of what is left is *shared model error* — the same mistakes on the same frames whatever the seed —
which averages to zero across frames while being immune to averaging across seeds. Only the other
third was ever seed noise, and four seeds have taken three quarters of it.

So the remaining work belongs to changes that alter what every seed does — features, architecture,
loss, data — and not to more copies of the same fit.

**And feature diversity, measured the same way, is WEAKER than seed diversity — the opposite of
what was assumed.** Scoring the three shipped artifacts separately and correlating their residuals:

| | mean pairwise r | the average keeps | floor at infinity |
|---|---|---|---|
| 4 seeds, one feature set | **0.561** | 67% | 56% |
| 3 feature sets | **0.746** | 83% | 75% |

Three feature sets decorrelate LESS than four seeds of one. Two reasons, and both are visible in
how they are built: the trio's members are themselves four-seed ensembles, so the seed component
has already been averaged out of each and what remains is the shared model error they hold in
common; and the vessel arms are the same 84-feature base with two or twenty extra integral columns,
which is a variation on a view rather than a different one.

**The bound this puts on the whole ensembling direction:** a fourth feature set of this kind takes
83% to 81%, a fifth to 80%, and the floor is **75%** — three quarters of one artifact's squared
error survives any number of them. That is the measured content of "it saturates at three"
(2026-08-16), which recorded the saturation without a number.

The caveat that keeps this honest: this measures the diversity of THIS family — leaky integrals
bolted onto one feature base. It says nothing about a genuinely different representation (other
targets, another architecture, a different decomposition), which has never been tried and is not
bounded by these numbers.

The three arms still queued (`huber`, `stopcomp`, `aux`) all change what every model does rather
than how many are averaged, which is now the only direction the measurements leave open.

## The seven Consistency scalars fail almost independently — 2026-08-20

Correlating the seven per-frame residuals against each other, each standardised by its own spread,
on the clean tail:

    eigenvalues  1.59  1.37  1.05  1.02  0.92  0.69  0.37
    leading component 22.7% of the variance   (complete independence would be 1/7 = 14.3%)
    mean |off-diagonal| correlation 0.12, max 0.36

**There is no dominant error mode.** The strongest pairs are kappa–tri_bot (0.36), kappa–tri_top
(0.29), kappa–li (**−0.27**), volume–li (0.23); everything else is under 0.15.

Two consequences, and the second is unwelcome:

* **No single fix lifts all seven.** That is why every arm measured so far produces a small, broad
  Consistency gain rather than a jump — there is no common latent cause to remove. It also lowers
  the expected value of any idea sold as "fix the boundary and the geometry follows".
* **kappa and li are ANTI-correlated at −0.27**, and they are precisely the two scalars holding
  half the remaining headroom (0.0268 and 0.0257 of room). A change that improves one may cost the
  other, so their gains cannot be assumed to add.

This does NOT argue against a per-frame position weight, which is sometimes confused with it: that
change reallocates optimisation effort to frames rather than removing a shared factor, so seven
independent errors at the same expensive frames can all shrink at once. What it does argue against
is expecting one mechanism to be worth the sum of the seven gaps.

(For contrast, the same measurement on MAST put the leading component at 34.8% — its error is
markedly more coherent than DIII-D's.)

## Everything that moves S at all moves it through Consistency — 2026-08-20

Three arms measured on the same instrument, decomposed the same way, in units of contribution to S:

| arm | R²ψ | R²qb | 1 − D_LCFS | Consistency | total |
|---|---|---|---|---|---|
| `jac` (n=16, refuted) | +0.00002 | −0.00001 | −0.00000 | **+0.00043** | +0.00044 |
| `ema` (n=16, accepted) | +0.00004 | +0.00008 | −0.00008 | **+0.00082** | +0.00101 |
| `bnd` (n=6, incomplete) | +0.00000 | −0.00006 | +0.00002 | **+0.00039** | +0.00032 |

**Not one of the three moves anything else.** R²ψ carries 55% of the composite's weight and has not
moved by more than 0.00004 in any arm, because the map is already at 0.9998 against a ceiling of
1.0000 — there is nothing there to win. The whole game is the seven functionals read OFF that map.

`bnd` is the sharpest illustration and the early rows are worth flagging before the arm completes:
the boundary term was added specifically to supply what the Jacobian cannot express about
**D_LCFS**, and at n=6 D_LCFS is the one term it does not move (+0.00002, 1 of 6 salts). What moves
is Consistency, 5 of 6. If that survives to sixteen, the reading is that displacing a level set
changes the functionals that READ the boundary long before it changes a Hausdorff maximum over it —
which is a fact about the metric, not about the term.

This converges with two independent measurements: C2 put **76% of the reachable 0.0045 in
Consistency**, and the noise decomposition puts **94% of the instrument's variance** there too. The
term that holds the headroom is also the term that holds the noise, and it is the only one anything
has yet moved.

## A22's open half, closed — the tail's error is the MODEL's, not EFIT's — 2026-08-20

A22 measured that the final tenth of every shot carries **24.1% of the geometry cost over 10.2% of
the frames**, with the implied coefficient error 2.22x the fold average, and left one hypothesis
explicitly open since 2026-08-15: part of that may be irreducible **label noise**, because EFIT's
own reconstruction is least constrained as the plasma terminates. That question decides the sign of
A22's proposed fix — up-weighting the tail helps if the error is the model's and fits noise harder
if it is EFIT's.

**It is answerable without training anything**, from the truth alone, and the shipped columns
`efit_li`, `efit_r_axis` and `efit_z_axis` are all it takes. On a 20 ms frame clock a physical
quantity evolves smoothly, so reconstruction noise shows up in the SECOND difference.

First reading, 60 shots, median |d2| in units of each shot's own spread — the truth IS rougher at
the end: **li 1.72x, z_axis 1.24x, r_axis 1.14x** against mid-shot. Taken alone that supports the
label-noise story.

**It does not survive the control, and the control reverses it.** Roughness conflates noise with
genuinely faster evolution — at ramp-down the plasma really is changing quickly. The discriminator
is d2 against d1: white noise inflates the second difference far more than the first, while fast
smooth motion raises both together.

| | d2/d1, last vs mid | first difference alone |
|---|---|---|
| li | **0.85x** | 2.01x |
| r_axis | **0.89x** | 1.29x |
| z_axis | **0.88x** | 1.39x |

The last decile is **smoother per unit of motion**, not rougher: li moves twice as fast there and
its d2/d1 ratio FALLS. That is coherent dynamics, not reconstruction noise, and li — the scalar
that looked worst on the raw measure — is simply the fastest-moving one.

**So the label-noise hypothesis is refuted and A22's fix is legitimate.** The tail's error belongs
to the model. Up-weighting those frames is fitting signal, not noise.

Two limits on how far to carry this. The three scalars measured are the ones shipped as columns;
kappa, the triangularities and volume would have to be derived frame by frame, and kappa is the
expensive one. And "the model's fault" does not mean "the loss's fault" — C7 already measured the
metric's sensitivity in the tail at 0.98x the median, so the frames are not under-weighted BY THE
METRIC. What is left is that they are simply harder, which is what a per-frame weight by position
is for.

## Scoring on 5% instead of 1% — worthless on the screen, worth 44% on production — 2026-08-20

The scored tail is 1% of the shot list, 70 shots, and `--share` takes from the end, so anything
past 1% reaches into validation. Enlarging it means moving shots out of validation
(0.80/0.15/0.05), which only affects early stopping.

**Whether it helps depends entirely on which instrument, and the split is stark.** Subsampling the
70 scored shots without replacement and correcting for the finite population — without that
correction the constant is not flat and the extrapolation is wrong — the sampling standard
deviation of a paired difference is `k/sqrt(n)` with **k = 0.00132**:

    sampling sd at  70 shots (1%)   0.000158
    sampling sd at 350 shots (5%)   0.000071

| instrument | total paired sd | of which sampling | at a 5% tail |
|---|---|---|---|
| quality screen, one net | 0.00085 | **3%** of the variance | −1% noise |
| production, four MLPs | 0.00017 | **86%** of the variance | **−44% noise** |

On the screen the noise is training randomness — which salt, where the trajectory lands — and more
scored shots do nothing about it. On a production comparison the training part is only 0.00006 and
**the 70-shot sample is almost all of it**; five times the shots would take the paired sd to
0.00009.

**Applied to the 27-hour final confirmation, which now scores 5%.** Scoring 350 shots instead of 70
costs a few minutes per run against 36 minutes of training, so the 44% is close to free. The
six-pair confirmation already running stays at 1% — changing the scored set mid-measurement would
make its pairs incomparable, and it is nearly done.

**The general lesson worth more than the arms:** the same knob has opposite verdicts on the two
instruments, and reading the decomposition rather than the total is what tells them apart. "Score
more shots to cut noise" is right or worthless depending on where the noise actually is.

## What limits the 16-salt instrument, and why the obvious fix is not worth buying

Decomposing the paired standard error of S into the four scored terms, over the ema arm:

| term | raw SE | × weight | SE in S | share of variance |
|---|---|---|---|---|
| **Consistency** | 0.00073 | 0.20 | **0.00015** | **94%** |
| R²qb | 0.00019 | 0.15 | 0.00003 | 3% |
| R²ψ | 0.00004 | 0.55 | 0.00002 | 3% |
| 1 − D_LCFS | 0.00005 | 0.10 | 0.00001 | 0% |

The quadrature sum is 0.00015 against 0.00014 measured directly on S, so the decomposition closes.
**Consistency alone sets what this instrument can resolve** — R²ψ carries the largest weight in the
composite and contributes 3% of its noise, because the model reproduces the map almost exactly
(0.9998) and reproduces the functionals read off it far less exactly.

**The obvious sharpening, priced and declined.** Consistency is scored on the 1% tail — 70 shots.
Doubling it to 140 (shares 0.80/0.18/0.02, taking from validation rather than training, since
validation only drives early stopping) would cut its noise by sqrt(2): paired SE 0.00014 → 0.00010,
2 sigma 0.00028 → 0.00020. It costs nothing in validity, because both arms of a pair lose the same
shots and the difference is untouched.

**It is not worth it yet.** A different tail is a different scored set, so every number in the
program above becomes incomparable and the baseline arm has to be re-run — 2.7 hours — to buy a
resolution improvement from 0.00028 to 0.00020 when the effects being chased are 0.0005 to 0.0010
and already clear the current threshold. Recorded because knowing WHERE the noise lives is worth
having; the fix goes in the drawer until an effect actually lands between 0.0002 and 0.0003.

## What a production run actually costs, and where it goes — 2026-08-20

Read off a full run's own fit timings, because the plan for the ema confirmation was built on a
guess and the guess was wrong by a factor of four:

| model | fit time |
|---|---|
| ridge | 1.2 s |
| mlp | 256.8 s |
| mlp1 | 451.7 s |
| mlp2 | 270.7 s |
| mlp3 | 306.2 s |
| **seq** | **3984.1 s** |

**The sequence model is 75% of a production run.** Four MLP seeds fit in 21 minutes together; seq
alone takes 66. A production fit is therefore about **1.7 hours**, not the ~10 minutes a quality
screen takes, and sixteen of them would have been 27 hours.

A trap in reading the logs that cost a wrong estimate first time: the bracketed timestamp is
`H:MM.S` on long runs, so `[3:47.0]` at the end of a run is three hours forty-seven, not three
minutes. The fit timings are what settle it.

**And the cost/benefit of seq is worth stating plainly.** A25 measured the ensemble at **0.9943**
with seq against **0.9941** without: seq buys **+0.0002** for 75% of the training budget. For
comparison ema, at zero cost, buys +0.00101 on the screen. If the paired production confirmation
lands anywhere near its predicted +0.0007, ema will be worth **three times what the sequence model
is worth, for none of the time**. That does not by itself argue for dropping seq — its value in an
ensemble is diversity rather than accuracy, and A25 showed the seq family carries real independent
signal (three seq seeds averaged 0.9932 against 0.9909/0.9914/0.9916 individually) — but it does
mean the 66 minutes should be justified rather than assumed.

## Measurement facts worth not re-deriving

- **MAST's real machine dimensions are known, not guessed.** The first wall is a closed 37-vertex
  polygon in `solver/machine.py` (`MAST_LIMITER`), taken from `efm/limiterr` / `efm/limiterz` in the
  public MAST archive at STFC Echo — **byte-identical on shots sampled from all five campaigns
  M5–M9**, so it is a machine constant of the same kind as the coil rectangles the competition
  ships. Do not re-derive it from the data. Specifically:
  - the **inboard wall is at R = 0.19524 m**, not the 0.1963 this fork had inferred from where
    `efit_lcfs_r` bottoms out on the demo shots — a 1.1 mm error, and the shape matters more than
    the radius: the outboard wall at R = 1.90, the top and bottom at Z = ±1.825 and every divertor
    step simply did not exist for the solver before;
  - it is worth **S 0.6310 → 0.6336** with the calibration untouched and **0.6359** refitted under
    it, and it improves the boundary search's tails (p90 5.35% → 4.69%, max 8.52% → 7.77%) rather
    than its median;
  - rounding the literal to four decimals costs 0.04 mm on a 30 mm grid and **0.0001 of S** —
    measured, then shipped exact anyway.
- **That archive is the competition's withheld MAST ground truth, bit for bit**, and nothing else
  is taken from it. Verified on demo shot 28348, whose truth the competition does ship: frames match
  to 0.0000 ms and `efm/psirz` against `efit_psirz` differs by **0.000e+00**, as do `li`, `betan`,
  `elongation`, `magnetic_axis_r` and `psi_axis`. The 1206-shot public test fold is drawn from the
  same campaigns — the demo shots are M8 — so its answers are in there. The terms allow external
  public datasets **with disclosure** and separately forbid any attempt to "de-anonymize, memorize,
  or otherwise leak the hidden ground truth": training on the archive would be the latter even
  unintentionally, and excluding the test fold first requires identifying it, which is also the
  latter. **The external dataset must be disclosed in any methods report.** The organisers have been
  asked about the wider question; until they answer, `solver/mast_archive.py` reads geometry only.
- **DIII-D's coil calibration must be fitted OUTSIDE the plasma boundary**, and with a per-frame
  plasma amplitude projected out. Inside the boundary one filament cannot represent a distributed
  current, and fitting over the whole grid returns F-coil gains from −1.4 to +2.8. With both, the
  F-coils come out at a median of **0.965, 10–90% 0.855–1.137**, which is what `coil_field.py`
  independently reports.
- The noise floor at production is **0.0009** of S over seeds; see the table above. An older figure
  of 0.0060 came from quality scale (352 shots) and does not apply to the production model — with
  3168 shots the optimisation lands in nearly the same place every time.
- **Salts, not seeds.** Ridge is deterministic given a split and moves 0.7117 → 0.6219 → 0.5067
  across salts 0/1/2, against 0.0 for any seed change.
- **Ceilings must be measured on the term that loses the score.** "50 components reconstruct ψ to
  R² = 1.000000" was true and measured on R²ψ, which saturates at 30 components; Consistency does
  not saturate until 50. The conclusion survived by luck.
- **The instantaneous currents nearly determine the targets**: nearest-neighbour across shots gives
  a ceiling of R² 0.97 (li) to 0.997 (betaN), while the model reaches 0.90–0.93. Features are not
  the binding constraint.
- **`jacobian_delta` is scale-dependent**: it must sit at √(1 − R²ψ) of the run it is used in —
  0.05 at quality, 0.033 at production.
- **The coil-field calibration** needs a plasma filament with a per-frame amplitude, or the coil
  gains absorb the plasma's own field. Pooled amplitude reads 0.997 on 10 shots and 0.911 on 40;
  per frame it is 0.87–0.89 at any size. `ECOILA` is not identifiable at all — its gain moves
  between +142 and −10 depending on the fit — so the pipeline fits its own gains instead.

## The composite was being PRINTED at four decimals — fixed 2026-08-21

`local_score.py` printed its summary row at `:8.4f`, and every screen and production CSV is built by
parsing that row. The differences these experiments resolve are around 0.0009, so the printer was
quantising each measurement into steps of 0.0001, contributing sd 0.00004 to a paired difference
whose sample sd is 0.00009 — a 10% wider interval, given away for nothing.

It is visible in the `ema` production pairs, whose per-salt differences are exact multiples of
1e-4: `+3.0 +0.0 +0.0 +1.0 -4.0`. Some of those zeros are not zeros. They are differences smaller
than the printer could show.

This is the second time the same class of bug has cost something here. The MAST inboard wall
literal was also first taken at four decimals — 0.04 mm on a 30 mm grid, worth 0.0001 of S, and it
was measured and then shipped exact. A number that is only ever read by a human can be rounded; one
that is parsed by a tool cannot.

Now `:10.6f`. Formatting only — no computation changed — and the file was replaced ATOMICALLY so a
run starting mid-edit could not import a half-written module. Every parser was checked against the
new width before the change: the runners' `awk` extraction, `paired.py`, and `mast/loo.py`'s
`COMPOSITE S` regex, which reads a different line that was left alone.

`my_experiments/prodpair.py` was added alongside it. `paired.py` matches a baseline FILE against arm
FILES, but the production confirmations put both arms in ONE file, because a lane runs both arms of
a salt back to back so a pair always completes together. It was validated by reproducing the known
`ema` answer — +0.00000, 2/5 signs — before being pointed at anything new.

## `seq` dropout 0.1: strongly negative, and it refutes the reason I ran it — 2026-08-21

First pair, salt 0: **0.990500 -> 0.986600, a fall of 0.0039**, with every term down — psi 0.9997
-> 0.9993, qb 0.9913 -> 0.9869, 1-D_LCFS 0.9860 -> 0.9837, Cons 0.9669 -> 0.9531. That is forty
times the effect these screens usually resolve, so one pair settles the sign.

**It is not a measurement artefact.** The obvious suspicion was that dropout stayed active during
validation, which would corrupt early stopping rather than the model; it does not — `seq_model.py`
calls `net.eval()` before validating (line 266) and `net.train()` after (275), and again on both
prediction paths. The dropout run also ended in 2767 s against the baseline's 5311 s, which looked
like a premature stop, but the curve says otherwise: its best was 0.044549 at step 25500 and
validation then ROSE — 0.0499, 0.0565, 0.0702 — so it had settled worse, not been cut off early.

**What actually happened is the opposite of the hypothesis.** Dropout made the model UNDERFIT.
Training loss never went below 0.0298 against the baseline's 0.012, and validation followed it down
only to 0.0445 against 0.0320. The train/val gap did narrow, 2.7x -> 1.66x, exactly as a
regulariser promises — but it narrowed by ruining the fit, not by improving generalisation.

So the 2.6x gap I cited as evidence of overfitting is **not the kind of overfitting regularisation
repairs**. `seq` is capacity-limited like the MLP before it, and dropout takes away capacity it
needs. The parameter-counting prior — 5633 sequences against 1.25M rows — was a real argument and
it is now refuted by measurement, which is the point of running it.

The remaining seven salts were cancelled rather than spend eighteen hours confirming a sign that
one pair had already established at forty times the noise.

## The better finding was hiding in the BASELINE curve: `seq` trains at a constant learning rate

Reading the control arm rather than the treatment: baseline validation swings about 30% between
adjacent evaluations late in the fit — 0.032029, 0.043272, 0.035239, 0.043711 — so the "best
checkpoint" is being selected partly out of noise. Both `seq` and `mlp` ship `lr_schedule: none`,
a constant 1e-3 for the whole run, and that volatility at convergence is what a constant learning
rate looks like.

`cosine` is implemented and unused for `seq`. The horizon is the trap — `models.py:384` records
that a fit stopping near a quarter of its declared horizon only traverses the cosine's flat
beginning — and here the horizon is known from the baseline: best at 73000, stopped at 83000. So
`lr_t_max_steps=80000` traverses the actual curve. Queued as the `cos` arm on salts 0-7.

## The `D_LCFS` weight had the wrong sign in the analysis tools — fixed 2026-08-21

`paired.py` and `prodpair.py` carried `"D_LCFS": -0.10`. The runners write `1 - D_LCFS` into that
column, because that is what `local_score.py:481` prints, and in the composite the term is
`+0.10 * (1 - D_LCFS)`. So the weight is **+0.10**: higher is better in that column, like every
other one.

Caught by arithmetic, not by reading the code. The per-term contributions must sum to the measured
dS, and for `huber` they summed to +0.00079 against a measured +0.00088. With the sign corrected
they sum to +0.00087. A decomposition that does not add up to the total it decomposes is the
cheapest possible check, and it should be run on every such table.

**No decision was affected.** Every gate is applied to the `S` column directly, which is parsed from
the score and never reconstructed from terms; the weights are used only to explain a result after
it is in. But the explanations built on them were wrong, which is why this is recorded rather than
quietly patched.

## Where `huber`'s gain actually comes from — and a sharper prediction for production

With the sign fixed, the 16-salt decomposition of the accepted `huber` arm:

| term | raw | as S | signs |
|---|---|---|---|
| R2_psi | +0.00002 | +0.00001 | 2/16 |
| R2_qb | +0.00068 | +0.00010 | 7/16 |
| 1-D_LCFS | +0.00041 | +0.00004 | 13/16 |
| **Consistency** | **+0.00359** | **+0.00072** | 12/16 |

**82% of the gain is Consistency, and R2_psi contributes nothing measurable** — +0.00001 with 2/16
salts up, which is not a small effect but no effect. That is a coherent mechanism: Huber
down-weights large-residual frames, Consistency is the mean of seven pooled R2 over all frames, and
a pooled statistic is dominated by its worst members.

**This partly undercuts my own prediction, and I would rather say so before the production pairs
land than after.** I argued `huber` should survive ensembling because it changes what every model
does, unlike `ema`, which the ensemble had already done for free. That still holds in kind. But
averaging four seeds ALSO helps worst-case frames most — variance reduction is largest where
variance is largest — so the specific term `huber` works through is a term the ensemble is already
improving by another route. The expected production effect is therefore somewhere between the
screen's +0.00088 and the full dilution `ema` suffered, and closer to the screen only if the two
mechanisms are as independent as they look.

Six pairs at a paired sd of ~0.00009 give SE ~0.00004, so even a quarter of the screen effect lands
at t = 6. The measurement can resolve the answer whichever way it falls.

## Did the 1% -> 5% tail actually cut the noise? Asked properly — 2026-08-21

The change was made on a decomposition: 86% of a production comparison's paired variance was the
shot SAMPLE rather than the training, so five times the shots should cut the paired sd from 0.00017
to 0.00009. Cost measured since: median wall 1854 s -> 2666 s, **+44% per run**.

**The direct check on live data does not confirm it, and cannot yet.** `ema` prod (70 shots) has a
paired sd of 0.00025 over 5 pairs; `huber` prod (352 shots) has 0.00042 over 5. The point estimate
moved the WRONG way, but the comparison is confounded — `ema`'s true effect is zero on every salt,
so its paired sd is nearly pure noise, while `huber`'s effect may genuinely vary by salt and that
heterogeneity sits inside the same number — and at n=5 each, an sd ratio of 1.7 is about one sigma.

**`my_experiments/tailnoise.py`** asks it without training anything: cut the tail into K disjoint
blocks, score each, and the spread across blocks is the sample noise at that block size; blocks
share no shots, so the full-tail sd is block_sd / sqrt(K).

Two things came out of the first run, one useful and one a correction to my own design.

**Useful: the shot sample dominates the ABSOLUTE score.** Across 5 disjoint blocks of 14, single
models range 0.9779 to 0.9957 — per-model sd 0.0029 to 0.0070, i.e. 0.0013 to 0.0032 at 70 shots.
Absolute S is not comparable across splits at anything like the resolution these experiments work
at, which is exactly why every arm is read PAIRED inside a salt. That practice is now measured
rather than assumed.

**The correction: two seeds are not two arms.** The tool first used seed pairs as a stand-in,
arguing they "differ only in initialisation". Measured, that proxy gives a pooled paired sd of
0.00125 at 70 shots against the 0.00025 the real base-vs-`ema` production difference shows — an
overstatement of about fivefold. A seed change is a LARGE perturbation whose effect varies shot to
shot; a loss change at fixed seed is a small one. The docstring now says so, and the mode prints a
warning; a `--artifact-b` mode was added that compares two artifacts differing in one setting, and
refuses if their salts or their train/val file lists disagree.

**The guard earned itself immediately.** Pointed at the shipped production artifact with a 5% tail,
the tool refused: 282 of those 352 shots were TRAINED or VALIDATED on. The shipped artifacts use
0.80 + 0.19 = 0.99, leaving exactly the 70 clean shots — which is why the original production pairs
used a 1% tail. It was not a choice. The 5%-tail runs free the tail by spending validation
(0.19 -> 0.15), so they are a valid paired comparison but NOT the shipped configuration.

Queued: `base` and `huber` artifacts at salt 0 under 0.80/0.15/0.05, kept rather than overwritten,
so the question can be asked of a real arm comparison.

**A lead, noticed in passing:** in block 0, `mlp2` scores 0.977925 where the other three seeds score
0.9825 to 0.9881 and where every other block has all four at 0.990+. One block, one seed, an order
more damage than typical — worth finding which shot does that and whether it is the same one the
tail-expert diagnostics keep landing on.

## `shot-tail-features` — the vessel integrals do NOT fix the tail. REFUTED 2026-08-21, for free

A22's missing-state branch: the last decile costs double because the current decays fastest there,
so vessel eddy currents are largest and most transient exactly where 21 instantaneous levels cannot
express them. The shipped submission blends three artifacts and the eddy-current integrals live only
in two of them, so the branch tests itself at zero cost — C1 decomposes the score onto frames and
trains nothing.

| artifact | total cost | last decile | share | density |
|---|---|---|---|---|
| `prod` | 0.00493 | **0.00119** | 24.0% | 2.35x |
| `vesseld` | 0.00504 | **0.00118** | 23.5% | 2.30x |
| `vesselp` | 0.00540 | **0.00119** | 22.0% | 2.15x |

**The absolute tail cost does not move: 0.00119, 0.00118, 0.00119.** The integrals do nothing there.

**And the shares would have said the opposite, which is the reusable part.** `vesselp`'s last decile
looks 2 points better as a SHARE — 22.0% against 24.0% — purely because its TOTAL cost is higher
(0.00540 against 0.00493). It is a worse model, so the tail is a smaller fraction of a bigger whole.
Comparing shares across models whose denominators differ reads improvements that are not there;
compare absolute costs.

**Consequence for the backlog.** The missing-state branch is refuted in its vessel-integral form,
which was the only form with features already built. That leaves `shot-tail-weight` as the one
remaining candidate for the measured 0.00106 bound, and raises it: the tail is expensive, the
expense is not the metric's doing (C7: sensitivity 0.98x, coefficient error 2.22x), not the basis's
(the ceiling is flat across the shot), and now not the vessel state's either.

**An unpredicted finding: both ENDS are expensive, not just the tail.** The first decile carries
15.1% of the cost over ~10% of the frames, a density of 1.48x, against 6.4-8.8% for the middle
eight. `my_experiments/frame_weight.py` already has `ends_weight()` beside `position_in_shot()`;
this says the two-sided form is the one to run, and that a tail-only step would leave half of a
measured concentration on the table.

## `huber` in production: ZERO. Six pairs, and the screen is now 0 for 2 — 2026-08-21

    n=6   mean +0.00000   SE 0.00017   t 0.0   signs 3/6
    per salt: +0.0 +5.0 -6.1 +1.4 +3.1 -3.2   (units 1e-4)

The term decomposition finishes it: **Consistency -0.00004 raw**, against **+0.00359** on the
screen. Not reduced — gone. Every other term is inside 0.00001 of S.

**The prediction that this tests was written before the data landed, and it holds.** When the
16-salt decomposition showed 82% of `huber`'s gain coming through Consistency, I wrote that this
undercut my own earlier argument: averaging four seeds also helps worst-case frames most, because
variance reduction is largest where variance is largest, so `huber` was working through a term the
ensemble already improves by another route. That is exactly what happened.

**The larger result is about the instrument, not the arm. The screen is now 0 for 2.**

| arm | screen | production |
|---|---|---|
| `ema` | +0.00101, 16/16, t=7.2 | **0.00000** |
| `huber` | +0.00088, 13/16, t=2.4 | **0.00000** |

Two changes have cleared the pre-registered gate and both measured exactly zero where it counts.
The distinction I drew after `ema` — variance-type changes are discounted by an ensemble, changes
to what every model does are not — survives as a mechanism but has stopped being useful as a filter,
because BOTH mechanisms turn out to route through terms the ensemble already covers.

**So the 16-salt screen on ONE net does not answer the question being asked.** It measures what a
change does to a single fit; the submission is a five-member ensemble, and the ensemble has already
taken the kind of gain a single fit shows. This belongs to backlog position 0 (`resolution`): the
screen needs to be replaced by, or gated behind, a production-configuration comparison. The cost is
roughly 4x a screen run, which is the price of measuring the right quantity instead of a correlate.

## I broke every fit in the fork for eleven minutes, and the runners' guard is what saved the data

Wiring the frame weight in, I added `frame_ends_strength: float = 0.0` to `TorchMLPModel` ABOVE
`grad_clip: float`, which has no default. Dataclasses reject a non-default field after a defaulted
one, so every fit raised `TypeError` at construction. Two queued lanes — the 12-pair `huber`
extension and the seq cosine arm — ran their whole salt lists in seconds, failing each run instantly.

**No bad data was written, and that was not luck.** Every runner extracts the `ensemble` row and
does `[ -z "$ROW" ] && { echo "  NO ensemble row"; continue; }` before appending. Twenty failed runs
produced twenty skips instead of twenty junk rows. A runner that had trusted the exit code, or
written whatever it parsed, would have poisoned two CSVs.

The fix: the two fields are now REQUIRED, not defaulted, with the reason written beside them —
`params.yaml` supplies both keys for all four MLP blocks, so defaults were never needed and were
only what broke the ordering rule. Verified by `load_params`, which constructs all six models; that
is what a run does at startup, so it is the check that matters rather than a bare import.

## A scale difference between MSE and Huber, chased down and mostly cleared — 2026-08-21

torch's `HuberLoss` is **0.5 x^2** in its quadratic region while `MSELoss` is **x^2**, so switching
loss halves the gradient on every ordinary frame. That looked at first like the confound this fork
has been bitten by before — a learning-rate change wearing another costume. Chased to the end, it
mostly is not, and the reason is worth keeping:

- **Adam is scale-invariant.** The update is m/(sqrt(v)+eps), so a constant factor on the loss
  cancels in the ratio. Both models use `torch.optim.Adam`. The halved gradient does NOT halve the
  step.
- **`weight_decay` is 0.0** for the MLP, so there is no coupled-L2 path for the scale to act through
  either — with torch's Adam the decay is added to the gradient, so a nonzero value WOULD have
  doubled in relative weight.
- **`grad_clip` is 1.0 and IS scale-sensitive.** Halved gradients cross the threshold less often, so
  `huber` and `mse` differ in how much clipping they receive. `models.py` already warns that a clip
  firing on most steps is a disguised rate cut; this is the same interaction seen from the other
  side, and it is a real if second-order difference between the two arms.

**One recorded sentence needs correcting.** The `hd4` write-up says "at delta 4.0 Huber is nearly
MSE". In VALUE it is nearly MSE/2, not MSE. In EFFECT, under Adam, it is nearly MSE — which is what
that argument needed — so the two-sided-optimum conclusion stands, but for a reason the entry did
not state.

**The clean control, if it is ever worth the run:** `huber_delta` at 1e6 is exactly MSE/2 with no
clipping whatever. Against `mse` it isolates the pure scale-and-clip effect from the outlier
handling. Not queued: `huber` measured zero in production, so this decides nothing that is still
open — it is recorded so the next person does not have to rediscover the asymmetry.

**Consequence for new code.** The sequence model's Huber is implemented SCALED BY 2, matching the
existing squared-error scale, so a `seq` huber arm differs from its control in outlier handling
alone and not in how often the gradient clip fires.

## `tail-noise` settled: the shot sample IS 73% of paired variance, and the 5% tail paid — 2026-08-21

Two artifacts, `base` and `huber`, trained at the SAME salt on the SAME split under 0.80/0.15/0.05,
then both scored on 5 disjoint blocks of ~71 shots from the 352-shot tail. Because the training is
identical across blocks, the spread of the paired difference across blocks is the SAMPLE component
alone, measured rather than argued.

| shots scored | paired sd |
|---|---|
| ~71 | 0.00079 |
| 352 | **0.00035** |

The observed TOTAL paired sd of the huber production comparison — six salts, 352 shots — is
**0.00041**. That decomposes the whole thing:

    sample     0.00035     -> 73% of the variance
    training   sqrt(0.00041^2 - 0.00035^2) = 0.000215

So at 70 shots the total would have been sqrt(0.00079^2 + 0.000215^2) = **0.00082**. The 5% tail
cut it to 0.00041 — **halved, for 44% more wall clock**. The change was worth making and the
original 86% claim was close: measured, it is 73%.

**And this explains the confounded comparison that made it look otherwise.** `ema`'s production
pairs showed a paired sd of 0.00025 at 70 shots, far below the 0.00082 predicted here — because
`ema`'s two arms are nearly the same model, so their difference barely varies shot to shot.
**Paired sd is a property of the ARM as much as of the sample size**, and only equal-sized changes
may be compared. This measurement uses base-vs-huber precisely because that is the comparison whose
noise it is meant to explain.

**The ceiling is now known.** With infinitely many scored shots the paired sd bottoms out at the
training component, **0.000215**. Shots cannot go below it; only PAIRS can, as 1/sqrt(n). With the
production shares only 5% of shots are free anyway, so shots are spent — 12 pairs is the remaining
lever, and it takes SE to 0.00012.

## `seq` gets weight averaging and Huber, and the EMA is PROVEN wired rather than assumed

`seq` had neither `loss` nor `ema_decay`. Both are now plumbed, with Huber **scaled by 2** so it
matches the squared-error scale and an arm differs from its control in outlier handling alone rather
than also in how often `grad_clip` fires.

The first smoke test returned byte-identical scores for `ema_decay=0.999` and the baseline, which is
exactly what a change that was never wired in would return. Rather than assume, the model now
REPORTS its win count — even when it is zero, because zero is the ambiguous case — and a second
check separated the two:

    ema_decay=0.9     averaged weights won 4 evaluation(s)    S 0.940815
    ema_decay=0.999   averaged weights won 0 evaluation(s)    S 0.937936  (identical to base)

Wired, and losing for a stated reason: at decay 0.999 the averaging window is 1000 steps against a
1200-step smoke run, so the average still carries the initialisation. On a real 68000-step fit the
window is a fraction of the trajectory.

## `shot-tail-weight` refuted — the backlog's best-supported entry, dead at 5 pairs — 2026-08-22

Backlog position 1, the only open entry with a MEASURED upper bound (0.00106, 27% of everything
left, concentrated in 10% of the frames). Run two-sided, because the cost decomposition had just
shown both ENDS of a shot are expensive, not only the tail: 24.0% of the cost in the last decile and
15.1% in the first, against 6.4-8.8% for the middle eight.

    ends   n=8   mean +0.00002   SE 0.00003   t 0.5   signs 4/8
    per salt: -0.3 +0.3 -0.6 -0.5 -0.8 +2.1 +0.7 +0.4   (units 1e-4)

**The effect is zero.** Not "fell short of the gate" — absent, against a measured bound of +0.00106.

*Corrected twice the same day, which is the part worth keeping.* At five pairs this read -0.00004
with 1/5 signs and every term slightly down, and was written up as "a uniformly tiny negative". The
sixth salt came in at +2.1 and moved the mean to zero; the eighth left it at 4/8, an exact
coin-flip. Five points of a null always have a shape, and describing that shape as a result is the
error — with |t| < 2 the answer is "zero", and the picture the points happen to make says nothing.
The refutation never moved (a bound of 0.00106 is missed either way), but the resolution claim
shrank with the sign: SE went 0.00002 -> 0.00003, so this arm is about five times finer than
`huber`'s, not eight.

Stopped at 8 of 12 pairs deliberately: the arm was settled, and its 20 job slots were slowing a
submission build that was actually wanted.

**The null is trustworthy because the setting was verified to land, not assumed.** The saved
artifact's own `params_yaml` carries `frame_ends_strength=1.5, edge=0.1` on all four MLPs. This
project has now produced two silent no-ops (a defaulted dataclass field that broke every fit, an EMA
that returned byte-identical scores), so a null is only worth recording once the knob is proven to
have moved.

**Why the resolution is so much better here than for `huber`, and it is a reusable point.** SE is
0.00004 at 6 pairs against `huber`'s 0.00017 at 6 — four times finer for the same protocol. Paired
sd is a property of the ARM, not of the sample size: `huber` replaced the loss everywhere, so its
effect genuinely varied from split to split, while this reweights 20.4% of frames by 1.5 with the
mean normalised out, which is a small and uniform perturbation. **The instrument is not one number;
it is one number per kind of change.**

**Why weighting cannot have worked, in hindsight that the earlier measurements already supported.**
A22 measured **55% of the last decile's residual as SHARED across the four seeds**. Shared residual
is bias — of this model-and-feature class — not variance. Re-weighting frames changes how much
attention the fit pays them; it does not supply information the inputs do not carry. So it can only
move error between the middle and the ends, which is exactly what a uniformly tiny negative looks
like. The same reasoning already killed `shot-tail-features` from the other side: adding the vessel
integrals left the absolute tail cost at 0.00119 against 0.00119.

**What this closes.** The tail family is now refuted in both of its separable mechanisms —
under-representation (this) and missing state (`shot-tail-features`) — leaving only label noise,
which A22 flagged as untestable by the measurements available and which no weighting or feature can
address. The 0.00106 bound stands as real, and unreachable by anything in this file.

## The board stayed at 0.9938, and that is the ceiling of the INSTRUMENT, not a refutation — 2026-08-22

The 2026-08-22 submission shipped one change: the GRU's cosine schedule, measured in production at
**+0.00009 with 3 signs of 3**. The board came back **0.9938, unchanged from 2026-08-19**.

That is not evidence against the change. **The board prints four decimals, so its resolution is
0.0001** — an effect of 0.00009 sits exactly on the boundary and may or may not move the last digit.
0.9938 is consistent with the measurement; it simply cannot distinguish it from zero.

**And that is the finding worth acting on.** Local measurement now resolves 0.00003 (the frame-weight
arm) while the LEADERBOARD resolves 0.0001. The remaining local headroom is 0.0039 spread over seven
functionals, and the changes actually available are worth 0.0001-0.001 each. So for Challenge 1 we
have entered a regime where:

- a genuinely good change moves the board by one or two units of the last printed digit;
- a good change and a useless one are indistinguishable there;
- and the only trustworthy signal is the local paired protocol, which costs 9-36 h per answer.

**The day's production ledger, all measured in the shipped configuration:**

| change | production | verdict |
|---|---|---|
| `huber` | +0.00000, 5/11, 12 pairs | dead |
| `ema` | +0.00000, 2/5 | dead |
| `one-vs-four` | −0.00106, 0/6 | decisively worse |
| `shot-tail-weight` | +0.00002, 4/8 | dead, against a 0.00106 bound |
| `seq-cosine` | +0.00009, 3/3 | real but below board resolution |

Five changes, one survivor, and the survivor is too small for the scoreboard to see.

**CORRECTION, same day, once the leaderboard was read properly.** The paragraph above argued for
abandoning Challenge 1 as finished. That was the wrong conclusion from the right numbers, and the
error is worth naming because it is a general one.

First place is **0.9944**; we are at **0.9938**. The gap is **+0.0006** — six times the board's
printed resolution, and well inside the 0.0039 of local headroom. Someone has already taken it, so
it is reachable in fact and not only in principle.

What I did was measure the resolution for detecting ONE change (0.0001, marginal) and conclude the
GOAL was out of reach. Those are different quantities. The goal is +0.0006, which is comfortably
detectable; it just cannot be delivered by one change. Three or four arms worth +0.0002 each close
it, and +0.0002 is exactly the size the local paired protocol resolves at t = 2 with 12 pairs.

The strategic argument for MAST stands on its own arithmetic — 0.43 unclaimed against 0.0039, and a
noiseless board — but "Challenge 1 is finished" was false. It is 0.0006 from first place.

## `coef-expansion` refuted, and the refutation is a lesson about sweeps — 2026-08-23

Re-opened from D5, which had measured the per-shot optimal coefficient factor at median **1.0095**
with **two thirds of shots wanting to be pushed AWAY from the fold mean** — and then closed the
family because its oracle, +0.00033 of S, sat below the 0.0013 a paired run then resolved. Only
SHRINKAGE had ever been run (-0.0005); expansion never had. The instrument now resolves 0.00017 and
the test costs no training at all: `FUSION_COEF_LAMBDA` scales the PCA coefficients at decode time,
and the basis is centred, so that scales each frame's deviation from the mean.

**The sweep looked like a find.** On the shipped artifact's 70 clean tail shots, eleven values gave a
clean unimodal curve peaking near D5's own median:

    0.9900 -0.000340 | 1.0000  0.000000 | 1.0050 +0.000075 | 1.0095 +0.000058
    1.0200 -0.000293 | 1.0400 -0.002230 |  (components 20+ only: 1.1000 +0.000035)

Shrinkage negative, expansion positive, peak next to the predicted place. Every part of it agreed
with the mechanism.

**Cross-validation killed it.** `my_experiments/lambda_cv.py` splits the 352-shot clean tail of
`artefacts/tailnoise/base_s0.joblib` into two disjoint halves, round-robin through the sha1 order,
picks lambda on one and reports what it earns on the other:

| fit on | lambda* | in-sample | **out-of-sample** |
|---|---|---|---|
| half 0 | 1.0050 | +0.000054 | **-0.000300** |
| half 1 | **1.0000** | 0 | 0 |

**mean out-of-sample: -0.000150.** The two halves do not agree on the direction: half 0 peaks at
1.005, half 1 peaks at 1.0000 and falls monotonically as lambda rises. An optimum that does not
survive a disjoint half of the SAME tail is not an optimum.

**The reusable part.** A sweep of N values always returns a positive best, because the best of N
noise draws is positive by construction — and here that bias (+0.000075) was LARGER than any effect
present. The first sweep was not wrong about its arithmetic; it was answering "what is the best of
eleven numbers" when the question was "does a lambda exist". Any future free sweep gets the same
two-column treatment: what the sweep reports, and what it is worth on shots the sweep never saw.

D5's original closure stands, and its sentence turns out to have been exact: *"there is nothing here
for a confidence signal to do, however good it is."*

## `report()` printed four decimals too — the same bug, the third time — 2026-08-23

`report_comparison` was widened to six decimals on 2026-08-21, but `report()` — the single-model
block — was missed, because nothing parsed it until the lambda sweep, which scores ONE model and so
never reaches the comparison table at all. Three sweep runs scored perfectly and recorded nothing,
for a second reason: the runner grepped for the comparison table's `ensemble` row, which only prints
when more than one model is scored.

Both are now fixed. Worth noting how it was caught: not by the runs failing — they succeeded — but
by the recurring health check's "rows written, not just headers" test, added after 2026-08-22's
2 h 40 m of lanes that looked alive and wrote nothing.

## `seq-cosine` CONFIRMED in production — the campaign's first survivor — 2026-08-23

    cos   n=6   mean +0.00011   SE 0.00003   t 3.9   signs 6/6
    per salt: +0.4 +2.1 +0.3 +1.4 +1.0 +1.2   (units 1e-4)
    R2_psi +0.00000 | R2_qb +0.00001 | 1-D_LCFS +0.00002 | Cons +0.00007   (as S)

Final at eight pairs: **+0.00010, SE 0.00002, t = 5.1, signs 8/8**, contributing through
Consistency (+0.00007) and D_LCFS (+0.00002) with R2_psi at exactly zero. Stopped at 8 of 12
deliberately — at t = 5.1 the remaining four pairs could not change the verdict or the shipping
decision, and the `seq-ema` lane behind it had been idle for ten hours.

Six salts of six, t = 3.9. Against a day in which `huber`, `ema`, `one-vs-four` and
`shot-tail-weight` all measured zero or worse in production, this is the one that transferred.

**Why it transferred, stated before the pairs landed and now confirmed.** `ema` and `huber` died
because the four-seed MLP ensemble already does what they do — average away trajectory noise, and
damp the worst frames. **`seq` is a single member with no ensemble of its own**, so nothing was
already doing this for it. The dilution arithmetic also came out right: one member of five, a screen
effect of +0.00111, so expect roughly +0.0002 in the ensemble; measured +0.00011.

**Where it was found is the part worth repeating.** Not in a treatment arm — in the CONTROL arm of
the `seq-dropout` experiment, whose validation curve swung ~30% between adjacent evaluations
(0.032029, 0.043272, 0.035239, 0.043711). That volatility at convergence is what a constant learning
rate looks like, and both `seq` and `mlp` shipped `lr_schedule: none`. The horizon came from the same
control curve — best at step 73000, stopped at 83000 — so `lr_t_max_steps=80000` traverses the real
curve instead of the cosine's flat beginning, which is the trap `models.py:384` records.

**What it implies for the backlog.** `seq` is a full production member that had never been swept at
all, and its FIRST swept knob paid. Its remaining knobs — `learning_rate`, `seq_hidden`,
`seq_layers`, `patience_steps` — are untested, and `ema_decay` and `loss` were only plumbed into it
on 2026-08-21. The same no-dilution argument applies to every one of them.

## `jac` refuted in production — the first `recheck` candidate, and it was right to be closed — 2026-08-23

`loss.jacobian_frames` 300 -> 1200: estimate the loss metric on four times the frames. It was the
strongest entry in the `recheck` section — screened at **+0.00044 against a 0.00065 gate**, i.e.
dismissed as indistinguishable rather than measured as zero — and it passed the ensemble filter,
because a better-estimated metric changes what every member optimises and averaging cannot
substitute for that.

    jac   n=6   mean -0.00000   SE 0.00009   t -0.0   signs 3/6
    per salt: +0.5 +2.7 -2.8 -2.4 +2.3 -0.4   (units 1e-4)
    R2_psi -0.00001 | R2_qb -0.00001 | 1-D_LCFS -0.00002 | Cons +0.00003   (as S)

Exactly zero, three signs each way. At SE 0.00009 a true +0.0002 would have shown t = 2.2, so the
effect these six pairs rule out is precisely the size the arm was re-opened to find.

**A confound was checked BEFORE the result was known, which is the only time such a check is worth
anything.** One `jac` run reached 90000 seq steps where the baseline usually stops near 83000, so
the arm might have been buying its effect with extra steps rather than with the metric. Mean stop
step over five runs each: **base 53500, jac 53200**. No systematic lengthening; that run was simply
a long one.

**What it says about the `recheck` idea itself.** The premise was right — a verdict reached with an
instrument coarser than the effect is not a measurement — but this particular verdict turns out to
have been correct anyway, for a reason it did not state. "Below the gate" became "measured zero".
That is the recheck working as intended: it converts an unknown into a fact, and the fact is
allowed to be a null.

Stopped at 6 of 12 pairs under the convention recorded this morning: the answer was in, and the
lane was needed for `recheck-a15`, whose answer is not.

## `seq-ema` refuted — and it CONFIRMS a rule that was written down before it closed — 2026-08-24

    ema   n=5   mean +0.00001   SE 0.00002   t 0.7   signs 2/5
    per salt: +0.8 +0.1 +0.0 -0.1 -0.1   (units 1e-4)

**The mechanism fired and the gain did not arrive.** The averaged weights won 99 and 123 evaluations
per run — this is not a silent no-op, and the model reports the count precisely so that "zero wins"
and "never wired in" can be told apart. It is wired, it is active, and it is worth nothing here.

**The prediction, recorded 2026-08-24 BEFORE these pairs landed.** The transfer rule up to then was:
a change measures zero when the ENSEMBLE already does its work — that killed `ema` and `huber` on the
MLPs, while `seq-cosine` transferred because `seq` is a single member with nothing to dilute it. When
`seq-ema` began fading on the same member cosine had just paid on, the rule was widened in advance:

> A change measures zero when its work is already being done by ANYTHING already in the
> configuration — the ensemble, or another change measured earlier.

Cosine and EMA both stabilise where the trajectory ends. Cosine was measured against a base without
it and paid; EMA is measured on top of it and does not. The pre-registration named the outcome that
would refute the widening — a positive `seq-ema` at six pairs — and that outcome did not occur.

**The operational consequence is larger than the arm.** Every measurement from here is an INCREMENT
on the accepted stack, not a standalone value. Screen effects cannot be added up: a set of arms that
each screened at +0.001 does not close a 0.0006 gap if they all work through the same mechanism.

## First pairs run high, and I have over-read them repeatedly — 2026-08-24

A pattern worth naming, because it has shaped three write-ups this week:

    seq-cosine  +0.4 +2.1 ... -> settled at +0.00010
    h512        +0.6 +1.5 -0.1 +0.7
    a15-width   +3.9 +0.9 +0.1
    seq-ema     +0.8 +0.1 +0.0 -0.1 -0.1 -> zero

The first salt is routinely the largest, and the arm then subsides. This is not a property of these
arms; it is regression to the mean under a large paired sd, made visible because a high first point
is what prompts a comment while a low one does not. **The first pair carries almost no information**,
and reporting its shape — however encouraging — is the same error as describing the shape of five
null points. Useful reading starts around four to six pairs.

## The lanes were discarding five sixths of every model they trained — 2026-08-25

`lane.sh` wrote each fit to `artefacts/lanes/<lane>.joblib`, one path per LANE, so every salt
overwrote the one before it. Six salts of four MLPs and a seq were trained per arm and exactly one
survived. Found while acting on the user's idea of submitting a blend across everything already
fitted — which is when it became obvious that "everything already fitted" was one artifact per lane,
not twenty-four.

Fixed: the path is now `artefacts/lanes/baseline_<row>.joblib`, one per salt AND arm, which is also
the naming `salts:a+b+c` resolves, so any future blend needs no extra training whatever.

**Why the discarded models mattered, and it is not sentiment about wasted compute.** Diversity was
measured on two axes and bounded on both — seeds correlate at **0.56**, feature sets at **0.75**,
which is worse — and `ideas.md` says in as many words that these numbers bound only that family:
a genuinely different representation "has never been tried and is not bounded". Salt is exactly such
an axis. Each salt is a different 80% of the shots, so its members share neither initialisation nor
training data, and nothing in this fork has ever measured how far apart that puts them.

**The nine that did survive, assembled as `artefacts/mega/`:** 4 distinct salts (0, 4, 5, 8) across
3 feature sets, about 50 MLPs and 7 seq, against the shipped trio's 12 MLPs and 1 seq on one salt.

**And it cannot be scored here — checked, not assumed.** The union of what those nine trained or
validated on covers **all 7041 shots** of the config; zero are unseen by all of them. So there is no
honest local number, and the leaderboard is the only instrument that can read this. The public test
split is a separate config no model has touched, so the blend is legitimate there.

## The mega-blend PAID: 0.9938 -> 0.994081 on the board, +0.0003 — 2026-08-25

The user's idea — submit a blend of everything already fitted, since testing it locally is
impossible but submitting it is cheap — returned **+0.0003**, three times the confirmed `seq-cosine`
gain, from nine artifacts across four salts with **no new training at all**. The gap to first place
(0.9944) is now **0.00032**.

**Split diversity is real, and it was the one axis never measured.** Seeds correlate at 0.56, feature
sets at 0.75; which 80% of the shots a member trains on had never been varied, and `ideas.md` says
in as many words that the measured bounds cover only their own family. This is the first direct
evidence on that axis, and it is worth more per unit of effort than anything measured this week —
because the effort was zero.

## The leaderboard's detailed breakdown, and it reorders everything — 2026-08-25

The submission returned a full per-term and per-scalar table from the TEST set. Recomputing the
composite from it gives 0.994081 to the last digit, so the metric is understood exactly.

| term | value | unclaimed S |
|---|---|---|
| R2_psi | 0.999832 | 0.000092 |
| R2_qb | 0.996539 | 0.000519 |
| 1-D_LCFS | 0.988773 | **0.001123** |
| Consistency | 0.979077 | **0.004185** |

And inside Consistency, at 0.20/7 = 0.02857 of S per unit of R2:

| scalar | R2 | unclaimed S | vs the 0.00032 gap |
|---|---|---|---|
| **li** | 0.960156 | 0.001138 | **3.6x** |
| **kappa** | 0.965434 | 0.000988 | **3.1x** |
| R_axis | 0.978202 | 0.000623 | 1.9x |
| volume | 0.984592 | 0.000440 | 1.4x |
| Z_axis | 0.984992 | 0.000429 | 1.3x |
| tri_bot | 0.989229 | 0.000308 | 1.0x |
| tri_top | 0.990932 | 0.000259 | 0.8x |

**`li` and `kappa` together hold 0.002126 — 6.6x the gap to first place.**

**This explains the week.** Nine production arms were general-purpose knobs — huber, ema, dropout,
width, hidden size, a better-estimated metric — and every one of them acts on all seven scalars
alike. The loss is concentrated in TWO of the seven. We were optimising an average whose damage is
not spread.

**And it is a FITTING loss, not a representation one — C2 measured that on 2026-08-15 and the entry
was left unused.** "Of the 0.0055 remaining, 0.0045 is reachable by FITTING and 0.0010 is the basis…
The model's psi is essentially perfect as pixels — 0.9998 against a ceiling of 1.0000 — and the
scalars derived FROM it are off by 2-4% of variance. The basis can produce those scalars to 0.99+;
the fit does not. So this is not representation and it is not capacity: it is that the loss and the
score disagree about where on the map the error matters." `kappa`'s basis ceiling is **0.9908**; the
board says we achieve **0.9654**. Twenty-five thousandths of kappa's R2 are lost to fitting alone.

**Which re-prices `aux`, demoted to position 8 on an argument that is wrong in an important way.**
It was ranked down because "the loss already up-weights the geometry directions 1434x". True — but
the Jacobian metric knows which DIRECTIONS in flux space matter, components 20-49 collectively. It
does not know that of the seven SCALARS the score reads, two carry two thirds of the loss. Teaching
`li` and `kappa` as auxiliary outputs targets exactly the thing the metric weighting cannot express.

## The surrogate gate CLOSED both arms — and it is why the gate was written that way — 2026-08-25

`aux-surrogate` would put a small net `g(c) -> (kappa, li)` on top of the predicted PCA coefficients,
so the auxiliary gradient could only reach a lower loss THROUGH them. L3 pre-registered the gate:
fit `g` on coefficients perturbed at the model's own error scale, then judge it on held-out frames by
how well it predicts the ACTUAL change of the functional along the real error direction
`dc = c_pred - c_true`. **Proceed only if kappa's correlation exceeds 0.90.**

Measured on 1183 frames from 6 clean-tail shots of `lanes/a`, half fitted and half held out:

| method | kappa | li |
|---|---|---|
| linear (what the loss effectively does) | **0.732** | 0.598 |
| surrogate | **0.610** | 0.780 |
| constant (control) | nan | 0.000 |

**0.610 against a required 0.90. Both arms close, and no training code was written.**

**The reason this gate exists is in its own numbers.** The surrogate reached a training loss of
**0.00001** on standardised targets — it predicts the VALUES almost exactly — and its GRADIENT on
kappa is still worse than the linearisation's. A gate on value R2 would have passed it brilliantly
and sent two arms to production to teach the model the wrong direction, invisibly. L3 wrote "gate on
gradient fidelity, not value R2" before any of this ran, and that sentence saved the days.

**One post-hoc observation, labelled as such rather than used to move the goalposts.** On `li` the
surrogate is the BETTER of the two, 0.780 against 0.598, while on kappa it is worse. The
pre-registered threshold was on kappa and it failed, so the arms are closed. But the asymmetry is
real and it is the second time these two scalars have split: D1 measured kappa **+0.0119** and li
**−0.0047** from the same head. Whatever helps one appears to hurt the other, and any future entry
aimed at "kappa and li" should expect to be two entries.

## `seq-weight` is NOT established — two artifacts, opposite answers — 2026-08-25

The first sweep looked like a free find: both halves of one artifact's clean tail agreed that `seq`
deserves MORE than its shipped 1/5, and the two-column cross-validation that killed the lambda sweep
returned **+0.000174** out of sample. A second artifact says otherwise:

| artifact | salt | seq config | mean out-of-sample gain |
|---|---|---|---|
| `lanes/a` | 5 | cosine | **+0.000174** |
| `lanes/c` | 8 | stock | **−0.000039** |

**The optimal weight is a property of the fit, not of the ensemble design**, at least across these
two. Nothing ships on this.

**A confound that must not be resolved by assertion:** the two differ in BOTH salt and configuration
— `a` carries the confirmed cosine and `c` does not. "A better `seq` deserves more weight" is a
coherent story and it is not evidence. Separating them needs a third sweep on an artifact with
cosine at a different salt (`lanes/e`, salt 4) and one without at the same salt as `a`
(`lanes/j`, salt 5, jac and no cosine).

## Two capacity arms closed to clear the deck for `aux` — 2026-08-26

`h512` (`seq_hidden` 256 -> 512), stopped at 10 of 12 pairs: **+0.00004, SE 0.00003, t = 1.6,
7/10**. Its t wandered 2.8 -> 1.3 -> 2.0 -> 1.6 over five readings, which is what an effect sitting
at the edge of resolution looks like rather than a signal about to declare itself. Two more pairs
could not change that, and its GRU fit is 1.48x the baseline's (136 min against 92), so every
comparison it takes part in costs half again as much.

`wide+dropout` (MLP [1024,1024] with dropout 0.02), stopped at 1 pair on the user's instruction to
focus. Its one reading was +0.00025 and is not reported as anything: four first pairs in a row have
run high and settled — cosine +2.1 survived, `a_wide` +3.9 died, `h512` +1.5 hovers, `seq-ema` +0.8
went to zero.

**What the capacity family leaves behind.** `a_wide` (MLP width, no dropout) went +0.00039 -> zero
over five pairs, so A15's "this model is not short of parameters" survives its re-run at
`grad_clip: 1.0` — the precondition its own author asked for. `h512` says the GRU may be slightly
short, at a size the instrument cannot resolve. Neither is where the test set says the loss is.

## Where a production run's time actually goes, measured — 2026-08-26

Asked whether a 4090 or 5090 would help. Measured rather than guessed, on a completed base run at
the production shares:

| phase | time | share | bound by |
|---|---|---|---|
| reading shots | 1.3 min | 0.6% | CPU |
| 4 MLP fits | 10.3 min | 5.0% | GPU, at 18% SM |
| **seq fit** | **131.0 min** | **62.8%** | **GPU** |
| scoring + predict | 66.0 min | 31.6% | CPU |
| total | 208.6 min | | |

GPU utilisation during a seq fit is **81% mean over 40 samples, range 79-86%** — the card really is
the bottleneck there, and this corrects the older "18% utilisation" note, which was measured on an
MLP fit and does not describe the seq one.

**An infinitely fast GPU would still leave 32% of the run.** By Amdahl: 2x on the GPU gives 1.5x
overall, 3x gives 1.8x, and the ceiling is 3.1x. A 4090 is 5.3x the V100 on paper FP32, but a GRU's
recurrence is sequential in time with small matrices (256x256 at batch 32), so it is latency- and
occupancy-bound rather than FLOPS-bound, and memory bandwidth rises only 1.12x. Two to three times
on the seq fit is the realistic expectation, i.e. 1.5-1.8x on the run.

**For this workload more cards beat a faster card.** We run two or three independent lanes sharing
one GPU, so a second V100 is close to 2x on experiment throughput, and unlike a faster card it also
hides the 32% CPU tail: one lane's scoring overlaps another lane's fit.

**Two free levers exist and were not spent.** Scoring is 32% of the run on 14 processes against a
30.7-CPU quota. And the fits run in FP32 while the V100's tensor cores do ~125 TFLOPS in FP16
against 15.7 in FP32 — mixed precision could be worth multiples on the same hardware, at the risk
that a half-precision recurrence converges differently, which one run would settle.

Decision: stay on the current machine.

## The Sobolev surrogate failed on MY training, not on the problem — checked, 2026-08-26

Trained on derivatives as well as values, the surrogate's gradient fidelity got WORSE, not better:
kappa correlation 0.610 value-only, then 0.199 / 0.236 / 0.331 at Sobolev weights 0.3 / 3.0 / 30.0.
Teaching a net to match derivatives made its derivatives worse, which needed an explanation before
it could be written down as a result.

**Two candidates, and the data did not separate them.** Either the finite-difference targets were
discretisation noise — `extract_lcfs` finds a contour numerically and a small coefficient change can
move which contour is found — or the training simply had not converged, since its loss went from
0.00001 to 0.03-0.53.

**The check: compute df/dc at three step sizes and correlate.** A smooth functional gives the same
derivative at each; a jumping contour gives unrelated numbers.

| steps compared | kappa | li |
|---|---|---|
| 0.5x vs 1.0x | **0.916** | 0.895 |
| 1.0x vs 2.0x | **0.923** | 0.895 |
| 0.5x vs 2.0x | 0.832 | 0.759 |

and the magnitude barely moves — |df/dc| median 0.2322, 0.2375, 0.2386 across a 4x range of step.
**The targets are signal.** So the noise explanation is refuted and the fault is mine: batch cut 4x
for the Sobolev steps while the step count stayed at 6000, on the CPU, with `create_graph` doubling
the backward cost per output.

**Worth the twenty minutes it cost.** The tidy conclusion — "the functional is not smooth enough at
the model's error scale for a surrogate to help" — was available, plausible, and wrong, and a whole
family would have closed on it.

Re-running with the training fixed: on the GPU, 60000 steps, a value-only warmup for the first
fifth so the net is not chasing slopes of a function it has not yet fit, and the value and gradient
losses reported separately so "did not converge" can never again be mistaken for "cannot work".

## The surrogate closes for a BETTER reason: the functional is already linear — 2026-08-26

Third pass on the gate, with the overfitting diagnosis acted on: 3000 frames instead of 300, hidden
64 instead of 128, weight decay 0.01, and the correlation on the FIT frames printed beside the
held-out one so the gap is shown rather than argued.

| | kappa | li |
|---|---|---|
| linear | **0.905** | 0.874 |
| surrogate | 0.787 | **0.907** |
| surrogate, on its own fit frames | 0.781 | 0.910 |

**Overfitting is gone** — 0.781 fitted against 0.787 held out. The diagnosis was right and the cure
was data.

**And the earlier numbers were themselves small-sample artefacts.** The linear predictor read 0.732
on 592 frames and reads **0.905** on 3000. L3's threshold — "above 0.90, against the linearisation's
0.77" — was set against a baseline measured under the same shortage, so the bar and its reference
have both moved.

**The real conclusion, and it is worth more than the arms it closes.** The nonlinear surrogate does
not beat a linear fit on kappa: 0.787 against 0.905. Over the range the model's errors actually
span, **the map from coefficients to kappa is close to linear** — so a nonlinear surrogate has
nothing to add there, and the linear version is not a new mechanism at all: it is exactly what the
Jacobian metric already carries.

**On li it is the other way**: the surrogate reaches 0.907 against linear's 0.874, so li does have
usable curvature. That is the **fourth** independent time these two scalars have split — D1 (+0.0119
kappa, −0.0047 li), gate 1 (0.610/0.793), gate 2, and now this. They should not be treated as one
target.

**Sobolev remains worse (0.476) and still does not converge** — value loss 0.129 after 20000 steps
on 3000 frames. With the data problem removed, that is now a property of the objective rather than
of the sample: matching per-frame derivatives is simply hard to optimise here.

**What it implies for what is running.** If kappa is linear in the coefficients, the Jacobian metric
is the right object and the only open question is its WEIGHTS — which is `cons-reweight`, already in
production and at +0.00011 on its first pair.

## `aux` refuted — the mechanism works and the trade loses — 2026-08-26

    aux   n=3   mean -0.00016   SE 0.00005   t -3.1   signs 0/3
    per salt: -0.6 -2.2 -1.9   (units 1e-4)
    R2_psi -0.00001 | R2_qb -0.00007 | 1-D_LCFS -0.00003 | Cons -0.00005   (as S)

Stopped at 3 of 6 pairs: at t = -3.1 with every salt negative the sign does not turn, and the lane
was wanted for `cons-reweight`, whose answer is unknown.

**This is not a null. The mechanism did exactly what it claims, and the exchange was still a loss** —
which is a more useful outcome than a flat zero, because it prices the idea rather than dismissing
it.

| | salt 0 | salt 1 | salt 2 |
|---|---|---|---|
| kappa | +0.0029 | -0.0013 | +0.0004 |
| **li** | **+0.0031** | **+0.0051** | **+0.0026** |

li rose on all three salts and kappa on two of three: teaching the two scalars directly does move
them. What it costs, on salt 0:

    R_axis   0.976245 -> 0.973885  (-0.0024)
    Z_axis   0.986569 -> 0.984157  (-0.0024)
    tri_top  0.991832 -> 0.991207  (-0.0006)
    q95/betaN                       -0.00047 raw, -0.00007 of S

**The losses on R_axis and Z_axis match the gains on kappa and li almost exactly, and q95/betaN is
lost on top.** The scorer weights the seven equally, so a one-for-one trade inside Consistency is
worth precisely zero — and the qb loss turns zero into negative.

**This is the objection written for `cons-reweight` BEFORE either ran**, arriving early and through
a different door: *"the five it takes weight FROM must not lose more than the two gain."* In `aux`
the weight is taken implicitly — capacity and attention — and the taking cost more than the giving
paid.

**Which sharpens what `cons-reweight` is testing.** It moves weight strictly INSIDE the Consistency
block and never touches q95/betaN, where a quarter of `aux`'s loss came from. If the remaining
three-quarters — the one-for-one trade among the seven — is also unavoidable, it will fail the same
way, and the two arms together will have measured that the seven scalars share one budget that
cannot be reallocated. That is a real finding either way.

## `cons-reweight` — NOT refuted, stopped early at a weak positive — 2026-08-27

**Correction, same day, after the user pushed back on the label.** This was written up as
REFUTED. It is not. +0.00003 with SE 0.00004 is an estimate SMALLER than its own error, t = 0.8,
signs 3/6 — by this project's own rule that reads "indistinguishable", not "refuted", and the
point estimate is POSITIVE. What actually happened is that I judged six more pairs not worth nine
hours and then recorded a judgement as a measurement.

The arithmetic that made stopping defensible, which is what should have been written instead: at
12 pairs SE falls to 0.000028, so the current +0.00003 would reach only t = 1.1. Clearing t = 2
needs a true effect of 0.000057, nearly double what is seen. Resumed to 12 pairs anyway once the
GPU was free — the estimate being positive is worth the hours when nothing else wants them.

    x3   n=6   mean +0.00003   SE 0.00004   t 0.8   signs 3/6
    per salt: +1.1 -0.1 -0.8 +1.7 -0.7 +0.7   (units 1e-4)
    R2_psi -0.00000 | R2_qb -0.00000 | 1-D_LCFS -0.00000 | Cons +0.00004   (as S)

`loss.cons_split=kappa=3,li=3,rest=1` — kappa and li at 1.91x the equal share, the other five at
0.64x, with the block's total weight against the flux untouched.

**The construction did exactly what it was built to do**, which is why the null is informative
rather than empty:

    kappa            +0.001450
    li               +0.000384
    the other five   -0.000428
    net across the seven  +0.001406  ->  +0.000040 of S

and **q95/betaN were not touched at all** (-0.00001 raw), against `aux`'s -0.00047. That was the
whole design difference: move weight strictly INSIDE the Consistency block, since a quarter of
`aux`'s loss had come from the neighbouring one. It worked, and it was not enough.

**Three mechanisms, one answer.** `aux` taught the scalars directly and lost 0.00016; the surrogate
family closed because kappa is already linear in the coefficients; `cons-reweight` moved the metric's
own weights and gained 0.00003. **What is added to two is subtracted from five, and no total value
is created.** The seven share one fitting budget.

**So the 0.0042 of unclaimed Consistency is not reachable by any weighting** — not through the
targets, not through the metric, not through a learned surrogate of the functional. Reaching it needs
a new input or a different representation, which is a different kind of work from anything tried
this week.

**What is left untested in this family, and it is now more interesting than before.** All three arms
moved weight between scalars whose headroom differs two- to three-fold, and all three found a ~1:1
exchange. Between q95 (R2 0.9967, 0.0033 unclaimed) and li (0.9602, 0.0398) the ratio is **12x** —
a regime none of them probed. `cons-reweight-all-9` is the only place the "a nearly saturated scalar
has little marginal return" hypothesis can still be true.
