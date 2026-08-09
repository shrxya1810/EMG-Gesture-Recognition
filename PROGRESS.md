# Project Progress Report

**Design and Development of a Real-Time EMG-Based Forearm Gesture Recognition System**

Shreya Agarwal (230907122)
School of Electrical Engineering, Manipal Institute of Technology
Guide: Ms. Vidya Kamath · Minor Specialization: Coursera Data Science

---

## 1. Executive summary

The project has a complete, working, reproducible sEMG gesture-recognition
pipeline on NinaPro DB5, with a measured and defensible accuracy figure for the
first time.

The headline result is **89.0% within-subject accuracy** (Extra Trees,
time-domain features, no preprocessing, probability-smoothed over 5 windows;
90.8% at k=7) and **65.5% cross-subject accuracy** (SVM with
subject-independent normalisation).

An audit of the earlier pipeline found that the previously reported ~84% figure
was not reproducible. It came from a leaky evaluation protocol applied to the
**wrong six gestures**, produced by code that crashes on the actual dataset.
Section 3 documents this in full. The corrected, honestly-measured baseline was
74.2%; targeted improvements raised it to 88.1%.

Four of five Semester V primary objectives are now complete. The preprocessing
stage ablation (§5.10) and the Table I grid search (§5.11) have both been run,
and inference latency has been measured for the first time (§5.12). What remains
in Semester V is the 1-D CNN, and the literature-sheet artifact. Two of five
*advanced* objectives — embedded deployment and the MyoWare hardware prototype —
are deferred and out of scope for this report, and are marked as such in §6
rather than dropped.

Two of the new results contradict the synopsis rather than confirming it. The
preprocessing chain it specifies makes accuracy worse on this corpus — on both
axes, for every model — and the grid search it specifies is worth +3.3 points on
SVM and nothing anywhere else. Both are documented with matched controls in
§5.10 and §5.11. The chain has now been dropped from the default and retained as
opt-in (§5.10a), which is what moved the headline from 88.1% to 89.0%.

---

## 2. Where the project started

The initial repository contained six modules and 251 lines of Python:
`data_loader`, `preprocessing`, `segmentation`, `features`, `build_features`,
`train`. It reported:

| Model | Reported |
| ----- | -------- |
| "Linear SVM" | ~72% |
| Random Forest | ~82% |
| Extra Trees | ~84% |

No `features.csv`, no saved models, and no metrics files existed in the
repository to support these numbers. They were produced by a local run that was
never committed.

---

## 3. Audit findings

### 3.1 The gesture selection was wrong

This was the most consequential defect.

The synopsis (§V.A) specifies six gestures: open hand, closed fist, wrist
flexion, wrist extension, forearm pronation, forearm supination. The code
selected `restimulus ∈ {1,2,3,4,5,6}` from Exercise B.

NinaPro's movement table (Atzori et al.) numbers each group from 1 and
Exercise B concatenates two groups: **8 hand postures, then 9 wrist
movements**. Labels 1–6 are therefore *all hand postures* — thumb up, various
finger flexions, finger abduction, fist. Only one of the six (label 6, fist)
coincided with the synopsis by accident.

Verification used two independent lines of evidence:

* **Documentary** — the Atzori movement table, cross-checked against the
  17-movement count in the `*E2*.mat` files.
* **Empirical** — channel-topography correlations across all 10 subjects.
  `corr(13,14) = +0.08`, by far the lowest pair in the set, exactly as
  antagonist muscle groups on opposite sides of the forearm should look.
  The two supination variants pair at `+0.92`, the two pronation variants at
  `+0.76`.

A secondary measurement showed the mistake also made the problem *harder*:

```
mean within-group correlation, movements 1-8  : +0.67
mean within-group correlation, movements 9-17 : +0.45
```

The hand postures used are more mutually similar than the wrist movements the
synopsis specifies.

**Corrected mapping**, now in `src/data_loader.py`:

| Label | Gesture |
| ----- | ------- |
| 5 | open hand (finger abduction) |
| 6 | closed fist |
| 9 | forearm supination |
| 10 | forearm pronation |
| 13 | wrist flexion |
| 14 | wrist extension |

### 3.2 The pipeline could not run on the dataset

`data_loader.py` called `mat73.loadmat` unconditionally. All 30 DB5 files are
**MATLAB v5**, which `mat73` rejects outright:

```
TypeError: S1_E2_A1.mat is not a MATLAB 7.3 file. Load with scipy.io.loadmat()
```

`mat73` was also absent from `requirements.txt`, and `models/` did not exist, so
`joblib.dump` would have raised `FileNotFoundError` even had training succeeded.

### 3.3 The evaluation protocol leaked

`StratifiedKFold(shuffle=True)` was applied across windows built with 50%
overlap, with all 10 subjects pooled. Adjacent windows share half their samples,
so near-duplicates landed on both sides of every fold boundary. Reported
accuracy was inflated by an unknown but material amount. The SVM path
additionally fit `StandardScaler` on the full matrix before cross-validation.

### 3.4 Thirty-two features were numerically dead

Mean and median frequency were computed on the two-sided FFT spectrum. Because a
real signal's PSD is symmetric, the negative half cancels the positive half:

```
MNF over 500 windows: mean = -2.34 Hz, std 3.09   (true one-sided MNF ~43 Hz)
MDF over 500 windows: 47% negative
```

Two of four frequency-domain features, across all 16 channels, contributed
noise.

### 3.5 Remaining defects

| Defect | Effect |
| ------ | ------ |
| `filter_gestures()` ran before `preprocess()` | Spliced non-adjacent segments, filters rang at every join |
| `normalize()` used the first 200 samples as "rest" | Rest had already been deleted; normalised to an arbitrary gesture |
| `wavedec(level=3)` on 40 samples | `dwt_max_level` is 2; coarsest subbands were boundary artifact |
| `zc`/`ssc` had no deadzone | Counted noise crossings; `np.sign(0)` double-counted |
| `nan_to_num` applied silently | Concealed the breakage above |
| `segmentation.py` | Dead code, and used a different labelling rule than `build_features.py` |

### 3.6 Errors in the synopsis document

Six issues were found in the synopsis itself, not the code:

1. **Bandpass 20–450 Hz is impossible.** DB5 samples at 200 Hz; Nyquist is
   100 Hz. The implementation uses 20–95 Hz.
2. **LDA with `solver=SVD` plus Ledoit-Wolf shrinkage is invalid** —
   scikit-learn's SVD solver does not support shrinkage. `lsqr` with
   `shrinkage="auto"` is Ledoit-Wolf and is used instead.
3. **The ≤100 ms latency target contradicts §II.E**, which cites 200–300 ms as
   the clinical bound. A 200 ms analysis window exceeds 100 ms before any
   inference occurs.
4. **The ≥92% accuracy target is undefined** as to per-window or per-repetition.
   The project currently sits at 88.1% per-window and ~98% per-repetition.
5. **The ≤8-point cross-subject target is unreachable for tree models** —
   measured degradation is 14.9–16.4 points. LDA and SVM do meet it.
6. **The gesture list did not match the labels used** (§3.1).

---

## 4. Current state

### 4.1 Dataset

NinaPro DB5, all 10 subjects, downloaded and verified (194 MB, 30 `.mat`
files). 16 channels from two stacked Myo armbands at 200 Hz. Exercise B
provides 17 movements with 6 repetitions each. After gesture selection,
**252,428 in-gesture samples**.

### 4.2 Pipeline

```
src/
  data_loader.py     dataset IO, gesture definitions, subject ids
  preprocessing.py   bandpass, notch, rest-RMS normalisation, stage selection
  features.py        272-feature extraction (TD + FD + DWT)
  build_features.py  recordings -> feature table, windowing, trimming
  evaluate.py        shared loading, metrics, confusion matrices
  train.py           four-classifier benchmark, grid search, Wilcoxon
  loso.py            leave-one-subject-out with matched baseline
  ablation.py        feature-family and preprocessing ablations
  eda.py             dataset analysis
  experiments.py     smoothing and configuration sweeps
  predict.py         offline inference
```

Every module carries a self-check runnable as `python3 src/<module>.py`.

### 4.3 Evaluation protocol

Two independent leak-free protocols are reported:

* **10-fold stratified CV grouped by subject-repetition.** Grouping is
  essential — 50% window overlap means an ungrouped shuffle inflates accuracy.
* **The NinaPro convention** — train on repetitions 1, 3, 4, 6; test on 2, 5.

The two agree to within **0.4 points** across all four models, which is the
strongest available evidence that the figures are honest.

---

## 5. Results

### 5.1 Corrected baseline

272 features, no trimming, grouped 10-fold:

| Model | Grouped CV | Canonical split | Macro-F1 |
| ----- | ---------- | --------------- | -------- |
| Extra Trees | **0.742 ± 0.027** | 0.745 | 0.745 |
| Random Forest | 0.721 ± 0.023 | 0.725 | 0.724 |
| SVM (RBF) | 0.573 ± 0.022 | 0.573 | 0.579 |
| LDA | 0.527 ± 0.022 | 0.529 | 0.535 |

All six pairwise Wilcoxon comparisons significant at α = 0.05 (p ≤ 0.0059).

**Two qualifications on this table.** Extra Trees is *not* one of the four
classifiers the synopsis names — §IV.A.5 specifies LDA, SVM, Random Forest and a
1-D CNN. It is carried because it was the previous best model and remains the
strongest here, and it is flagged as out-of-synopsis in
`results/model_comparison.csv`. Separately, none of these four figures is tuned:
the synopsis requires a grid search over SVM's C and γ, and that search has never
been run, so `best_params` is empty for every row. The SVM and LDA numbers are
scikit-learn defaults and should be expected to move — which matters, because
§5.9 puts LDA on the critical path.

### 5.2 Error structure

Both dominant confusions are antagonist pairs:

* pronation → supination **0.20** (largest single error)
* wrist flexion ↔ extension **0.12** in both directions
* closed fist is cleanest at **0.87**

### 5.3 Feature ablation

Random Forest, 5-fold grouped, chance = 0.167:

| Arm | Features | Accuracy |
| --- | -------- | -------- |
| **TD only** | **144** | **0.792** |
| TD + FD | 208 | 0.792 |
| TD + FD + DWT | 272 | 0.782 |
| FD only | 64 | 0.270 |
| DWT only | 64 | 0.279 |

**Frequency and wavelet features are barely above chance and cost 1.1 points
when added to TD.** The cause is structural: every FD and DWT feature is
amplitude-invariant by construction (MNF, MDF, entropy, dominant frequency are
scale-free; DWT features are energy *ratios* summing to 1), while the RF
importances show the discriminative signal is almost entirely amplitude —
`mav`, `rms`, `iemg`, `wl`, `var`.

Rebuilding the wavelet features as **absolute** log-energy recovered +0.43
points but still lost to deleting them outright.

**This answers advanced objective 2**: the minimum feature subset within 2% of
the full set is TD alone, and it is in fact better.

### 5.4 Improvement experiments

Extra Trees, out-of-fold predictions under the same grouped 10-fold protocol.
The baseline row covers all 12,058 untrimmed windows; the four trimmed rows
cover the 8,288 that survive trimming, so the rows are *not* scored on the same
windows:

| Configuration | Features | Accuracy |
| ------------- | -------- | -------- |
| Baseline (untrimmed, 272, full preprocessing) | 272 | 0.742 |
| + onset/offset trim (15%) | 272 | 0.809 |
| + TD-only | 144 | 0.830 |
| + probability smoothing, k=3 | 144 | 0.858 |
| + probability smoothing, k=5 | 144 | 0.881 |
| **+ drop preprocessing (§5.10)** | 144 | **0.890** |
| + probability smoothing, k=7 instead of k=5 | 144 | **0.908** |

The last two rows are the current default configuration. Dropping the
preprocessing chain is worth **+0.9 points** on top of everything else, and the
full ladder on the new default is:

| Smoothing | Latency | Accuracy |
| --------- | ------- | -------- |
| none | 0 ms | 0.842 |
| probability, k=3 | 200 ms | 0.871 |
| **probability, k=5** | **400 ms** | **0.890** |
| probability, k=7 | 600 ms | 0.908 |
| oracle, whole repetition | — | 0.988 |

k=7 reaches **90.8%**, within 1.2 points of the synopsis' ≥92% target, at 600 ms
of smoothing latency. Since the ≤100 ms latency target is already unreachable by
a factor of two on the analysis window alone (§5.12), the trade between k=5 and
k=7 is worth putting to the guide alongside the per-window / per-repetition
question.

**Caveat on trimming.** Discarding the first and last 15% of each repetition
changes the *test set*, not only the model — it removes the hardest windows.
The honest phrasing is "80.9% on steady-state windows, 74.2% including
transients", not "trimming gained 6.8 points".

**Probability averaging beats hard majority voting** by 1.2–2.2 points at
identical latency, because a hard vote discards all confidence information.

### 5.5 Negative result: rotation alignment

The EDA identified armband rotation as the likely cause of the pronation /
supination confusion. Circularly aligning each armband's channels to its
peak-activation channel was implemented and measured. **It failed badly.**

| Extra Trees LOSO | Unaligned | Aligned |
| ---------------- | --------- | ------- |
| raw | 0.507 | 0.358 |
| subject-normalised | 0.645 | 0.467 |

Every model and both normalisations degraded. The peak channel of a mean
activation profile is not a stable landmark: when two subjects' peaks differ by
one channel through noise rather than genuine rotation, alignment *injects* a
spurious rotation. Aligning each armband independently also destroys the
relative offset between the two rings, which is apparently informative. The
feature has been kept behind an off-by-default flag and is not recommended.

### 5.6 Cross-subject generalisation (LOSO)

All 10 subjects, matched within-subject baselines on the same feature table, on
the current default (no preprocessing, 272 features):

| Model | Within-subject | LOSO raw | LOSO subject-norm | Degradation | ≤8 pts? |
| ----- | -------------- | -------- | ----------------- | ----------- | ------- |
| Extra Trees | 0.820 | 0.563 | 0.648 | 17.2 | No |
| Random Forest | 0.810 | 0.571 | 0.650 | 16.0 | No |
| **SVM** | 0.747 | 0.585 | **0.655** | 9.2 | No |
| LDA | 0.687 | 0.587 | 0.649 | **3.8** | **Yes** |

The best cross-subject figure is now **SVM at 65.5%**, up from LDA's 61.9% under
the old chain. Every model gained 2.7–3.5 points cross-subject.

**The ≤8-point target now looks worse while the system got better**, which
sharpens the point below. Under the old chain LDA (−0.6) and SVM (3.2) both met
it; now only LDA does, and SVM has slipped to 9.2. Nothing about cross-subject
performance degraded — SVM's *absolute* LOSO accuracy rose from 0.622 to 0.655.
Its degradation grew because its within-subject baseline rose faster, from 0.654
to 0.747. A model that improves on both axes can fail this target purely by
improving more on the easier one.

For comparison, the same table under the full preprocessing chain:

| Model | Within-subject | LOSO subject-norm | Degradation | ≤8 pts? |
| ----- | -------------- | ----------------- | ----------- | ------- |
| Extra Trees | 0.809 | 0.645 | 16.4 | No |
| Random Forest | 0.797 | 0.648 | 14.9 | No |
| SVM | 0.654 | 0.622 | 3.2 | Yes |
| LDA | 0.614 | 0.619 | −0.6 | Yes |

**Subject-independent normalisation is worth +12 to +14 points on every model.**
It z-scores features within each subject; it uses the held-out subject's own
statistics but no labels, so it is a calibration step rather than leakage, and
must be described as such.

**LDA and SVM meet the cross-subject target; the tree models miss it by
roughly double.** That result needs reading carefully, because the target
rewards a weak starting point. Degradation is measured against each model's own
within-subject accuracy, and LDA's is only 0.614 — it has little left to lose.
In *absolute* cross-subject accuracy the ordering reverses: Random Forest
reaches 0.648 and Extra Trees 0.645, against LDA's 0.619. The models that fail
the synopsis target are the more accurate ones across subjects. A target defined
on relative degradation cannot distinguish a model that generalises well from
one that was never good to begin with; this should be raised with the guide
alongside the per-window / per-repetition ambiguity in the ≥92% target.

### 5.7 EDA: why the rotations are hard

Mean pairwise inter-subject correlation of each gesture's channel-activation
pattern:

| Gesture | Mean corr | Min |
| ------- | --------- | --- |
| forearm supination | **0.164** | **−0.557** |
| forearm pronation | 0.289 | −0.313 |
| open hand | 0.365 | −0.129 |
| closed fist | 0.469 | +0.039 |
| wrist extension | 0.473 | −0.198 |
| wrist flexion | 0.619 | −0.057 |

The two forearm rotations are the least consistent across subjects, and are
exactly the pair with 20% mutual confusion within-subject. A negative minimum
means the activation pattern is effectively inverted between some subject pairs.

### 5.8 Aggregation ceiling

Aggregating predictions over a known repetition reaches **0.974–0.985** in every
configuration tested — with the same model, features and per-window predictions.
Per-window errors are therefore largely uncorrelated noise within a gesture, not
systematic confusion. The remaining gap is a temporal-aggregation problem, not a
feature-quality problem.

### 5.9 Model size

| Model | Pickle size |
| ----- | ----------- |
| Extra Trees | 285 MB |
| Random Forest | 124 MB |
| SVM | 23 MB |
| **LDA** | **0.6 MB** |

A 300-tree unlimited-depth forest is roughly three orders of magnitude larger
than LDA, for a 2–3 point gain in absolute LOSO accuracy. Combined with §5.6 —
where LDA meets the cross-subject degradation target and the trees do not — this
means **LDA's 61.4% is the number on the critical path**, not Extra Trees' 80.9%.

### 5.10 Preprocessing stage ablation (primary objective 3)

Four feature tables, identical in every respect except the preprocessing chain:
same 8,288 windows, same class balance, same 272 features. Random Forest,
5-fold grouped.

| Arm | Accuracy | Marginal effect of the stage added |
| --- | -------- | ---------------------------------- |
| **no preprocessing** | **0.8008** | — |
| + bandpass 20–95 Hz | 0.7907 | **−1.01** |
| + 50 Hz notch | 0.7936 | +0.29 |
| + rest-RMS normalisation (the full chain) | 0.7816 | **−1.19** |

**The canonical preprocessing chain costs 1.9 points.** No preprocessing at all
is the best arm.

Because rest-RMS normalisation is justified in synopsis §V.B as compensation for
*inter-subject* variation, a within-subject ablation cannot judge it. It was
therefore re-measured under LOSO, with and without that stage:

| Model | LOSO raw: off → on | LOSO subject-norm: off → on |
| ----- | ------------------ | --------------------------- |
| Extra Trees | 0.552 → 0.507 (**−4.6**) | 0.648 → 0.645 (−0.3) |
| Random Forest | 0.562 → 0.511 (**−5.1**) | 0.648 → 0.648 (+0.1) |
| SVM | 0.547 → 0.484 (**−6.3**) | 0.621 → 0.622 (+0.1) |
| LDA | 0.551 → 0.497 (**−5.4**) | 0.620 → 0.619 (−0.1) |

**Rest-RMS normalisation fails at the job the synopsis assigns it.** It costs
4.6–6.3 points cross-subject, and once subject-independent normalisation is
applied at feature level it contributes nothing either way.

The bandpass was then measured on the same axis, completing the picture. All
three arms, LOSO, 272 features:

| | Extra Trees | RF | SVM | LDA |
| --- | --- | --- | --- | --- |
| **LOSO raw** — none | **0.5632** | **0.5711** | **0.5845** | **0.5871** |
| bandpass + notch | 0.5523 | 0.5622 | 0.5474 | 0.5506 |
| full chain | 0.5065 | 0.5113 | 0.4842 | 0.4970 |
| **LOSO subject-norm** — none | **0.6476** | **0.6495** | **0.6548** | **0.6489** |
| bandpass + notch | 0.6476 | 0.6475 | 0.6214 | 0.6199 |
| full chain | 0.6451 | 0.6481 | 0.6221 | 0.6195 |

**The ordering is monotonic for every model: none > bandpass+notch > full
chain.** Each stage added costs cross-subject accuracy, so the earlier
within-subject result was not an artifact of the axis it was measured on.

### 5.10a Decision taken, and what it changed

The default chain is now **empty**. `DEFAULT_STAGES = ()` in
`preprocessing.py`; all three stages remain available via `--stages` and are
retained for the raw-front-end path (advanced objective 4), where they are
correct and necessary. Re-measuring the whole result set on the new default:

| Model | Full chain | No preprocessing | Gain |
| ----- | ---------- | ---------------- | ---- |
| Extra Trees | 0.8306 | **0.8386** | +0.79 |
| Random Forest | 0.8050 | **0.8186** | +1.36 |
| SVM | 0.7298 | **0.7707** | +4.10 |
| LDA | 0.6145 | **0.6705** | +5.60 |

**Every model improved.** The gain is largest for the two models that were most
damaged by rest-RMS normalisation flattening the between-channel amplitude
pattern, which is the mechanism §5.3 identified.

The cause is visible in `results/eda_spectra.png`, which plots the *raw*
recordings — `eda.py` applies no preprocessing:

* There is already a deep, sharp null at exactly 50 Hz. **NinaPro ships
  DB5_Preproc filtered.** Our notch is redundant.
* Real energy survives between 5 and 20 Hz. The 20 Hz high-pass, taken from
  Reaz et al. for *raw* clinical sEMG with motion artifact, discards it.
* Dividing each channel by its own rest RMS flattens the between-channel
  amplitude pattern — and §5.3 already established that amplitude (`mav`,
  `rms`, `iemg`, `wl`, `var`) is where the discriminative signal lives.

This is a methodology finding, not a bug: the chain is correct for raw sEMG and
wrong for an already-preprocessed corpus.

### 5.11 Table I grid search (primary objective 5)

Tuned against an untuned run on the **same** table (trimmed, TD-144), so the
comparison isolates tuning rather than mixing in trimming and feature selection:

| Model | Untuned | Tuned | Gain from tuning | Selected parameters |
| ----- | ------- | ----- | ---------------- | ------------------- |
| Extra Trees | 0.8306 | 0.8306 | **+0.00** | `max_depth=None, n=300` |
| Random Forest | 0.8030 | 0.8050 | +0.20 | `max_depth=20, n=300` |
| SVM (RBF) | 0.6969 | **0.7298** | **+3.29** | `C=100, gamma=0.01` |
| LDA | 0.6145 | 0.6145 | **+0.00** | none — Table I gives LDA no search space |

**The grid search is worth +3.3 points on SVM and nothing measurable on
anything else.** Quoting the tuned SVM against the old 0.573 baseline would
credit the search with +15.7 points; roughly 12.4 of those come from trimming
and the TD-only feature set.

Two caveats that belong next to these numbers:

* **Three of the four searches selected a grid boundary** — `C=100` is the top
  of `[0.1, 1, 10, 100]`, and `n_estimators=300` the top of `[100, 200, 300]`
  for both forests. The optimum may lie outside the range Table I specifies.
* `evaluate_model` is tune-then-evaluate, not nested CV. The search sees all the
  data, so the tuned figures are mildly optimistic.

### 5.12 Inference latency (measured for the first time)

`predict.py` had never been executed. Running it exposed a real defect first —
it extracted all 272 features unconditionally, so it could not load a TD-144
model at all. With that fixed, one window at a time on subject 1:

| Model | Feature extraction | Inference | Compute | + 200 ms window | ≤100 ms? |
| ----- | ------------------ | --------- | ------- | --------------- | -------- |
| Extra Trees (300 trees) | 12.88 ms | 45.75 ms | 58.63 ms | 258.6 ms | No |
| **LDA** | 12.32 ms | **0.16 ms** | 12.47 ms | 212.5 ms | No |

**LDA inference is 286× faster than Extra Trees.** Together with §5.9's 475×
size difference, the case for LDA on any deployment path is now quantified
rather than asserted.

The ≤100 ms target is unreachable for every model, and not because of the
models: filling the 200 ms analysis window exceeds the budget before a single
feature is computed. This confirms §3.6.4 with a measurement.

---

## 6. Objective status

### Primary objectives (Semester V)

| # | Objective | Status |
| - | --------- | ------ |
| 1 | Literature survey and gap analysis | Substantially done in synopsis §II; the separate *literature sheet* artifact does not exist |
| 2 | Full EDA on DB5 | **Complete** — `eda.py`, tables and figures in `results/` |
| 3 | Preprocessing pipeline + stage ablation | **Complete** — §5.10, within-subject and cross-subject. The measured answer contradicts the synopsis chain |
| 4 | 272-feature multi-domain extraction + CSV schema | **Complete**; evidence now favours the 144-feature TD subset |
| 5 | LDA / SVM / RF / 1-D CNN, metrics, Wilcoxon | LDA, SVM, RF complete with full metrics, Wilcoxon and **grid search (§5.11)**; **1-D CNN not started** |

### Advanced objectives (Semester VI)

Numbering follows synopsis §IV.B exactly.

| # | Objective | Status |
| - | --------- | ------ |
| 1 | LOSO + subject-independent normalisation | **Complete** |
| 2 | Feature ablation to minimum subset | **Complete** |
| 3 | Quantise and deploy the best model on an embedded platform | **Deferred** |
| 4 | MyoWare 2.0 + Arduino Nano prototype, self-collected dataset | **Deferred** |
| 5 | Real-time Streamlit dashboard | **Not started** |

Objectives 3 and 4 are the hardware and embedded track. They are **deferred, not
cancelled**, and are outside the scope of this report. No hardware has been
acquired, assembled or validated, and none of the associated targets — ≤100 ms
on-device latency, ≤$40 total cost, cross-device validation against NinaPro —
has been measured. The 1-D CNN is tracked under **primary** objective 5 above,
where the synopsis places it, not here.

### Synopsis performance targets

| Target | Status |
| ------ | ------ |
| ≥92% accuracy | 88.1% per-window; ~98% per-repetition. **Definition must be clarified** |
| ≤8-point cross-subject degradation | **Met by LDA (−0.6) and SVM (3.2)**; missed by trees |
| ≤100 ms inference latency | **Measured (§5.12): 212 ms for LDA, 259 ms for Extra Trees.** Missed by every model — the 200 ms window alone exceeds the budget |

---

## 7. Pending work

### 7.1 Immediate

Done since the last revision: trimming is now the `build_features.py` default,
`--align` is deleted, `features_*.csv` is ignored, the work is committed, the
grid search and preprocessing ablation have been run (§5.11, §5.10), latency is
instrumented and measured (§5.12), `predict.py` has been executed, and the
README is current.

**Resolved:** the preprocessing chain is dropped from the default and kept as
opt-in (§5.10a). Every model improved on both axes; the synopsis text needs a
corresponding correction, which joins the §3.6 list.

1. **Smoothing k=5 or k=7?** k=7 reaches 90.8% against k=5's 89.0%, for 200 ms
   more latency. The ≤100 ms target is already unreachable by a factor of two,
   so the extra 200 ms may cost nothing that was achievable anyway. Needs the
   guide.
2. **Widen the Table I grid.** Three of four searches selected a boundary value
   (§5.11), so the specified range is probably truncated below the optimum.
3. **PCA → LDA.** Still on the critical path: LDA is 475× smaller and 286×
   faster at inference, and within 3 points of the trees cross-subject.
4. **Nested CV for the tuned figures**, if the mild optimism in §5.11 matters.

### 7.2 Semester VI

11. **1-D CNN** to Table I specification (PyTorch; `torch` is not yet a
    dependency). Primary objective 5 is incomplete without it.
12. **Decide the rest / no-gesture class.** It gates both the CNN output layer
    and the dashboard's behaviour, so it should be settled before either.
13. **Streamlit dashboard** — live 16-channel waveform, predicted label and
    confidence, running accuracy log.
14. **Real-time inference loop** feeding the dashboard, built on `predict.py`.
15. **Adaptive LMS filter** — §II.B motivates it explicitly; the pipeline is
    currently fixed-coefficient throughout.
16. **52-gesture extension**, reserved by §V.A.
17. **Literature sheet** artifact.
18. **Correct the six synopsis errors** listed in §3.6.

### 7.3 Sequencing

The rest-class decision is the one item that gates others: it determines the
CNN's output layer and the dashboard's behaviour, so settling it early avoids
rebuilding both.

Recommended order: consolidate and commit → PCA/LDA and latency measurement →
decide the rest class → CNN → dashboard.

---

## 8. Summary of the numbers

| Metric | Value |
| ------ | ----- |
| Previously reported (unreproducible, wrong gestures, leaky) | ~84% |
| Corrected baseline, all features, all windows | 74.2% |
| **Best within-subject** (Extra Trees, TD-144, no preprocessing, smoothed k=5) | **89.0%** |
| Same, smoothed k=7 (600 ms) | **90.8%** |
| Per-repetition aggregation ceiling | 98.8% |
| **Best cross-subject** (SVM, subject-normalised) | **65.5%** |
| Cross-subject degradation, LDA | 3.8 points (target ≤8) |
| Feature count, best configuration | 144 (from 272) |
| Smallest model, LDA | 0.6 MB (vs 285 MB for Extra Trees) |
| Cost of the synopsis preprocessing chain | −0.8 to −5.6 within-subject, −1.1 to −8.8 cross-subject |
| Value of the Table I grid search | +3.3 points on SVM, +0.0 on everything else |
| Inference latency, LDA vs Extra Trees | 0.16 ms vs 45.75 ms per window |
| End-to-end latency, best case | 212 ms (target ≤100 ms, unreachable — the window alone is 200 ms) |
