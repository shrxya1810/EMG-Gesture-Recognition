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

The headline result is **88.1% within-subject accuracy** (Extra Trees,
time-domain features, probability-smoothed over 5 windows) and **61.9%
cross-subject accuracy** (LDA with subject-independent normalisation, which
meets the synopsis' cross-subject degradation target).

An audit of the earlier pipeline found that the previously reported ~84% figure
was not reproducible. It came from a leaky evaluation protocol applied to the
**wrong six gestures**, produced by code that crashes on the actual dataset.
Section 3 documents this in full. The corrected, honestly-measured baseline was
74.2%; targeted improvements raised it to 88.1%.

Semester V is the strongest part of the picture, but not finished: of five
primary objectives, the literature-sheet artifact does not exist, the
preprocessing ablation is implemented but has never been run, and objective 5
needs both the 1-D CNN and the grid search. Two of five *advanced* objectives —
embedded deployment and the MyoWare hardware prototype — are deferred and out of
scope for this report, and are marked as such in §6 rather than dropped.

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
| Baseline (untrimmed, 272) | 272 | 0.742 |
| + onset/offset trim (15%) | 272 | 0.809 |
| + TD-only | 144 | 0.830 |
| + probability smoothing, k=3 | 144 | 0.858 |
| + probability smoothing, k=5 | 144 | **0.881** |

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

All 10 subjects, matched within-subject baselines on the same feature table:

| Model | Within-subject | LOSO raw | LOSO subject-norm | Degradation | ≤8 pts? |
| ----- | -------------- | -------- | ----------------- | ----------- | ------- |
| Extra Trees | 0.809 | 0.507 | 0.645 | 16.4 | No |
| Random Forest | 0.797 | 0.511 | 0.648 | 14.9 | No |
| SVM | 0.654 | 0.484 | 0.622 | **3.2** | **Yes** |
| LDA | 0.614 | 0.497 | 0.619 | **−0.6** | **Yes** |

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

---

## 6. Objective status

### Primary objectives (Semester V)

| # | Objective | Status |
| - | --------- | ------ |
| 1 | Literature survey and gap analysis | Substantially done in synopsis §II; the separate *literature sheet* artifact does not exist |
| 2 | Full EDA on DB5 | **Complete** — `eda.py`, tables and figures in `results/` |
| 3 | Preprocessing pipeline + stage ablation | Pipeline complete; **ablation implemented but never run** |
| 4 | 272-feature multi-domain extraction + CSV schema | **Complete**; evidence now favours the 144-feature TD subset |
| 5 | LDA / SVM / RF / 1-D CNN, metrics, Wilcoxon | LDA, SVM, RF complete with full metrics and Wilcoxon; **1-D CNN not started**; **grid search implemented but never run** |

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
| ≤100 ms inference latency | **Never measured**, and unreachable as specified |

---

## 7. Pending work

### 7.1 Immediate

1. **Fold the measured wins into the default pipeline.** Trimming, TD-only
   features and probability smoothing currently live only in `experiments.py`.
   `build_features.py` defaults and `train.py` still produce the 74.2%
   configuration, and the saved models in `models/` are from that run. A fresh
   clone currently reproduces the old number.
2. **Remove the `--align` option** — measured harmful (§5.5).
3. **Add `features_*.csv` to `.gitignore`** — 199 MB of variant tables are
   currently stageable.
4. **Commit.** Nothing has been committed; 23 paths are uncommitted.
5. **PCA → LDA.** On the critical path for the cross-subject target, and the
   smallest model by three orders of magnitude. Trimming alone already lifted
   LDA from 0.527 to 0.614.
6. **Run the full Table I grid search.** SVM and LDA figures are untuned.
7. **Run the preprocessing ablation** — required by primary objective 3.
8. **Add latency instrumentation.** The ≤100 ms target has never been measured
   for anything; this converts it into a number.
9. **Execute `predict.py` once** — written but never run.
10. **Refresh the README** — written before the TD-only and smoothing results.

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
| Best within-subject (Extra Trees, TD-only, smoothed k=5) | **88.1%** |
| Per-repetition aggregation ceiling | ~98% |
| Best cross-subject (LDA, subject-normalised) | 61.9% |
| Cross-subject degradation, LDA | −0.6 points (target ≤8) |
| Feature count, best configuration | 144 (from 272) |
| Smallest model, LDA | 0.6 MB (vs 285 MB for Extra Trees) |
