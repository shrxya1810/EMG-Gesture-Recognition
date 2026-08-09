# EMG Gesture Recognition (sEMG + Machine Learning)

End-to-end surface EMG gesture recognition on the **NinaPro DB5** dataset:
preprocessing, sliding-window segmentation, multi-domain feature extraction and
a benchmarked set of classical classifiers.

Companion to the project synopsis, *Design and Development of a Real-Time
EMG-Based Forearm Gesture Recognition System*.

---

## Status

The pipeline is implemented end to end and benchmarked. Full write-up in
[`PROGRESS.md`](PROGRESS.md); raw numbers in `results/`.

| Metric | Value |
| ------ | ----- |
| Corrected baseline, 272 features, all windows | 0.742 |
| Best within-subject (Extra Trees, TD-144, smoothed k=5) | **0.881** |
| Per-repetition aggregation ceiling | ~0.98 |
| Best cross-subject (LDA, subject-normalised) | 0.619 |

Two findings that contradict the synopsis, both measured with matched controls:

- **The preprocessing chain costs accuracy.** No preprocessing beats the full
  bandpass + notch + rest-RMS chain by 1.9 points within-subject, and rest-RMS
  normalisation costs 4.6–6.3 points cross-subject. DB5_Preproc arrives already
  filtered — the raw spectra in `results/eda_spectra.png` have a deep 50 Hz null
  before we touch them. The default has **not** been changed; use `--stages` to
  pick a chain. See PROGRESS.md §5.10.
- **The Table I grid search is worth +3.3 points on SVM and nothing else.**
  Measured against an untuned run on the same table. Three of four searches
  selected a grid boundary, so the specified range is likely truncated.
  See PROGRESS.md §5.11.

Two caveats before quoting any number:

- **Tuned figures are tune-then-evaluate, not nested CV** — the search sees all
  the data, so they are mildly optimistic.
- **`models/*.pkl` are untuned 272-feature models; `models/*_tuned.pkl` are the
  tuned TD-144 ones.** Pass `--groups TD` to `predict.py` for the latter.

Implemented and run: preprocessing, segmentation, features, classifier
benchmark with grid search, LOSO, feature ablation, preprocessing ablation, EDA,
offline inference with latency measurement.
Not started: 1-D CNN, Streamlit dashboard (Semester VI).
Deferred: embedded deployment, MyoWare hardware prototype (synopsis advanced
objectives 3 and 4 — deferred, not cancelled).

---

## Gestures

DB5 Exercise B numbers **hand postures 1–8** and **wrist movements 9–17**. The
six gestures named in the synopsis are therefore:

| Label | Gesture |
| ----- | ------------------------------ |
| 5     | open hand (finger abduction)   |
| 6     | closed fist                    |
| 9     | forearm supination             |
| 10    | forearm pronation              |
| 13    | wrist flexion                  |
| 14    | wrist extension                |

Earlier revisions used labels 1–6, which are six *hand postures* and do not
correspond to the synopsis. The mapping lives in `src/data_loader.py`; do not
renumber it without checking the Atzori movement table.

---

## Dataset

NinaPro DB5, not committed. Download the ten `DB5_Preproc` archives and extract
so that the `*E2*.mat` files sit anywhere under `data/raw/`:

```
data/raw/s1/s1/S1_E2_A1.mat
...
data/raw/s10/s10/S10_E2_A1.mat
```

16 channels, 200 Hz, 6 repetitions per gesture, 10 subjects. The files are
MATLAB v5, so `scipy.io.loadmat` reads them; `mat73` is not required.

---

## Install

```
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Run

```
python3 src/eda.py                        # objective 2: dataset analysis
python3 src/build_features.py             # -> features.csv (trimmed, 272 features)
python3 src/train.py --groups TD --label tuned    # objective 5: benchmark + grid search
python3 src/loso.py                       # advanced 1: cross-subject
python3 src/ablation.py features          # advanced 2: feature families
python3 src/predict.py --mat <file.mat> --model models/extratrees_tuned.pkl --groups TD
```

`build_features.py` trims 15 % off each end of every repetition by default;
`--trim 0` rebuilds the transient-inclusive baseline. `train.py --quick` skips
the grid search. `--groups TD` selects the 144-feature time-domain subset that
the ablation favours; `--label X` suffixes every output file so a second run
cannot overwrite the first. Every module runs its own self-check when executed
directly, e.g. `python3 src/features.py`.

Preprocessing ablation (objective 3) needs one feature table per configuration:

```
python3 src/build_features.py --stages ""             -o feat_none.csv
python3 src/build_features.py --stages bandpass       -o feat_bp.csv
python3 src/build_features.py --stages bandpass,notch -o feat_bpn.csv
python3 src/ablation.py preprocessing feat_none.csv feat_bp.csv feat_bpn.csv features.csv
```

Whether the stage helps cross-subject is a separate question from whether it
helps within-subject, and the two disagree here — see PROGRESS.md §5.10:

```
python3 src/loso.py --features feat_bpn.csv --label bpn    # normalisation off
python3 src/loso.py --features features.csv --label full   # normalisation on
```

---

## Pipeline

**Preprocessing** — 4th-order zero-phase Butterworth bandpass, 50 Hz IIR notch
(Q = 30), per-channel normalisation to the subject's rest RMS. Applied to the
continuous recording *before* gestures are selected, so the filters never run
across a splice.

**Segmentation** — 200 ms windows (40 samples), 50 % overlap. Windows less than
90 % pure in their dominant label are dropped, which removes gesture
transitions.

**Features** — 17 per channel × 16 channels = **272**.

| Domain | Features |
| ------ | -------- |
| Time (9) | MAV, RMS, WL, ZC, SSC, IEMG, variance, skewness, kurtosis |
| Frequency (4) | MNF, MDF, spectral entropy, dominant frequency |
| Wavelet (4) | level-3 db4 subband energy ratios (cA3, cD3, cD2, cD1) |

ZC and SSC use an amplitude deadzone so they do not count noise crossings.
Spectral features are computed on the one-sided spectrum after a Hamming
window.

**Classifiers** — LDA, SVM (RBF), Random Forest, Extra Trees, each inside a
`Pipeline` with a scaler so no scaling statistics leak across folds. Grid
search follows synopsis Table I.

**Evaluation** — 10-fold stratified CV *grouped by subject-repetition*, plus
the NinaPro convention (train on repetitions 1, 3, 4, 6; test on 2, 5).
Grouping matters: windows overlap by 50 %, so an ungrouped shuffle puts
near-duplicate windows on both sides of a fold boundary and inflates accuracy.
Reported per model: per-class and macro precision / recall / F1, confusion
matrices, and pairwise Wilcoxon signed-rank tests at α = 0.05.

---

## Known deviations from the synopsis

| Synopsis | Here | Why |
| -------- | ---- | --- |
| Bandpass 20–450 Hz | 20–95 Hz | DB5 samples at 200 Hz; Nyquist is 100 Hz |
| LDA solver SVD + shrinkage | solver lsqr + shrinkage | scikit-learn's SVD solver does not support shrinkage |
| RMS normalisation over 500 ms rest | all rest samples in the recording | same intent, more stable estimate |
| Preprocessing chain always applied | measured, and it loses | −1.9 pts within-subject, −4.6 to −6.3 cross-subject on DB5_Preproc, which is already filtered. Default unchanged pending the guide's decision |
| Windows include onset/offset | first and last 15 % of each repetition dropped | +6.8 pts, but it changes the test set as well as the model |
| — | Extra Trees | not in the synopsis; kept as the previous best model, flagged in the output |

Level-3 db4 on a 40-sample window exceeds `pywt.dwt_max_level` (= 2), so the
coarsest subbands carry boundary effects. `mode='periodization'` keeps padding
from inventing energy. The clean fix is a window longer than 56 samples, which
conflicts with the 200 ms the synopsis specifies.

---

## Layout

```
src/
  data_loader.py     dataset IO, gesture definitions
  preprocessing.py   filtering and normalisation
  features.py        272-feature extraction
  build_features.py  raw recordings -> features.csv
  evaluate.py        shared loading and metrics
  train.py           classifier benchmark
  loso.py            leave-one-subject-out
  ablation.py        feature and preprocessing ablations
  eda.py             dataset analysis
  experiments.py     smoothing and configuration sweeps
  predict.py         offline inference and latency measurement
data/raw/            dataset (not committed)
models/              trained models (not committed)
results/             metrics, confusion matrices, figures
```

---

## Authors

Shreya Agarwal · Shruti Jha · Ekansh Bansal
School of Electrical Engineering, Manipal Institute of Technology
Guide: Ms. Vidya Kamath
