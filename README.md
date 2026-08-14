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
| Corrected baseline, 272 features, all windows, full preprocessing | 0.742 |
| **Best within-subject** (Extra Trees, TD-144, no preprocessing, smoothed k=5) | **0.890** |
| Same at k=7 (600 ms smoothing latency) | 0.908 |
| Per-repetition aggregation ceiling | 0.988 |
| **Best cross-subject** (SVM, subject-normalised) | **0.655** |

Two findings that contradict the synopsis, both measured with matched controls:

- **The preprocessing chain costs accuracy, so it is off by default.** Measured
  on both axes: within-subject every model gains 0.8–5.6 points without it, and
  under LOSO the ordering is monotonic (none > bandpass+notch > full chain) for
  all four. DB5_Preproc arrives already filtered — the raw spectra in
  `results/eda_spectra.png` have a deep 50 Hz null before we touch them.
  `DEFAULT_STAGES = ()`; every stage is still one flag away via `--stages`, and
  they are correct and necessary for a raw front-end. See PROGRESS.md §5.10.
- **The Table I grid search is worth +3.3 points on SVM and nothing else.**
  Measured against an untuned run on the same table. Three of four searches
  selected a grid boundary, so the specified range is likely truncated.
  See PROGRESS.md §5.11.

Two caveats before quoting any number:

- **Tuned figures are tune-then-evaluate, not nested CV** — the search sees all
  the data, so they are mildly optimistic.
- **One model file per configuration is kept**, suffixed by `--label`:
  `*_mavrms.pkl` is the current best (32 columns, §5.14), `*_nopp.pkl` the
  TD-144 default it replaced, plus the tuned and untuned arms and the CNNs.
  None are committed — `models/` is ignored, and every one is re-creatable with
  `train.py --label <name>`. The numbers they produced live in `results/`.

Implemented and run: preprocessing, segmentation, features, classifier
benchmark with grid search, LOSO, feature ablation, preprocessing ablation, EDA,
offline inference with latency measurement, the rest gate, the 1-D CNN
(§5.15 — it loses to every classical model), and the replay dashboard.
Not started: the literature-sheet artifact (primary objective 1).
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
python3 src/build_features.py             # -> features.csv (trimmed, no preprocessing)
python3 src/train.py --groups TD --label tuned    # objective 5: benchmark + grid search
python3 src/loso.py                       # advanced 1: cross-subject
python3 src/ablation.py features          # advanced 2: feature families
python3 src/predict.py --mat <file.mat>   # offline inference + latency
python3 src/cnn.py --self-check           # objective 5: verify, trains nothing
python3 src/cnn.py --label spec           # objective 5: 1-D CNN, ~30 min CPU
python3 src/report.py                     # every model's score, one table
python3 src/report.py --mat <f.mat> --at 248.9   # + what each predicts there
streamlit run src/dashboard.py            # advanced 5: replay dashboard
```

`build_features.py` trims 15 % off each end of every repetition by default;
`--trim 0` rebuilds the transient-inclusive baseline. `train.py --quick` skips
the grid search. `--groups` selects features, by family (`TD`, `FD`, `DWT`) or
by individual statistic (`mav`, `rms`, `zc`, …). **`--groups mav,rms` is the
current best configuration** — 32 columns, 0.856 per-window against TD-144's
0.842, see PROGRESS.md §5.14. `--label X` suffixes every output file so a
second run cannot overwrite the first. Every module runs its own self-check when executed
directly, e.g. `python3 src/features.py`.

Preprocessing ablation (objective 3) needs one feature table per configuration.
`features.csv` is already the no-preprocessing arm, since that is the default:

```
python3 src/build_features.py --stages bandpass                 -o feat_bp.csv
python3 src/build_features.py --stages bandpass,notch           -o feat_bpn.csv
python3 src/build_features.py --stages bandpass,notch,normalize -o feat_full.csv
python3 src/ablation.py preprocessing features.csv feat_bp.csv feat_bpn.csv feat_full.csv
```

Whether a stage helps cross-subject is a separate question from whether it helps
within-subject. They agree here, but only because both were measured — see
PROGRESS.md §5.10:

```
python3 src/loso.py --features features.csv --label nopp   # nothing applied
python3 src/loso.py --features feat_bpn.csv --label bpn    # bandpass + notch
python3 src/loso.py --features feat_full.csv --label full  # the whole chain
```

### Dashboard (advanced objective 5)

```
streamlit run src/dashboard.py
```

Replays a recording window by window at its native 100 ms step, showing the
current 16-channel window, the predicted gesture with its confidence, the true
label, and a running accuracy. There is no armband — advanced objectives 3 and
4 are deferred — so this is replay, not live capture.

The sidebar selects the recording, the model, and the `--groups` and `--stages`
that model was trained with. Those must match: the page checks the feature
count against the model and refuses in words rather than failing inside
sklearn. It defaults to S6 and `extratrees_mavrms.pkl`, the strongest
configuration (§5.14). The first load scores the whole recording and takes a
minute or two, then caches per (recording, model, groups, stages, gate).

**Condense to gesture zones**, on by default, skips the idle stretches. An E2
recording is only 11-14 % trained gesture against ~58 % rest, so a straight
replay is mostly an idle arm; condensing keeps every trained-gesture window
plus context and thins what surrounds it, which puts the six at 44-49 % of the
replay and shortens it from ~17 min to ~4.5 min. It never drops a
trained-gesture window, so the gesture-only figure is identical either way.

**Reading the accuracy.** It is scored over rest plus the six trained gestures.
Windows straddling a label change, and windows carrying one of the other eleven
Exercise B gestures, are skipped: `build_features.py` drops both, the model was
never trained on them and cannot output them, so scoring against them would be
guaranteed-wrong windows — left in, they cap the figure at 72 % on any model.
It is still not comparable to the 89-93 % in PROGRESS.md, which additionally
trims each repetition's transients and excludes rest altogether. Rest dominates
what remains and the gate clears ~99 % of it, so the page reports the rest and
gesture halves separately rather than one flattering average.

---

## Pipeline

**Preprocessing** — available but **off by default** (`DEFAULT_STAGES = ()`).
The stages are a 4th-order zero-phase Butterworth bandpass, a 50 Hz IIR notch
(Q = 30), and per-channel normalisation to the subject's rest RMS. When enabled
they run on the continuous recording *before* gestures are selected, so the
filters never cross a splice. All three measured worse on DB5_Preproc, which
arrives already filtered; they are retained for a raw front-end, where they are
correct and necessary.

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
| Bandpass + notch + RMS normalisation always applied | off by default, opt-in via `--stages` | Costs 0.8–5.6 pts within-subject and up to 8.8 cross-subject on DB5_Preproc, which is already filtered. Kept in the code for the raw front-end |
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
  rest_gate.py       amplitude gate for rest / no-gesture at inference
  cnn.py             1-D CNN (Table I), raw 16x40 input, same folds as train.py
  report.py          all model scores in one table; per-window predictions
  dashboard.py       Streamlit replay dashboard (advanced objective 5)
data/raw/            dataset (not committed)
models/              trained models (not committed)
results/             metrics, confusion matrices, figures
```

---

## Authors

Shreya Agarwal · Shruti Jha · Ekansh Bansal
School of Electrical Engineering, Manipal Institute of Technology
Guide: Ms. Vidya Kamath
