"""Real-time-replay dashboard (PROGRESS.md 7.2 item 3): 16-channel waveform,
predicted gesture + confidence, running accuracy -- built on predict.py's own
pipeline pieces, replayed window by window at recording speed.

    streamlit run src/dashboard.py

There is no live armband (7.2 item 3/4 are deferred hardware), so "real-time"
here means replaying a NinaPro recording at its native 100 ms/window rate.
"""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from build_features import STEP, WINDOW_SIZE
from data_loader import GESTURES, load_subject
from evaluate import select_groups
from features import FEATURE_NAMES, extract_all
from predict import windows
from preprocessing import preprocess
from rest_gate import K, calibrate, is_gesture

st.set_page_config(page_title="EMG gesture dashboard", layout="wide")


@st.cache_resource
def get_model(path):
    return joblib.load(path)


@st.cache_resource
def get_signal(mat_path, stages):
    """Preprocessed recording and its labels.

    cache_resource, not cache_data: this is ~23 MB and read-only, and
    cache_data re-unpickles its whole value on every rerun -- which during
    playback is every 100 ms.
    """
    emg, labels, _ = load_subject(mat_path)
    emg = preprocess(emg, rest_mask=(labels == 0),
                     stages=tuple(s for s in stages.split(",") if s))
    return emg, labels


# Rest plus the six gestures: the only labels the model can be right about.
# E2 holds all 17 Exercise B gestures and build_features.py drops the other
# eleven ("label not in GESTURES"), so the model has never seen them and can
# never output them. Scoring against them would be guaranteed-wrong windows.
IN_SCOPE = [0, *GESTURES]


def true_name(v):
    """NaN means the window straddles a label change (predict.py MIN_PURITY)."""
    if pd.isna(v):
        return "transition"
    v = int(v)
    if v == 0:
        return "rest"
    return GESTURES.get(v, f"gesture {v} (not one of the six)")


def condense(true, pad=15, keep_other_every=8):
    """Row mask keeping the six trained gestures and thinning everything else.

    An E2 recording is 13% trained gesture against 58% rest (measured on S6),
    so a straight replay is mostly an idle arm. This keeps every trained-gesture
    window plus `pad` windows either side -- the onset and offset ramps, and the
    rest between repetitions -- and one in `keep_other_every` of the untrained
    gesture blocks, so rest and the other eleven stay on screen without
    dominating it. On S6 that moves the six from 13.5% of the replay to 49%.

    It filters rows only. window_start still indexes the untouched recording,
    so the waveform and the clock stay honest and the gaps stay visible.
    """
    six = true.isin(list(GESTURES)).to_numpy()
    keep = np.convolve(six.astype(int), np.ones(2 * pad + 1), "same") > 0

    other = ((~true.isin(IN_SCOPE)) & true.notna()).to_numpy()
    runs = np.cumsum(np.r_[True, other[1:] != other[:-1]])
    for n, g in enumerate(np.unique(runs[other])):
        if n % keep_other_every == 0:
            lo, hi = np.flatnonzero(runs == g)[[0, -1]]
            keep[lo:hi + 1] = True
    return keep


@st.cache_data
def get_predictions(mat_path, model_path, groups, stages, rest_k):
    """One pass over the recording: predicted label + confidence per window.

    Batched up front rather than per-frame, same as predict.py -- inference
    over ~9k windows is sub-second, and it keeps the playback loop below from
    re-running the model on every rerun. The raw windows are deliberately not
    returned: they are 46 MB, and cache_data would re-unpickle them per rerun.
    Callers slice get_signal() instead.
    """
    model = get_model(model_path)
    emg, labels = get_signal(mat_path, stages)
    rest_mav = calibrate(emg, labels == 0)

    gated, active = [], []
    for start, w, label in windows(emg, labels):
        row = {"window_start": start, "true": label}
        # Gate before extracting features, not after -- same order as
        # predict.py, and on a resting arm it skips the 14 ms extraction.
        if rest_k and not is_gesture(w, rest_mav, rest_k):
            gated.append(row)
        else:
            row["feat"] = extract_all(w)
            active.append(row)

    if active:
        F, _ = select_groups(np.asarray([r.pop("feat") for r in active]),
                             FEATURE_NAMES, groups)
        pred = model.predict(F)
        # NaN rather than None throughout, so the column stays float64 -- a
        # None here makes it object dtype and `x is not None` then lies.
        conf = (model.predict_proba(F).max(axis=1)
                if hasattr(model, "predict_proba") else np.full(len(F), np.nan))
        for r, p, c in zip(active, pred, conf):
            r.update(predicted=int(p), confidence=float(c),
                     predicted_name=GESTURES.get(int(p), "?"))

    for r in gated:
        r.update(predicted=0, predicted_name="rest", confidence=np.nan)

    df = pd.DataFrame(gated + active).sort_values("window_start",
                                                  ignore_index=True)
    df["true_name"] = df["true"].map(true_name)
    return df


st.sidebar.header("Recording")
mat_files = sorted(Path("data/raw").glob("*/*/S*_E2_A1.mat"))
mat_path = st.sidebar.selectbox(
    "Subject recording", mat_files,
    # S6 holds the most trained-gesture windows of the ten (1389); the default
    # path sort would otherwise open S1, whose first gesture is 245 s in.
    index=next((i for i, p in enumerate(mat_files)
                if p.name == "S6_E2_A1.mat"), 0),
    format_func=lambda p: p.name)
condensed = st.sidebar.checkbox(
    "Condense to gesture zones", value=True,
    help="Skip the long idle stretches. Keeps every trained-gesture window "
         "with context, plus some rest and some untrained gestures.")

st.sidebar.header("Model")
# cnn_*.pkl are state-dict dicts, not sklearn estimators -- they have no
# .predict, and this whole page is built on predict.py's classical path.
model_files = sorted(p for p in Path("models").glob("*.pkl")
                     if not p.name.startswith("cnn"))
model_path = st.sidebar.selectbox(
    "Model", model_files,
    index=next((i for i, p in enumerate(model_files)
                if p.name == "extratrees_mavrms.pkl"), 0),
    format_func=lambda p: p.name)
groups = st.sidebar.text_input("--groups (must match training)", "mav,rms")
stages = st.sidebar.text_input("--stages (must match training)", "")
rest_k = st.sidebar.slider("Rest gate (x resting MAV)", 0.0, 4.0, K, 0.25)

# Fail fast and in words. predict.py raises SystemExit on the same mismatch,
# but here it would land after the ~2 min extraction, as a bare sklearn error.
try:
    n_cols = len(select_groups(np.zeros((1, len(FEATURE_NAMES))),
                               FEATURE_NAMES, groups)[1])
except SystemExit as e:
    st.error(str(e))
    st.stop()
n_expected = getattr(get_model(str(model_path)), "n_features_in_", n_cols)
if n_expected != n_cols:
    st.error(f"{model_path.name} was trained on {n_expected} features, but "
             f"--groups '{groups}' gives {n_cols}. Use `mav,rms` for the 32 "
             f"models, `TD` for the 144 ones, or leave it empty for 272.")
    st.stop()

# extract_all costs ~14 ms/window (PROGRESS.md 5.14, no per-statistic fast
# path yet -- 7.3 item 4) -- a full ~9k-window recording takes over a minute
# the first time; cached per (file, model, groups, stages, rest_k) after that.
with st.spinner("Scoring recording (first load is slow, then cached)..."):
    df = get_predictions(str(mat_path), str(model_path), groups, stages, rest_k)

# After scoring, never before: the whole recording stays cached, so ticking
# this on and off is a row filter rather than another two-minute extraction.
if condensed:
    df = df[condense(df["true"])].reset_index(drop=True)

st.session_state.setdefault("idx", 0)
st.session_state.setdefault("playing", False)

c1, c2, c3 = st.sidebar.columns(3)
if c1.button("Play"):
    st.session_state.playing = True
if c2.button("Pause"):
    st.session_state.playing = False
if c3.button("Reset"):
    st.session_state.idx = 0
    st.session_state.playing = False

# Everything that moves the cursor must happen *before* the slider is created:
# a keyed widget's session_state entry is writable up to instantiation and
# raises afterwards. That ordering is also what lets the slider own `idx`
# outright -- passing value= instead would rebuild the widget every frame,
# because Streamlit derives widget identity from its arguments.
if st.session_state.playing:
    st.session_state.idx = min(st.session_state.idx + 1, len(df) - 1)
st.session_state.idx = min(st.session_state.idx, len(df) - 1)   # shorter file
st.sidebar.slider("Window", 0, len(df) - 1, key="idx")

row = df.iloc[st.session_state.idx]
seen = df.iloc[:st.session_state.idx + 1]
scored = seen[seen["true"].isin(IN_SCOPE)]      # isin drops NaN transitions too
skipped = len(seen) - len(scored)
hit = scored["predicted"] == scored["true"]
running_acc = hit.mean() if len(scored) else float("nan")
# Rest is ~80% of the windows in a recording and the gate gets ~99% of it, so
# the combined figure is mostly a rest-rejection score. Split it, or someone
# reads 96% next to PROGRESS.md's 89% and concludes the wrong thing.
is_rest = scored["true"] == 0
rest_acc = hit[is_rest].mean() if is_rest.any() else float("nan")
gest_acc = hit[~is_rest].mean() if (~is_rest).any() else float("nan")

emg, _ = get_signal(str(mat_path), stages)
left, right = st.columns([3, 1])
with left:
    st.subheader(f"Window {st.session_state.idx}  "
                f"(t={row.window_start / 200:.2f}s)")
    st.line_chart(pd.DataFrame(
        emg[row.window_start:row.window_start + WINDOW_SIZE]))
with right:
    st.metric("Predicted", row.predicted_name,
              f"{row.confidence:.0%} conf." if pd.notna(row.confidence) else None)
    st.metric("True label", row.true_name)
    st.metric("Running accuracy", f"{running_acc:.1%}" if running_acc == running_acc
              else "n/a", f"{len(scored)} scored, {skipped} skipped")
    st.caption(f"**{rest_acc:.0%}** of that is rest correctly rejected "
              f"({int(is_rest.sum())} windows); **{gest_acc:.0%}** on the six "
              f"gestures themselves ({int((~is_rest).sum())} windows).")
    st.caption(("Condensed view: idle stretches skipped, so this is **not** a "
                "whole-recording figure -- untick to score the full replay. "
                if condensed else "")
              + "Scored over rest plus the six trained gestures. Skipped: "
              "windows straddling a label change, and windows holding one of "
              "the other eleven Exercise B gestures -- build_features.py drops "
              "both, so the model was never trained on them. Still not "
              "directly comparable to PROGRESS.md's 89-93%, which also trims "
              "each repetition's transients and excludes rest (5.13/5.14).")

st.subheader("Recent predictions")
st.dataframe(df.iloc[max(0, st.session_state.idx - 19):st.session_state.idx + 1]
            [["window_start", "predicted_name", "confidence", "true_name"]]
            .rename(columns={"true_name": "true"}).iloc[::-1], hide_index=True)

if st.session_state.playing and st.session_state.idx < len(df) - 1:
    time.sleep(STEP / 200)      # 20 samples at 200 Hz: the real window step
    st.rerun()                  # the advance itself happens above, pre-widget
