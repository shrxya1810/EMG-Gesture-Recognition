"""Run a trained model over a recording (synopsis: model export / inference).

    python3 src/predict.py --mat data/raw/s1/s1/S1_E2_A1.mat

Slides the same windows as build_features.py, extracts the same features and
prints a prediction per window. This is the offline stand-in for the real-time
loop, and the piece the Streamlit dashboard should call.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from build_features import MIN_PURITY, STEP, WINDOW_SIZE
from data_loader import GESTURES, load_subject
from features import extract_all
from preprocessing import preprocess


def windows(emg, labels):
    """Yield (start, window, true_label_or_None) over a whole recording."""
    for start in range(0, len(emg) - WINDOW_SIZE + 1, STEP):
        wlab = labels[start:start + WINDOW_SIZE]
        label = int(np.bincount(wlab).argmax())
        pure = np.mean(wlab == label) >= MIN_PURITY
        yield start, emg[start:start + WINDOW_SIZE], (label if pure else None)


def main(args):
    model = joblib.load(args.model)

    emg, labels, _ = load_subject(args.mat)
    emg = preprocess(emg, rest_mask=(labels == 0))

    starts, feats, truth = [], [], []
    for start, w, label in windows(emg, labels):
        if args.gestures_only and label not in GESTURES:
            continue
        starts.append(start)
        feats.append(extract_all(w))
        truth.append(label)

    if not feats:
        raise SystemExit("no windows to score")

    pred = model.predict(np.asarray(feats))

    out = pd.DataFrame({
        "window_start": starts,
        "time_s": np.asarray(starts) / 200.0,
        "predicted": pred,
        "predicted_name": [GESTURES.get(int(p), "?") for p in pred],
        "true": truth,
    })

    if hasattr(model, "predict_proba"):
        out["confidence"] = model.predict_proba(np.asarray(feats)).max(axis=1)

    scored = out.dropna(subset=["true"])
    if len(scored):
        acc = float((scored["predicted"] == scored["true"]).mean())
        print(f"{len(scored)} labelled windows, accuracy {acc:.4f}")

    if args.out:
        out.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    else:
        print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mat", required=True, help="a NinaPro *E2*.mat recording")
    ap.add_argument("--model", default="models/rf.pkl")
    ap.add_argument("--out", help="write per-window predictions to this CSV")
    ap.add_argument("--gestures-only", action="store_true",
                    help="score only windows labelled with one of the six gestures")
    main(ap.parse_args())
