"""Run a trained model over a recording (synopsis: model export / inference).

    python3 src/predict.py --mat data/raw/s1/s1/S1_E2_A1.mat \\
        --model models/extratrees_tuned.pkl --groups TD

Slides the same windows as build_features.py, extracts the same features and
prints a prediction per window. This is the offline stand-in for the real-time
loop, and the piece the Streamlit dashboard should call.

--groups must match what the model was trained on. A TD-144 model fed the full
272-feature vector fails on the feature count, which is the first thing that
goes wrong when the default configuration changes.
"""
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from build_features import MIN_PURITY, STEP, WINDOW_SIZE
from data_loader import GESTURES, load_subject
from evaluate import select_groups
from features import FEATURE_NAMES, extract_all
from preprocessing import DEFAULT_STAGES, preprocess


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
    stages = tuple(s for s in args.stages.split(",") if s)
    emg = preprocess(emg, rest_mask=(labels == 0), stages=stages)

    starts, feats, truth, t_feat = [], [], [], []
    for start, w, label in windows(emg, labels):
        if args.gestures_only and label not in GESTURES:
            continue
        t0 = time.perf_counter()
        f = extract_all(w)
        t_feat.append(time.perf_counter() - t0)
        starts.append(start)
        feats.append(f)
        truth.append(label)

    if not feats:
        raise SystemExit("no windows to score")

    F, _ = select_groups(np.asarray(feats), FEATURE_NAMES, args.groups)

    n_expected = getattr(model, "n_features_in_", F.shape[1])
    if n_expected != F.shape[1]:
        raise SystemExit(
            f"{args.model} expects {n_expected} features, got {F.shape[1]}. "
            f"Pass --groups to match how it was trained (TD gives 144)."
        )

    # One window at a time: batching measures throughput, the synopsis target
    # is per-decision latency.
    t_inf = []
    for row in F[:args.latency_windows]:
        t0 = time.perf_counter()
        model.predict(row.reshape(1, -1))
        t_inf.append(time.perf_counter() - t0)

    pred = model.predict(F)

    out = pd.DataFrame({
        "window_start": starts,
        "time_s": np.asarray(starts) / 200.0,
        "predicted": pred,
        "predicted_name": [GESTURES.get(int(p), "?") for p in pred],
        "true": truth,
    })

    if hasattr(model, "predict_proba"):
        out["confidence"] = model.predict_proba(F).max(axis=1)

    scored = out.dropna(subset=["true"])
    if len(scored):
        acc = float((scored["predicted"] == scored["true"]).mean())
        print(f"{len(scored)} labelled windows, accuracy {acc:.4f}")

    ms = lambda v: 1000 * np.asarray(v)
    feat_ms, inf_ms = ms(t_feat), ms(t_inf)
    total = np.median(feat_ms) + np.median(inf_ms)
    window_ms = 1000 * WINDOW_SIZE / 200

    print(f"\nlatency per window ({len(t_feat)} windows, "
          f"{Path(args.model).name}, {F.shape[1]} features)")
    print(f"  feature extraction  median {np.median(feat_ms):6.2f} ms  "
          f"p95 {np.percentile(feat_ms, 95):6.2f} ms")
    print(f"  inference           median {np.median(inf_ms):6.2f} ms  "
          f"p95 {np.percentile(inf_ms, 95):6.2f} ms")
    print(f"  compute total       median {total:6.2f} ms")
    print(f"  + {window_ms:.0f} ms to fill the analysis window "
          f"= {window_ms + total:.2f} ms before a decision exists")
    print(f"  synopsis target is 100 ms end-to-end: "
          f"{'MET' if window_ms + total <= 100 else 'MISSED on the window alone'}")

    if args.latency_csv:
        pd.DataFrame([{
            "model": Path(args.model).name,
            "n_features": F.shape[1],
            "n_windows": len(t_feat),
            "feature_ms_median": np.median(feat_ms),
            "feature_ms_p95": np.percentile(feat_ms, 95),
            "inference_ms_median": np.median(inf_ms),
            "inference_ms_p95": np.percentile(inf_ms, 95),
            "compute_ms_median": total,
            "window_fill_ms": window_ms,
            "end_to_end_ms": window_ms + total,
            "meets_100ms": bool(window_ms + total <= 100),
        }]).to_csv(args.latency_csv, index=False)
        print(f"wrote {args.latency_csv}")

    if args.out:
        out.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    else:
        print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mat", required=True, help="a NinaPro *E2*.mat recording")
    ap.add_argument("--model", default="models/rf.pkl")
    ap.add_argument("--groups", default="",
                    help="feature families the model was trained on, e.g. TD")
    ap.add_argument("--stages", default=",".join(DEFAULT_STAGES),
                    help="preprocessing stages the model was trained with; "
                         "must match build_features.py or the features shift "
                         "under the model")
    ap.add_argument("--out", help="write per-window predictions to this CSV")
    ap.add_argument("--latency-csv", help="write the latency summary here")
    ap.add_argument("--latency-windows", type=int, default=500,
                    help="how many windows to time individually")
    ap.add_argument("--gestures-only", action="store_true",
                    help="score only windows labelled with one of the six gestures")
    main(ap.parse_args())
