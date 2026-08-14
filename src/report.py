"""Scoreboard for every trained model, and what each predicts on one window.

    python3 src/report.py                                     # scores only
    python3 src/report.py --mat data/raw/s1/s1/S1_E2_A1.mat --at 12.5
    python3 src/report.py --self-check                        # loads every model

Scores are read from `results/`, not recomputed -- they are the numbers the
training runs recorded, and recomputing them here would be a second code path
that could disagree with the first.
"""
import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from build_features import WINDOW_SIZE
from cnn import TableICNN
from data_loader import GESTURES, load_subject
from evaluate import select_groups
from features import FS, FEATURE_NAMES, N_CHANNELS, extract_all
from preprocessing import DEFAULT_STAGES, preprocess
from experiments import causal_proba
from rest_gate import K, calibrate, is_gesture

MODELS = Path("models")
RESULTS = Path("results")

# What each `train.py --label` run actually was. The artifacts do not record it:
# a pickle stores the feature *count*, not which columns, and neither the pickle
# nor the results CSV stores the preprocessing chain or the trim at all. So two
# rows can both read "TD" and still not be comparable.
#
# Reconstructed from PROGRESS.md and checkable against the scores recorded
# there: original = §5.1/§5.4 (0.742), td_untuned/tuned = §5.11 (0.8306),
# nopp = §5.10a (0.8386), mavrms = §5.14 (0.8565).
#
# train.py and cnn.py now stamp this into their own outputs, so this table is
# only a fallback for results written before that. The configs listed here are
# the reconstructed ones; anything not in this dict came from the run itself.
CONFIGS = {
    #             features      preprocessing   trim
    "original":  ("TD,FD,DWT",  "full chain",   "none"),
    "td_untuned": ("TD",        "full chain",   "15%"),
    "tuned":     ("TD",         "full chain",   "15%"),
    "nopp":      ("TD",         "none",         "15%"),
    "mavrms":    ("mav,rms",    "none",         "15%"),
}
LABEL_GROUPS = {k: v[0] for k, v in CONFIGS.items()}


def scoreboard():
    """Every recorded model score, classical and CNN, in one table."""
    rows = []

    for p in sorted(RESULTS.glob("model_comparison*.csv")):
        label = p.stem.replace("model_comparison", "").lstrip("_") or "original"
        for _, r in pd.read_csv(p).iterrows():
            if "stages" in r and isinstance(r.get("features"), str):
                feats, prep, trim = r["features"], r["stages"], str(r["trim"])
            else:
                feats, prep, trim = CONFIGS.get(label, ("?", "?", "?"))
            rows.append({"model": r["model"], "config": label,
                         "features": feats, "preproc": prep, "trim": trim,
                         "cv_mean": r["cv_mean"], "cv_std": r["cv_std"],
                         "macro_f1": r["macro_f1"],
                         "canonical": r.get("canonical_acc", np.nan)})

    for p in sorted(RESULTS.glob("cnn_summary_*.csv")):
        label = p.stem.replace("cnn_summary_", "")
        for _, r in pd.read_csv(p).iterrows():
            rows.append({"model": f"CNN ({r['n_params']:,} params)",
                         "config": label,
                         "features": r.get("features", "raw 16x40"),
                         "preproc": r.get("stages", "none"),
                         "trim": str(r.get("trim",
                                     "0.0" if label == "untrimmed" else "0.15")),
                         "cv_mean": r["cv_mean"], "cv_std": r["cv_std"],
                         "macro_f1": r["macro_f1"], "canonical": np.nan})

    if not rows:
        raise SystemExit(f"no results found in {RESULTS}/ -- run train.py first")

    # Attach the smoothing ladder wherever out-of-fold probabilities exist.
    for r in rows:
        stem = ("cnn_oof_" + r["config"] if r["model"].startswith("CNN")
                else f"oof_{r['model']}_{r['config']}")
        f = RESULTS / f"{stem}.csv"
        if f.exists():
            lad = smoothed(f)
            best = max(lad, key=lad.get)
            # A peak at the top of the ladder is a boundary hit, not a maximum
            # -- the same trap 5.11 records for the Table I grid. Flag it.
            edge = "+" if best == LADDER[-1] else ""
            r["k=5"] = f"{lad[5]:.4f}"
            r["k=9"] = f"{lad[9]:.4f}"
            r["best"] = f"{lad[best]:.4f} @k={best}{edge}"
            r["best_ms"] = 200 + (best - 1) * 100
        else:
            r["k=5"] = r["k=9"] = r["best"] = "-"
            r["best_ms"] = np.nan

    df = pd.DataFrame(rows).sort_values("cv_mean", ascending=False)
    df["cv"] = [f"{m:.4f} +/- {s:.3f}" for m, s in zip(df.cv_mean, df.cv_std)]
    df["canonical"] = [f"{v:.4f}" if np.isfinite(v) else "-" for v in df.canonical]
    df["macro_f1"] = df.macro_f1.map("{:.4f}".format)

    cols = ["model", "config", "features", "preproc", "trim", "cv",
            "k=5", "k=9", "best"]
    print(df[cols].to_string(index=False))
    print(f"\n{len(df)} models. cv = per-window, grouped 10-fold. k = causal "
          f"probability averaging over a trailing k windows, costing "
          f"(k-1) x 100 ms on top of the 200 ms window.")
    print("'-' means no out-of-fold probabilities were saved for that run; "
          "rerun train.py to generate them. A '+' on the best k means the "
          "optimum sits at the top of LADDER and may lie beyond it.")
    print("Rows differing in features/preproc/trim are NOT comparable.")


# Smoothing is a runtime dial on stored probabilities, not a hyperparameter, so
# one trained model serves this whole range. PROGRESS.md 5.16: the curve peaks
# around k=13 and declines after, because the buffer never resets at a gesture
# boundary and the contaminated fraction grows linearly with k.
LADDER = (1, 3, 5, 7, 9, 11, 13, 15, 17, 21, 25, 31, 41)


def smoothed(path):
    """Per-window and smoothed accuracy from one out-of-fold probability file.

    Returns {k: accuracy}. Latency is 200 ms for the analysis window plus
    (k-1) x 100 ms of trailing buffer.
    """
    d = pd.read_csv(path)
    pcols = [c for c in d.columns if c.startswith("p_")]
    classes = np.array([int(c[2:]) for c in pcols])
    P, y = d[pcols].to_numpy(), d["gesture"].to_numpy()
    meta = d[["subject", "gesture", "repetition", "window_start"]]

    out = {1: float((classes[P.argmax(1)] == y).mean())}
    for k in LADDER[1:]:
        out[k] = float((causal_proba(P, meta, k, classes) == y).mean())
    return out


def load_any(path):
    """Return (predict_fn, feature_kind) for a classical or CNN pickle."""
    obj = joblib.load(path)

    if isinstance(obj, dict) and "state_dict" in obj:        # CNN
        widths = tuple(int(w) for w in obj["widths"].split(","))
        net = TableICNN(len(obj["classes"]), widths=widths)
        net.load_state_dict(obj["state_dict"])
        net.eval()
        classes, scale = np.asarray(obj["classes"]), obj["scale"]

        def predict(window):
            x = torch.from_numpy((window.T[None] / scale).astype(np.float32))
            with torch.no_grad():
                p = torch.softmax(net(x), dim=1).numpy()[0]
            return classes[p.argmax()], p.max()

        return predict, "raw 16x40"

    label = re.sub(r"^[a-z]+_", "", path.stem)               # extratrees_mavrms
    groups = LABEL_GROUPS.get(label)
    if groups is None:
        raise SystemExit(f"{path.name}: unknown label {label!r}. Add it to "
                         f"LABEL_GROUPS so the feature subset is unambiguous.")

    def predict(window):
        F, _ = select_groups(np.asarray(extract_all(window))[None],
                             FEATURE_NAMES, groups)
        if F.shape[1] != obj.n_features_in_:
            raise SystemExit(f"{path.name} wants {obj.n_features_in_} features, "
                             f"{groups} gives {F.shape[1]}")
        pred = obj.predict(F)[0]
        conf = obj.predict_proba(F).max() if hasattr(obj, "predict_proba") else np.nan
        return pred, conf

    return predict, groups


def predict_one(args):
    emg, labels, _ = load_subject(args.mat)
    rest_mask = labels == 0
    emg = preprocess(emg, rest_mask=rest_mask,
                     stages=tuple(s for s in args.stages.split(",") if s))

    start = int(args.at * FS)
    if not 0 <= start <= len(emg) - WINDOW_SIZE:
        raise SystemExit(f"--at {args.at}s is outside the recording "
                         f"(0 to {(len(emg) - WINDOW_SIZE) / FS:.1f}s)")

    window = emg[start:start + WINDOW_SIZE]
    truth = int(np.bincount(labels[start:start + WINDOW_SIZE]).argmax())
    rest_mav = calibrate(emg, rest_mask)
    passed = is_gesture(window, rest_mav, args.rest_k)

    print(f"\n{Path(args.mat).name} at {args.at:.2f}s "
          f"(samples {start}-{start + WINDOW_SIZE})")
    print(f"  true label      {truth} "
          f"({GESTURES.get(truth, 'rest' if truth == 0 else 'other gesture')})")
    print(f"  window MAV      {np.abs(window).mean():.3f}  "
          f"(resting {rest_mav:.3f}, gate at {args.rest_k} x = "
          f"{args.rest_k * rest_mav:.3f})")
    print(f"  rest gate       {'PASS - classify it' if passed else 'GATED - report rest'}")

    if not passed and not args.force:
        print("\n  (gated, so no model is consulted; --force to run them anyway)")
        return

    rows = []
    for p in sorted(MODELS.glob("*.pkl")):
        predict, kind = load_any(p)
        pred, conf = predict(window)
        rows.append({"model": p.stem, "input": kind,
                     "predicted": f"{int(pred)} {GESTURES.get(int(pred), '?')}",
                     "confidence": f"{conf:.3f}" if np.isfinite(conf) else "-",
                     "correct": "yes" if int(pred) == truth else "no"})

    print()
    print(pd.DataFrame(rows).to_string(index=False))
    if truth not in GESTURES:
        print("\n  NOTE: the true label is not one of the six trained gestures, "
              "so every 'correct' above is 'no' by construction.")


def self_check():
    rng = np.random.default_rng(0)
    window = (rng.standard_normal((WINDOW_SIZE, N_CHANNELS)) * 12).astype(float)

    found = sorted(MODELS.glob("*.pkl"))
    assert found, "no models on disk to check"

    for p in found:
        predict, kind = load_any(p)
        pred, conf = predict(window)
        assert int(pred) in GESTURES, f"{p.name} predicted {pred}, not a gesture"
        assert not np.isfinite(conf) or 0 <= conf <= 1, f"{p.name} conf {conf}"

    assert set(LABEL_GROUPS) >= {
        re.sub(r"^[a-z]+_", "", p.stem) for p in found
        if not p.stem.startswith("cnn")}, "a model label has no LABEL_GROUPS entry"

    print(f"report self-check ok: {len(found)} models load and predict")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mat", help="a NinaPro *E2*.mat recording")
    ap.add_argument("--at", type=float, default=0.0,
                    help="seconds into the recording to classify")
    ap.add_argument("--stages", default=",".join(DEFAULT_STAGES))
    ap.add_argument("--rest-k", type=float, default=K)
    ap.add_argument("--force", action="store_true",
                    help="consult the models even when the gate says rest")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()

    if a.self_check:
        self_check()
    else:
        scoreboard()
        if a.mat:
            predict_one(a)
