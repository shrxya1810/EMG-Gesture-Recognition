"""Ablation studies (synopsis primary objective 3, advanced objective 2).

Feature groups -- which families are actually earning their place:

    python3 src/ablation.py features

Preprocessing stages -- build one table per configuration first, then compare:

    python3 src/build_features.py --stages ""                   -o feat_none.csv
    python3 src/build_features.py --stages bandpass             -o feat_bp.csv
    python3 src/build_features.py --stages bandpass,notch       -o feat_bpn.csv
    python3 src/build_features.py                               -o features.csv
    python3 src/ablation.py preprocessing feat_none.csv feat_bp.csv \\
        feat_bpn.csv features.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import RESULTS, load_features, window_groups
from features import FEATURE_GROUPS

SEED = 42
# Fixed model across every arm: the ablation is about the inputs, so the
# classifier has to be held still.
REFERENCE = Pipeline([("scaler", StandardScaler()),
                      ("clf", RandomForestClassifier(n_estimators=300,
                                                     random_state=SEED,
                                                     n_jobs=-1))])

# Cumulative arms named by the synopsis, plus each family alone for contrast.
FEATURE_ARMS = {
    "TD": ["TD"],
    "FD": ["FD"],
    "DWT": ["DWT"],
    "TD+FD": ["TD", "FD"],
    "TD+FD+DWT": ["TD", "FD", "DWT"],
}


def score(X, y, groups, folds):
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    pred = cross_val_predict(clone(REFERENCE), X, y, groups=groups, cv=cv,
                             n_jobs=1)
    return accuracy_score(y, pred), f1_score(y, pred, average="macro")


def feature_ablation(features_csv, folds):
    X, y, meta, names = load_features(features_csv)
    groups = window_groups(meta)
    index = {n: i for i, n in enumerate(names)}

    rows = []
    for arm, groups_used in FEATURE_ARMS.items():
        cols = [index[c] for g in groups_used for c in FEATURE_GROUPS[g]
                if c in index]
        acc, f1 = score(X[:, cols], y, groups, folds)
        rows.append({"arm": arm, "n_features": len(cols),
                     "accuracy": acc, "macro_f1": f1})
        print(f"  {arm:12s} {len(cols):4d} features  acc {acc:.4f}  f1 {f1:.4f}")

    df = pd.DataFrame(rows)
    full = df.loc[df.arm == "TD+FD+DWT", "accuracy"].iloc[0]
    df["delta_vs_full_pts"] = 100 * (df["accuracy"] - full)
    # objective 2: smallest subset within 2 points of the full set
    df["within_2pts"] = df["delta_vs_full_pts"] >= -2
    return df


def preprocessing_ablation(csvs, folds):
    rows = []
    for path in csvs:
        X, y, meta, _ = load_features(path)
        acc, f1 = score(X, y, window_groups(meta), folds)
        rows.append({"features_file": Path(path).name, "n_windows": len(y),
                     "accuracy": acc, "macro_f1": f1})
        print(f"  {Path(path).name:20s} acc {acc:.4f}  f1 {f1:.4f}")

    df = pd.DataFrame(rows)
    df["delta_vs_first_pts"] = 100 * (df["accuracy"] - df["accuracy"].iloc[0])
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    f = sub.add_parser("features", help="ablate feature families")
    f.add_argument("--features", default="features.csv")
    f.add_argument("--folds", type=int, default=5)

    p = sub.add_parser("preprocessing", help="compare preprocessing configs")
    p.add_argument("csvs", nargs="+")
    p.add_argument("--folds", type=int, default=5)

    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    if args.mode == "features":
        out = feature_ablation(args.features, args.folds)
        out.to_csv(RESULTS / "ablation_features.csv", index=False)
    else:
        out = preprocessing_ablation(args.csvs, args.folds)
        out.to_csv(RESULTS / "ablation_preprocessing.csv", index=False)

    print("\n" + out.to_string(index=False))
