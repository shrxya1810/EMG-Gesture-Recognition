"""Shared loading and metric helpers for train.py / loso.py / ablation.py."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

from build_features import META_COLUMNS, PROV_COLUMNS
from data_loader import GESTURES
from features import FEATURE_GROUPS

RESULTS = Path("results")
LABELS = sorted(GESTURES)


def select_groups(X, names, groups):
    """Keep only the named feature families, e.g. "TD" or "TD,FD".

    Empty or None keeps everything. The ablation found TD alone beats the full
    272, so this is how the winning configuration is selected at train time
    rather than baked into the feature table.
    """
    if not groups:
        return X, names

    unknown = set(groups.split(",")) - set(FEATURE_GROUPS)
    if unknown:
        raise SystemExit(f"unknown feature group(s) {sorted(unknown)}; "
                         f"known: {sorted(FEATURE_GROUPS)}")

    wanted = {c for g in groups.split(",") for c in FEATURE_GROUPS[g]}
    cols = [i for i, n in enumerate(names) if n in wanted]
    if not cols:
        raise SystemExit(f"no columns matched groups={groups}")
    return X[:, cols], [names[i] for i in cols]


def suffix(label):
    """"" -> "", "tuned" -> "_tuned".

    train.py and loso.py write fixed filenames, so without this a second run
    silently overwrites the first. That bit once: a run on the rotation-aligned
    table left discredited numbers sitting in results/loso_summary.csv.
    """
    return f"_{label}" if label else ""


def load_features(path="features.csv"):
    """Return (X, y, meta_df, feature_names).

    Reports how many non-finite values it repaired rather than silently
    zeroing them -- a jump in that count means a feature is broken upstream.
    """
    df = pd.read_csv(path)

    missing = [c for c in META_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{path} has no {missing} column(s). It predates the metadata "
            f"schema -- rebuild it with src/build_features.py."
        )

    # Provenance columns are optional: tables built before PROV_COLUMNS existed
    # still load, and report as "unrecorded" rather than failing.
    meta_cols = META_COLUMNS + [c for c in PROV_COLUMNS if c in df.columns]

    feats = df.drop(columns=meta_cols)
    X = feats.to_numpy(dtype=float)

    n_bad = int(np.sum(~np.isfinite(X)))
    if n_bad:
        print(f"warning: {n_bad} non-finite values "
              f"({100 * n_bad / X.size:.3f}% of the table) replaced with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, df["gesture"].to_numpy(), df[meta_cols], list(feats.columns)


def provenance(meta, groups):
    """What a training run should record about its inputs.

    `groups` comes from the caller because feature selection happens at train
    time, not build time; everything else is read back out of the table.
    """
    def got(c):
        return str(meta[c].iloc[0]) if c in meta.columns and len(meta) else "unrecorded"

    return {"features": groups or "all(272)", "stages": got("stages"),
            "trim": got("trim"), "window": got("window"), "step": got("step")}


def window_groups(meta):
    """Group id that keeps every window of one subject-repetition together.

    Windows overlap by 50%, so neighbours share half their samples. Splitting
    them across folds leaks the test set into training.
    """
    return (meta["subject"] * 100 + meta["repetition"]).to_numpy()


def per_class_report(y_true, y_pred):
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0)

    rows = [{"gesture": g, "name": GESTURES[g], "precision": p[i],
             "recall": r[i], "f1": f1[i], "support": int(sup[i])}
            for i, g in enumerate(LABELS)]

    rows.append({
        "gesture": "macro", "name": "macro avg",
        "precision": p.mean(), "recall": r.mean(), "f1": f1.mean(),
        "support": int(sup.sum()),
    })
    return pd.DataFrame(rows)


def save_confusion(y_true, y_pred, title, path, normalize=True):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    shown = cm / cm.sum(axis=1, keepdims=True) if normalize else cm

    names = [GESTURES[g] for g in LABELS]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=shown.max())

    ax.set(xticks=range(len(LABELS)), yticks=range(len(LABELS)),
           xlabel="predicted", ylabel="true",
           title=f"{title}  (acc {accuracy_score(y_true, y_pred):.3f})")
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{shown[i, j]:.2f}" if normalize else f"{cm[i, j]}",
                    ha="center", va="center", fontsize=8,
                    color="white" if shown[i, j] > shown.max() * 0.6 else "black")

    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return cm
