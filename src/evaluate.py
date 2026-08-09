"""Shared loading and metric helpers for train.py / loso.py / ablation.py."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

from build_features import META_COLUMNS
from data_loader import GESTURES

RESULTS = Path("results")
LABELS = sorted(GESTURES)


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

    feats = df.drop(columns=META_COLUMNS)
    X = feats.to_numpy(dtype=float)

    n_bad = int(np.sum(~np.isfinite(X)))
    if n_bad:
        print(f"warning: {n_bad} non-finite values "
              f"({100 * n_bad / X.size:.3f}% of the table) replaced with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, df["gesture"].to_numpy(), df[META_COLUMNS], list(feats.columns)


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
