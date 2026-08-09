"""Leave-one-subject-out evaluation (synopsis advanced objective 1).

    python3 src/loso.py --features features_trim.csv

Reports per-subject accuracy for each classifier, with and without the
subject-independent normalisation, against a within-subject baseline computed
on the *same* feature table. The synopsis target is a cross-subject
degradation of 8 points or less.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import RESULTS, load_features, per_class_report, save_confusion, window_groups
from train import model_zoo

SEED = 42


def pipe_for(est):
    return Pipeline([("scaler", StandardScaler()), ("clf", est)])


def per_subject_normalize(X, subjects):
    """Z-score every feature within each subject.

    Removes per-subject offset and scale, which is where most of the
    inter-subject variability sits. This uses the held-out subject's own
    feature statistics -- unsupervised, no labels, so it is a calibration step
    rather than label leakage, but it does assume you get a batch of that
    subject's data before predicting. Report it as such.
    """
    Xn = X.copy()
    for s in np.unique(subjects):
        m = subjects == s
        Xn[m] = (Xn[m] - Xn[m].mean(0)) / (Xn[m].std(0) + 1e-8)
    return Xn


def within_subject_baseline(X, y, meta, models, folds):
    """Grouped CV on the same table LOSO is about to use.

    Computed here rather than read from results/model_comparison.csv: that file
    may have been produced from a different feature table (different window,
    trimming, alignment), and comparing across tables silently understates the
    cross-subject gap.
    """
    groups = window_groups(meta)
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)

    out = {}
    for name, (est, _) in models.items():
        pred = np.zeros_like(y)
        for tr, te in cv.split(X, y, groups=groups):
            pred[te] = clone(pipe_for(est)).fit(X[tr], y[tr]).predict(X[te])
        out[name] = accuracy_score(y, pred)
        print(f"  {name:12s} within-subject {out[name]:.4f}")
    return out


def run_loso(X, y, subjects, models):
    logo = LeaveOneGroupOut()
    rows, oof = [], {}

    for name, (est, _) in models.items():
        pred = np.zeros_like(y)
        for tr, te in logo.split(X, y, groups=subjects):
            pred[te] = clone(pipe_for(est)).fit(X[tr], y[tr]).predict(X[te])
            acc = accuracy_score(y[te], pred[te])
            rows.append({"model": name, "subject": int(subjects[te][0]),
                         "accuracy": acc})
            print(f"  {name:12s} subject {subjects[te][0]:2d}  acc {acc:.4f}")
        oof[name] = pred

    return pd.DataFrame(rows), oof


def main(args):
    X, y, meta, _ = load_features(args.features)
    subjects = meta["subject"].to_numpy()
    models = model_zoo(quick=True)      # fixed params: LOSO is 10 fits a model

    RESULTS.mkdir(exist_ok=True)

    print("===== within-subject baseline, same feature table =====")
    baseline = within_subject_baseline(X, y, meta, models, args.folds)

    summaries = []
    for tag, Xv in [("raw", X),
                    ("subject_norm", per_subject_normalize(X, subjects))]:
        print(f"\n===== LOSO, {tag} features =====")
        detail, oof = run_loso(Xv, y, subjects, models)
        detail["normalisation"] = tag
        summaries.append(detail)

        if tag == "subject_norm":
            for name, pred in oof.items():
                save_confusion(y, pred, f"{name} LOSO ({tag})",
                               RESULTS / f"confusion_loso_{name}.png")
                per_class_report(y, pred).to_csv(
                    RESULTS / f"per_class_loso_{name}.csv", index=False)

    detail = pd.concat(summaries, ignore_index=True)
    detail.to_csv(RESULTS / "loso_per_subject.csv", index=False)

    summary = (detail.groupby(["model", "normalisation"])["accuracy"]
                     .agg(["mean", "std", "min", "max"]).reset_index())
    summary["within_subject"] = summary["model"].map(baseline)
    summary["degradation_pts"] = 100 * (summary["within_subject"]
                                        - summary["mean"])
    summary["meets_8pt_target"] = summary["degradation_pts"] <= 8

    # ponytail: one fixed output name, so the last run wins. That bit once --
    # a run on features_trim_align.csv left the discredited aligned numbers
    # sitting in the canonical file. Suffix by --features if --align survives;
    # the plan is to delete --align from build_features.py instead.
    summary.to_csv(RESULTS / "loso_summary.csv", index=False)
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--folds", type=int, default=10)
    main(ap.parse_args())
