"""Benchmark the classifiers from synopsis Table I.

    python3 src/train.py            # full Table I grid search (slow)
    python3 src/train.py --quick    # fixed sensible params, for a fast pass

Two protocols are reported:
  * 10-fold stratified CV, grouped by subject-repetition (leak-free)
  * the NinaPro convention: train on repetitions 1,3,4,6, test on 2,5
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from evaluate import (RESULTS, load_features, per_class_report, provenance,
                      save_confusion, select_groups, suffix, window_groups)

MODELS_DIR = Path("models")
SEED = 42
CANONICAL_TRAIN_REPS = [1, 3, 4, 6]
CANONICAL_TEST_REPS = [2, 5]


def model_zoo(quick=False):
    """Search spaces from synopsis Table I.

    Table I specifies LDA with solver=SVD *and* Ledoit-Wolf shrinkage, which
    scikit-learn rejects -- the SVD solver does not support shrinkage. lsqr
    with shrinkage="auto" is Ledoit-Wolf, so that is used instead.

    ExtraTrees is not in the synopsis. It is kept because it was the previous
    best model, and is flagged as off-spec in the output.
    """
    return {
        "LDA": (LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), {}),
        "SVM": (SVC(kernel="rbf", random_state=SEED),
                {} if quick else {"clf__C": [0.1, 1, 10, 100],
                                  "clf__gamma": ["scale", 1e-3, 1e-2]}),
        "RF": (RandomForestClassifier(n_estimators=300, random_state=SEED,
                                      n_jobs=-1),
               {} if quick else {"clf__n_estimators": [100, 200, 300],
                                 "clf__max_depth": [10, 20, None]}),
        "ExtraTrees": (ExtraTreesClassifier(n_estimators=300, random_state=SEED,
                                            n_jobs=-1),
                       {} if quick else {"clf__n_estimators": [100, 200, 300],
                                         "clf__max_depth": [10, 20, None]}),
    }


def evaluate_model(name, est, grid, X, y, groups, n_splits):
    """Grid-search then cross-validate, both grouped by subject-repetition.

    This is tune-then-evaluate, not fully nested CV: the search sees all the
    data, so the reported CV score is mildly optimistic. Full nesting would
    multiply runtime by the inner fold count, which is what made the earlier
    grid search impractical.
    """
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", est)])
    best_params = {}

    if grid:
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        gs = GridSearchCV(pipe, grid, cv=inner, scoring="accuracy", n_jobs=-1)
        gs.fit(X, y, groups=groups)
        pipe, best_params = gs.best_estimator_, gs.best_params_
        print(f"  best params: {best_params}")

    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                 random_state=SEED)
    oof = np.zeros_like(y)
    classes = np.unique(y)
    # Out-of-fold probabilities as well as labels, so report.py can apply the
    # smoothing ladder without refitting. SVC has no predict_proba unless it is
    # built with probability=True, which Table I does not ask for; those models
    # get a one-hot stand-in and can only be majority-voted, not averaged.
    oof_p = np.zeros((len(y), len(classes)))
    fold_acc = []

    for k, (tr, te) in enumerate(outer.split(X, y, groups=groups), 1):
        fitted = clone(pipe).fit(X[tr], y[tr])
        oof[te] = fitted.predict(X[te])
        if hasattr(fitted, "predict_proba"):
            oof_p[te] = fitted.predict_proba(X[te])
        else:
            oof_p[te, np.searchsorted(classes, oof[te])] = 1.0
        fold_acc.append(accuracy_score(y[te], oof[te]))
        print(f"  fold {k:2d}/{n_splits}  acc {fold_acc[-1]:.4f}")

    return {
        "name": name,
        "estimator": clone(pipe).fit(X, y),
        "best_params": best_params,
        "fold_acc": np.array(fold_acc),
        "oof": oof,
        "oof_proba": oof_p,
        "classes": classes,
        "hard_only": not hasattr(pipe, "predict_proba"),
    }


def canonical_score(pipe, X, y, meta):
    """Held-out score under the NinaPro repetition split."""
    rep = meta["repetition"].to_numpy()
    tr = np.isin(rep, CANONICAL_TRAIN_REPS)
    te = np.isin(rep, CANONICAL_TEST_REPS)
    if not tr.any() or not te.any():
        return np.nan, None
    pred = clone(pipe).fit(X[tr], y[tr]).predict(X[te])
    return accuracy_score(y[te], pred), (y[te], pred)


def wilcoxon_table(results):
    """Two-sided paired Wilcoxon on per-fold accuracy, alpha = 0.05."""
    rows = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a, b = results[i]["fold_acc"], results[j]["fold_acc"]
            if np.allclose(a, b):
                stat, p = np.nan, 1.0
            else:
                stat, p = wilcoxon(a, b)
            rows.append({
                "model_a": results[i]["name"], "model_b": results[j]["name"],
                "mean_a": a.mean(), "mean_b": b.mean(),
                "statistic": stat, "p_value": p,
                "significant": bool(p < 0.05),
            })
    return pd.DataFrame(rows)


def main(args):
    X, y, meta, feature_names = load_features(args.features)
    X, feature_names = select_groups(X, feature_names, args.groups)
    groups = window_groups(meta)
    sfx = suffix(args.label)
    prov = provenance(meta, args.groups)
    print(f"provenance: {prov}")

    print(f"{X.shape[0]} windows, {X.shape[1]} features, "
          f"{meta.subject.nunique()} subjects, {len(np.unique(groups))} groups")
    if args.quick:
        print("--quick: skipping the Table I grid search\n")

    RESULTS.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    results, summary = [], []
    for name, (est, grid) in model_zoo(args.quick).items():
        print(f"\n===== {name} =====")
        res = evaluate_model(name, est, grid, X, y, groups, args.folds)
        results.append(res)

        canon_acc, canon = canonical_score(res["estimator"], X, y, meta)

        report = per_class_report(y, res["oof"])
        report.to_csv(RESULTS / f"per_class_{name}{sfx}.csv", index=False)
        save_confusion(y, res["oof"], f"{name} (grouped {args.folds}-fold)",
                       RESULTS / f"confusion_{name}{sfx}.png")
        if canon is not None:
            save_confusion(*canon, f"{name} (NinaPro repetition split)",
                           RESULTS / f"confusion_{name}_canonical{sfx}.png")

        clf = res["estimator"].named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            (pd.DataFrame({"feature": feature_names,
                           "importance": clf.feature_importances_})
               .sort_values("importance", ascending=False)
               .to_csv(RESULTS / f"feature_importance_{name}{sfx}.csv",
                       index=False))

        proba = pd.DataFrame(res["oof_proba"],
                             columns=[f"p_{c}" for c in res["classes"]])
        pd.concat([meta.reset_index(drop=True), proba], axis=1).to_csv(
            RESULTS / f"oof_{name}{sfx}.csv", index=False)

        res["estimator"].provenance_ = dict(prov, label=args.label)
        joblib.dump(res["estimator"], MODELS_DIR / f"{name.lower()}{sfx}.pkl")

        summary.append({
            "model": name,
            "in_synopsis": name != "ExtraTrees",
            **prov,
            "cv_mean": res["fold_acc"].mean(),
            "cv_std": res["fold_acc"].std(),
            "macro_f1": f1_score(y, res["oof"], average="macro"),
            "canonical_acc": canon_acc,
            "best_params": str(res["best_params"]),
        })
        print(f"  {name}: CV {summary[-1]['cv_mean']:.4f} "
              f"+/- {summary[-1]['cv_std']:.4f} | "
              f"canonical {canon_acc:.4f} | macro-F1 {summary[-1]['macro_f1']:.4f}")

    summary = pd.DataFrame(summary).sort_values("cv_mean", ascending=False)
    summary.to_csv(RESULTS / f"model_comparison{sfx}.csv", index=False)

    pd.DataFrame({r["name"]: r["fold_acc"] for r in results}).to_csv(
        RESULTS / f"fold_accuracy{sfx}.csv", index_label="fold")
    wilcoxon_table(results).to_csv(RESULTS / f"wilcoxon{sfx}.csv", index=False)

    print("\n" + summary.to_string(index=False))
    print(f"\nwrote results to {RESULTS}/ and models to {MODELS_DIR}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--groups", default="",
                    help="feature families to keep, e.g. TD or TD,FD "
                         "(default: all 272)")
    ap.add_argument("--label", default="",
                    help="suffix for result and model filenames, so a second "
                         "run does not overwrite the first")
    ap.add_argument("--quick", action="store_true",
                    help="skip the grid search and use fixed parameters")
    main(ap.parse_args())
