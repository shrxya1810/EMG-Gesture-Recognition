"""Measure candidate accuracy improvements on a feature table.

    python3 src/experiments.py --features features_trim.csv --label trim
    python3 src/experiments.py --features features_trim.csv --groups TD --label trim_td

Reports, per model: raw per-window accuracy, causal majority vote over k
windows, causal probability averaging over k windows, and an oracle vote within
each known repetition (an upper bound, not an achievable number).
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import RESULTS, load_features, window_groups
from features import FEATURE_GROUPS

SEED = 42
FS = 200
VOTE_K = (3, 5, 7)


def models():
    return {
        "ExtraTrees": Pipeline([
            ("sc", StandardScaler()),
            ("clf", ExtraTreesClassifier(n_estimators=300, random_state=SEED,
                                         n_jobs=-1))]),
        "HistGB": Pipeline([
            ("sc", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(random_state=SEED))]),
    }


def oof_proba(pipe, X, y, groups, folds):
    """Out-of-fold class probabilities under the grouped CV."""
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    classes = np.unique(y)
    P = np.zeros((len(y), len(classes)))
    for tr, te in cv.split(X, y, groups=groups):
        fitted = clone(pipe).fit(X[tr], y[tr])
        P[te] = fitted.predict_proba(X[te])
    return P, classes


def _time_order(meta):
    subj = meta["subject"].to_numpy()
    return np.lexsort((meta["window_start"].to_numpy(), subj)), subj


def causal_vote(pred, meta, k):
    """Trailing majority vote over the last k windows of the same subject.

    Uses no label information and never looks ahead, so it is what a real-time
    controller could actually do. Latency cost is (k-1) window increments.
    """
    order, subj = _time_order(meta)
    p, s = pred[order], subj[order]

    smoothed = p.copy()
    for i in range(len(p)):
        lo = max(0, i - k + 1)
        while lo < i and s[lo] != s[i]:
            lo += 1
        vals, counts = np.unique(p[lo:i + 1], return_counts=True)
        smoothed[i] = vals[np.argmax(counts)]

    out = pred.copy()
    out[order] = smoothed
    return out


def causal_proba(P, meta, k, classes):
    """Same window as causal_vote, but average the probabilities.

    A hard vote throws away every bit of confidence information; averaging
    keeps it, at identical latency and cost.
    """
    order, subj = _time_order(meta)
    Q, s = P[order], subj[order]
    cs = np.vstack([np.zeros(Q.shape[1]), np.cumsum(Q, axis=0)])

    smoothed = np.empty(len(Q), dtype=int)
    for i in range(len(Q)):
        lo = max(0, i - k + 1)
        while lo < i and s[lo] != s[i]:
            lo += 1
        smoothed[i] = classes[np.argmax((cs[i + 1] - cs[lo]) / (i + 1 - lo))]

    out = np.empty(len(P), dtype=int)
    out[order] = smoothed
    return out


def oracle_vote(pred, meta):
    """One vote per known repetition. Requires ground-truth segmentation, so
    this is an upper bound on what smoothing can buy, not a usable result."""
    df = pd.DataFrame({"p": pred, "s": meta["subject"], "g": meta["gesture"],
                       "r": meta["repetition"]})
    return (df.groupby(["s", "g", "r"])["p"]
              .transform(lambda v: v.mode().iloc[0]).to_numpy())


def main(args):
    X, y, meta, names = load_features(args.features)

    if args.groups:
        wanted = {c for g in args.groups.split(",") for c in FEATURE_GROUPS[g]}
        cols = [i for i, n in enumerate(names) if n in wanted]
        X = X[:, cols]

    groups = window_groups(meta)
    step = int(np.median(np.diff(sorted(meta["window_start"].unique()))))

    print(f"{args.label}: {X.shape[0]} windows, {X.shape[1]} features, "
          f"step ~{step} samples ({1000 * step / FS:.0f} ms)")

    rows = []

    def record(model, how, acc, base, pred, k=0):
        rows.append({"config": args.label, "model": model, "smoothing": how,
                     "latency_ms": 1000 * step * k / FS if k else 0,
                     "accuracy": acc, "gain_pts": 100 * (acc - base),
                     "macro_f1": f1_score(y, pred, average="macro")})

    for name, pipe in models().items():
        P, classes = oof_proba(pipe, X, y, groups, args.folds)
        pred = classes[P.argmax(axis=1)]
        base = accuracy_score(y, pred)
        record(name, "none", base, base, pred)
        print(f"  {name:11s} raw              {base:.4f}")

        for k in VOTE_K:
            v = causal_vote(pred, meta, k)
            a = accuracy_score(y, v)
            record(name, f"vote_{k}", a, base, v, k - 1)

            q = causal_proba(P, meta, k, classes)
            b = accuracy_score(y, q)
            record(name, f"proba_{k}", b, base, q, k - 1)

            print(f"  {name:11s} k={k}  vote {a:.4f} ({100*(a-base):+.2f})   "
                  f"proba {b:.4f} ({100*(b-base):+.2f})   "
                  f"+{1000 * step * (k - 1) / FS:.0f} ms")

        o = oracle_vote(pred, meta)
        a = accuracy_score(y, o)
        record(name, "oracle_repetition", a, base, o)
        print(f"  {name:11s} oracle           {a:.4f} ({100*(a-base):+.2f}, "
              f"upper bound)")

    out = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    out.to_csv(RESULTS / f"experiments_{args.label}.csv", index=False)
    print(f"\nwrote results/experiments_{args.label}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--groups", help="comma-separated subset of TD,FD,DWT")
    ap.add_argument("--folds", type=int, default=10)
    main(ap.parse_args())
