"""1-D CNN to synopsis Table I (primary objective 5).

    python3 src/cnn.py --self-check              # fast, no training
    python3 src/cnn.py --label spec              # Table I as written
    python3 src/cnn.py --label small --widths 32,64,128

Trains under the *same* protocol as `train.py` -- StratifiedGroupKFold(10)
grouped by subject-repetition, same seed -- and writes out-of-fold
probabilities, so `experiments.py`'s smoothing ladder applies unchanged and the
comparison against the classical models is like for like.

**Input is the raw 16x40 window, not the feature table.** Table I does not
specify the input representation. A 1-D convolution slides a kernel along an
ordered axis; the feature table's axis is `[ch00_mav, ch00_rms, ch00_wl, ...]`,
whose ordering is arbitrary, so convolving across it asserts an adjacency that
does not exist. The raw window's axis is time, which is the axis the
architecture was designed for. Rows are reconstructed from `features.csv`'s
`window_start`, so the CNN sees exactly the windows the classical models saw,
in exactly the same folds.

**The input is scaled by one scalar per fold.** This is the single choice that
decides whether the model works at all, and the standard practice is wrong here.
Per-window standardisation divides each window by its own amplitude -- and
PROGRESS.md 5.14 measured that per-channel amplitude *is* the discriminative
signal: 16 `mav` values score 0.854 while the 64 amplitude-free shape features
score 0.457 against a 0.167 chance line. Normalising per window would discard
the signal and leave the network the noise. Per-channel scaling is wrong for the
same reason 5.10a gives for rest-RMS normalisation: it flattens the
between-channel amplitude pattern. So: one constant, computed on the training
fold, applied to every channel and every window alike.

**Expected result.** Global average pooling after ReLU computes, for a zero-mean
signal, the mean rectified response of a learned FIR filter -- that is, the MAV
of a filtered channel mixture. The network's representation is therefore 256
learned filtered MAVs, and it must beat 32 plain ones. 5.10 measured filtering
as actively harmful on this corpus, so parity is the realistic outcome and a
negative result is a legitimate one to report (5.5 is the precedent).
Benchmark: `mav`+`rms` at 0.856 per-window, 0.906 at k=5, 0.928 at k=9.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from build_features import WINDOW_SIZE
from data_loader import load_subject, subject_id
from evaluate import (RESULTS, load_features, per_class_report, provenance,
                      save_confusion, suffix, window_groups)
from features import N_CHANNELS
from preprocessing import DEFAULT_STAGES, preprocess

MODELS_DIR = Path("models")
SEED = 42
TABLE_I_WIDTHS = (64, 128, 256)


def raw_windows(meta, raw_dir="data/raw", stages=DEFAULT_STAGES,
                window=WINDOW_SIZE):
    """Rebuild the (N, 16, window) raw tensor for the rows of a feature table.

    Row i of the output is the window that produced row i of `features.csv`.
    `window_start` already encodes which windows survived trimming and the
    purity filter, so no re-selection happens here -- only slicing. The
    preprocessing stages must match the ones the table was built with, or the
    CNN trains on a differently-conditioned signal than the baseline did.
    """
    starts = meta["window_start"].to_numpy()
    subj = meta["subject"].to_numpy()
    out = np.empty((len(meta), N_CHANNELS, window), dtype=np.float32)

    seen = set()
    for path in sorted(Path(raw_dir).rglob("*E2*.mat")):
        sid = subject_id(path)
        rows = np.flatnonzero(subj == sid)
        if not len(rows):
            continue
        emg, labels, _ = load_subject(path)
        emg = preprocess(emg, rest_mask=(labels == 0), stages=stages)
        for i in rows:
            out[i] = emg[starts[i]:starts[i] + window].T
        seen.add(sid)

    missing = sorted(set(np.unique(subj).tolist()) - seen)
    if missing:
        raise SystemExit(f"no *E2*.mat found under {raw_dir} for subject(s) "
                         f"{missing}; the feature table and the raw data disagree")
    return out


class TableICNN(nn.Module):
    """Three conv layers (kernel 3), BatchNorm, ReLU, GAP, FC, dropout.

    `widths` is the only knob: Table I specifies (64, 128, 256), which is
    160,966 parameters against 7,459 training windows per fold -- 21.6 per
    sample, and 3.4x the ~47k of the Hu et al. network synopsis II calls
    "directly applicable at this project's scale". A narrower setting is the
    planned second configuration, not a speculative option.
    """

    def __init__(self, n_classes, in_ch=N_CHANNELS, widths=TABLE_I_WIDTHS,
                 fc=128, dropout=0.3):
        super().__init__()
        layers, c = [], in_ch
        for w in widths:
            layers += [nn.Conv1d(c, w, 3, padding=1), nn.BatchNorm1d(w),
                       nn.ReLU()]
            c = w
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(c, fc), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fc, n_classes),
        )

    def forward(self, x):
        return self.head(self.conv(x))


def fit_fold(Xtr, ytr, Xte, n_classes, args, seed):
    """Train one fold, return (model, scale, test probabilities)."""
    torch.manual_seed(seed)

    # One scalar, training fold only. See the module docstring -- per-window or
    # per-channel scaling would remove the amplitude the task runs on.
    scale = float(Xtr.std()) or 1.0

    model = TableICNN(n_classes, in_ch=Xtr.shape[1],
                      widths=tuple(int(w) for w in args.widths.split(",")),
                      dropout=args.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()

    Xt = torch.from_numpy(Xtr / scale)
    yt = torch.from_numpy(ytr)

    model.train()
    for _ in range(args.epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), args.batch):
            idx = perm[i:i + args.batch]
            # BatchNorm cannot compute batch statistics over one sample, and a
            # trailing batch of one is a real possibility at these sizes.
            if len(idx) < 2:
                continue
            opt.zero_grad()
            lossf(model(Xt[idx]), yt[idx]).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(Xte / scale))
        P = torch.softmax(logits, dim=1).numpy()
    return model, scale, P


def main(args):
    torch.set_num_threads(args.threads)

    _, y, meta, _ = load_features(args.features)
    X = raw_windows(meta, args.raw_dir,
                    tuple(s for s in args.stages.split(",") if s))
    groups = window_groups(meta)

    classes = np.unique(y)
    y_idx = np.searchsorted(classes, y).astype(np.int64)
    sfx = suffix(args.label)
    prov = provenance(meta, "raw 16x%d" % X.shape[2])
    print(f"provenance: {prov}")

    print(f"{X.shape[0]} windows, {X.shape[1]}x{X.shape[2]} raw input, "
          f"{meta.subject.nunique()} subjects, {len(np.unique(groups))} groups")
    n_params = sum(p.numel() for p in
                   TableICNN(len(classes),
                             widths=tuple(int(w) for w in args.widths.split(",")
                                          )).parameters())
    print(f"widths {args.widths}, {n_params} parameters, "
          f"{args.epochs} epochs, lr {args.lr}, batch {args.batch}\n")

    RESULTS.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                              random_state=SEED)
    P = np.zeros((len(y), len(classes)), dtype=float)
    fold_acc = []

    for k, (tr, te) in enumerate(cv.split(X, y_idx, groups=groups), 1):
        _, _, P[te] = fit_fold(X[tr], y_idx[tr], X[te], len(classes),
                               args, SEED + k)
        fold_acc.append(accuracy_score(y[te], classes[P[te].argmax(1)]))
        print(f"  fold {k:2d}/{args.folds}  acc {fold_acc[-1]:.4f}")

    fold_acc = np.array(fold_acc)
    oof = classes[P.argmax(1)]
    print(f"\n  CNN: CV {fold_acc.mean():.4f} +/- {fold_acc.std():.4f} | "
          f"macro-F1 {f1_score(y, oof, average='macro'):.4f}")

    per_class_report(y, oof).to_csv(RESULTS / f"per_class_CNN{sfx}.csv",
                                    index=False)
    save_confusion(y, oof, f"1-D CNN (grouped {args.folds}-fold)",
                   RESULTS / f"confusion_CNN{sfx}.png")

    # Probabilities, not just labels: experiments.py smooths over these, and a
    # hard vote throws away the confidence that averaging needs (5.4).
    proba = pd.DataFrame(P, columns=[f"p_{c}" for c in classes])
    pd.concat([meta.reset_index(drop=True), proba], axis=1).to_csv(
        RESULTS / f"cnn_oof{sfx}.csv", index=False)
    pd.DataFrame([{
        "model": "CNN", "in_synopsis": True, **prov, "widths": args.widths,
        "n_params": n_params, "epochs": args.epochs,
        "cv_mean": fold_acc.mean(), "cv_std": fold_acc.std(),
        "macro_f1": f1_score(y, oof, average="macro"),
    }]).to_csv(RESULTS / f"cnn_summary{sfx}.csv", index=False)

    # train.py saves an estimator refit on all the data. Match that: a fold
    # model has seen only 90% of the windows, and *which* fold it came from is
    # an artifact of loop order rather than a choice.
    print("  refitting on all windows for the saved model")
    model, scale, _ = fit_fold(X, y_idx, X[:2], len(classes), args, SEED)
    joblib.dump({"state_dict": model.state_dict(), "scale": scale,
                 "classes": classes, "widths": args.widths,
                 "provenance": dict(prov, label=args.label)},
                MODELS_DIR / f"cnn{sfx}.pkl")
    print(f"wrote results to {RESULTS}/ and models to {MODELS_DIR}/")


def self_check():
    rng = np.random.default_rng(0)

    # Table I fidelity: the spec is 160,966 parameters. If this moves, the
    # architecture no longer matches the document it is benchmarked against.
    m = TableICNN(6)
    assert sum(p.numel() for p in m.parameters()) == 160966

    x = torch.from_numpy(rng.standard_normal((8, N_CHANNELS, WINDOW_SIZE)
                                             ).astype(np.float32))
    m.eval()
    with torch.no_grad():
        assert m(x).shape == (8, 6)

    # A narrower net must still build, and be small enough to be a different
    # answer to the capacity question rather than a rounding of the same one.
    narrow = sum(p.numel() for p in
                 TableICNN(6, widths=(32, 64, 128)).parameters())
    assert narrow < 0.4 * 160966, narrow

    # THE invariant: scaling is one scalar, so two windows differing only in
    # overall amplitude must still differ after it. Per-window normalisation
    # would collapse them, and that is the failure this module exists to avoid.
    base = rng.standard_normal((4, N_CHANNELS, WINDOW_SIZE)).astype(np.float32)
    Xtr = np.concatenate([base, base * 5])
    scale = float(Xtr.std())
    quiet, loud = base / scale, (base * 5) / scale
    assert not np.allclose(quiet, loud)
    assert np.isclose(np.abs(loud).mean() / np.abs(quiet).mean(), 5, rtol=1e-3)

    # The training loop actually descends: overfit a tiny separable set.
    n_cls = 3
    y = np.repeat(np.arange(n_cls), 12).astype(np.int64)
    X = (rng.standard_normal((len(y), N_CHANNELS, WINDOW_SIZE)) * 0.1
         + y[:, None, None]).astype(np.float32)
    args = argparse.Namespace(widths="8,8,8", dropout=0.0, lr=1e-2,
                              epochs=30, batch=12)
    _, _, P = fit_fold(X, y, X, n_cls, args, SEED)
    assert (P.argmax(1) == y).mean() > 0.9, "training loop does not converge"

    # Window reconstruction, only if the dataset is present: the per-channel
    # mean absolute value of every rebuilt window must equal that row's mav
    # columns in features.csv. This is the check that catches a misaligned
    # window_start, which would otherwise train silently on shifted data.
    if Path("features.csv").exists() and Path("data/raw").is_dir():
        from features import FEATURE_GROUPS
        Xf, _, meta, names = load_features("features.csv")
        take = rng.choice(len(meta), 400, replace=False)
        sub = meta.iloc[take].reset_index(drop=True)
        W = raw_windows(sub)
        cols = [names.index(c) for c in FEATURE_GROUPS["mav"]]
        assert np.allclose(np.abs(W).mean(axis=2), Xf[take][:, cols],
                           rtol=1e-4, atol=1e-6), "windows do not match the table"
        print("  window reconstruction verified against features.csv")

    print("cnn self-check ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-check", action="store_true",
                    help="run the fast asserts and exit; trains nothing")
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--label", default="spec")
    ap.add_argument("--stages", default=",".join(DEFAULT_STAGES),
                    help="must match how --features was built")
    ap.add_argument("--widths", default=",".join(map(str, TABLE_I_WIDTHS)),
                    help="conv channel widths; Table I is 64,128,256")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=50)   # Table I
    ap.add_argument("--lr", type=float, default=1e-3)   # Table I
    ap.add_argument("--dropout", type=float, default=0.3)   # Table I
    ap.add_argument("--batch", type=int, default=64,
                    help="not specified by Table I")
    ap.add_argument("--threads", type=int, default=6,
                    help="torch CPU threads; the box has more, be polite")
    a = ap.parse_args()
    self_check() if a.self_check else main(a)
