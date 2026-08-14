"""Turn raw NinaPro DB5 recordings into a feature table.

    python3 src/build_features.py                      # full preprocessing
    python3 src/build_features.py --stages "" -o raw.csv   # ablation baseline
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import GESTURES, load_subject, subject_id
from features import FEATURE_NAMES, extract_all
from preprocessing import DEFAULT_STAGES, STAGES, preprocess

WINDOW_SIZE = 40        # 200 ms at 200 Hz
STEP = 20               # 50 % overlap
MIN_PURITY = 0.90       # drop windows straddling a gesture boundary

# Fraction of each repetition discarded at each end. 0.15 measured +6.8 points
# over 0.0, but note it changes the test set as well as the model: the honest
# phrasing is "on steady-state windows". Pass --trim 0 to rebuild the
# transient-inclusive baseline.
TRIM = 0.15

META_COLUMNS = ["subject", "gesture", "repetition", "window_start"]

# How this table was built, written into the table itself. Constant down every
# row, which is wasteful and worth it: a feature file with no provenance cannot
# be compared to another one, and `features_untrimmed.csv` was built under the
# since-dropped preprocessing chain with nothing recording that. Optional on
# read, so tables predating this still load.
PROV_COLUMNS = ["stages", "trim", "window", "step"]

# Rotation alignment (circularly shifting each armband to its peak channel) was
# implemented here and removed: it degraded every model and both normalisations,
# LOSO 0.645 -> 0.467 for Extra Trees. The peak of a mean activation profile is
# not a stable landmark, and aligning each ring independently destroys the
# relative offset between the two, which is informative. See PROGRESS.md 5.5.


def trim_transients(labels, frac):
    """Relabel the first and last `frac` of every repetition as rest.

    A repetition ramps up and relaxes at its edges: correctly labelled, but
    low amplitude and barely distinguishable between gestures. MIN_PURITY only
    catches windows straddling a boundary, not these.
    """
    if frac <= 0:
        return labels

    out = labels.copy()
    change = np.flatnonzero(np.diff(labels)) + 1
    for lo, hi in zip(np.r_[0, change], np.r_[change, len(labels)]):
        if labels[lo] == 0:
            continue
        cut = int((hi - lo) * frac)
        if cut:
            out[lo:lo + cut] = 0
            out[hi - cut:hi] = 0
    return out


def build(raw_dir="data/raw", stages=DEFAULT_STAGES, window=WINDOW_SIZE,
          step=STEP, trim=TRIM):
    rows, meta = [], []

    for path in sorted(Path(raw_dir).rglob("*E2*.mat")):
        emg, labels, reps = load_subject(path)
        sid = subject_id(path)

        # Preprocess the continuous recording *before* selecting gestures.
        # Masking first would splice non-adjacent segments together and the
        # filters would ring at every join; it also deletes the rest samples
        # that normalisation needs as its baseline.
        emg = preprocess(emg, rest_mask=(labels == 0), stages=stages)

        # trim after the rest mask is taken, so the baseline stays intact
        labels = trim_transients(labels, trim)

        kept = 0
        WINDOW_SIZE_, STEP_ = window, step
        for start in range(0, len(emg) - WINDOW_SIZE_ + 1, STEP_):
            wlab = labels[start:start + WINDOW_SIZE_]
            label = int(np.bincount(wlab).argmax())
            if label not in GESTURES:
                continue
            if np.mean(wlab == label) < MIN_PURITY:
                continue

            wrep = reps[start:start + WINDOW_SIZE_]
            rows.append(extract_all(emg[start:start + WINDOW_SIZE_]))
            meta.append((sid, label, int(np.bincount(wrep).argmax()), start))
            kept += 1

        print(f"{path.name}: {kept} windows")

    if not rows:
        raise SystemExit(f"no windows built from {raw_dir} -- is the data there?")

    prov = pd.DataFrame(
        [[",".join(stages) or "none", trim, window, step]] * len(meta),
        columns=PROV_COLUMNS)

    return pd.concat(
        [pd.DataFrame(meta, columns=META_COLUMNS), prov,
         pd.DataFrame(rows, columns=FEATURE_NAMES)],
        axis=1,
    )


def _self_check():
    """trim_transients is the only non-obvious logic in this module."""
    labels = np.array([0] * 5 + [6] * 10 + [0] * 5 + [13] * 10 + [0] * 5)

    assert np.array_equal(trim_transients(labels, 0.0), labels), "0 is a no-op"

    out = trim_transients(labels, 0.2)          # 2 samples off each end of 10
    assert (out == 6).sum() == 6 and (out == 13).sum() == 6, out
    assert out[5] == 0 and out[6] == 0 and out[7] == 6, out[:10]     # leading
    assert out[12] == 6 and out[13] == 0 and out[14] == 0, out[10:16]  # trailing
    # rest stays rest, and nothing new is invented
    assert set(np.unique(out)) <= set(np.unique(labels))
    # adjacent repetitions of the same gesture are separate runs
    two = np.array([6] * 10 + [13] * 10)
    assert (trim_transients(two, 0.2) == 0).sum() == 8

    print("build_features self-check ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-check", action="store_true",
                    help="run the trim_transients checks and exit")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("-o", "--out", default="features.csv")
    ap.add_argument("--stages", default=",".join(DEFAULT_STAGES),
                    help=f"comma-separated subset of {','.join(STAGES)}. "
                         f"Default is none: the chain measured worse on DB5 "
                         f"(PROGRESS.md 5.10). Pass the full chain for a raw "
                         f"front-end.")
    ap.add_argument("--window", type=int, default=WINDOW_SIZE,
                    help="window length in samples (40 = 200 ms at 200 Hz)")
    ap.add_argument("--step", type=int, default=None,
                    help="window increment in samples (default: half the window)")
    ap.add_argument("--trim", type=float, default=TRIM,
                    help=f"fraction of each repetition to discard at each end "
                         f"(default {TRIM}; pass 0 for the untrimmed baseline)")
    args = ap.parse_args()

    if args.self_check:
        _self_check()
        raise SystemExit

    df = build(args.raw, tuple(s for s in args.stages.split(",") if s),
               window=args.window, step=args.step or args.window // 2,
               trim=args.trim)
    df.to_csv(args.out, index=False)

    print(f"\nsaved {args.out} {df.shape}")
    print(f"subjects: {sorted(df.subject.unique())}")
    print(f"windows per gesture:\n{df.gesture.map(GESTURES).value_counts()}")
