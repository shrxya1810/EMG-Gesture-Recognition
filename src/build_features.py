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
from preprocessing import STAGES, preprocess

WINDOW_SIZE = 40        # 200 ms at 200 Hz
STEP = 20               # 50 % overlap
MIN_PURITY = 0.90       # drop windows straddling a gesture boundary
TRIM = 0.0              # fraction of each repetition to discard at each end

RING = 8                # DB5 stacks two 8-electrode Myo armbands

META_COLUMNS = ["subject", "gesture", "repetition", "window_start"]


def align_rings(emg, mask, ring=RING):
    """Circularly shift each armband so its strongest channel sits first.

    The Myo sits at a different rotational offset on every forearm, which
    rotates the whole spatial activation pattern. The EDA measured forearm
    supination at only 0.16 mean cross-subject correlation with negative
    minima, i.e. inverted patterns between some subject pairs.

    The shift is computed once per subject from the whole recording, never per
    window: a per-window shift would move every gesture's peak to channel 0 and
    destroy exactly the spatial information that separates the gestures.
    """
    profile = np.abs(emg[mask]).mean(0)
    out = emg.copy()
    for lo in range(0, emg.shape[1], ring):
        shift = int(np.argmax(profile[lo:lo + ring]))
        out[:, lo:lo + ring] = np.roll(emg[:, lo:lo + ring], -shift, axis=1)
    return out


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


def build(raw_dir="data/raw", stages=STAGES, window=WINDOW_SIZE, step=STEP,
          trim=TRIM, align=False):
    rows, meta = [], []

    for path in sorted(Path(raw_dir).rglob("*E2*.mat")):
        emg, labels, reps = load_subject(path)
        sid = subject_id(path)

        # Preprocess the continuous recording *before* selecting gestures.
        # Masking first would splice non-adjacent segments together and the
        # filters would ring at every join; it also deletes the rest samples
        # that normalisation needs as its baseline.
        emg = preprocess(emg, rest_mask=(labels == 0), stages=stages)

        if align:
            emg = align_rings(emg, np.isin(labels, list(GESTURES)))

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

    return pd.concat(
        [pd.DataFrame(meta, columns=META_COLUMNS),
         pd.DataFrame(rows, columns=FEATURE_NAMES)],
        axis=1,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("-o", "--out", default="features.csv")
    ap.add_argument("--stages", default=",".join(STAGES),
                    help=f"comma-separated subset of {','.join(STAGES)}; "
                         f"empty string disables preprocessing entirely")
    ap.add_argument("--window", type=int, default=WINDOW_SIZE,
                    help="window length in samples (40 = 200 ms at 200 Hz)")
    ap.add_argument("--step", type=int, default=None,
                    help="window increment in samples (default: half the window)")
    ap.add_argument("--trim", type=float, default=TRIM,
                    help="fraction of each repetition to discard at each end")
    ap.add_argument("--align", action="store_true",
                    help="rotation-align each armband's channels per subject")
    args = ap.parse_args()

    df = build(args.raw, tuple(s for s in args.stages.split(",") if s),
               window=args.window, step=args.step or args.window // 2,
               trim=args.trim, align=args.align)
    df.to_csv(args.out, index=False)

    print(f"\nsaved {args.out} {df.shape}")
    print(f"subjects: {sorted(df.subject.unique())}")
    print(f"windows per gesture:\n{df.gesture.map(GESTURES).value_counts()}")
