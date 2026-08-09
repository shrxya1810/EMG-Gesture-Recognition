"""Exploratory data analysis on NinaPro DB5 (synopsis primary objective 2).

    python3 src/eda.py

Writes per-channel signal statistics, class distribution, per-gesture spectral
profiles and an inter-subject variability quantification into results/.
Reads the raw recordings, so it does not need features.csv.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

from data_loader import GESTURES, load_subject, subject_id
from evaluate import RESULTS
from preprocessing import FS

LABELS = sorted(GESTURES)


def collect(raw_dir):
    """One pass over the recordings, gathering everything the report needs."""
    stats, counts, patterns, spectra = [], [], {}, {}

    for path in sorted(Path(raw_dir).rglob("*E2*.mat")):
        emg, labels, _ = load_subject(path)
        sid = subject_id(path)
        rest = np.abs(emg[labels == 0]).mean(0) + 1e-12

        for ch in range(emg.shape[1]):
            active = np.abs(emg[np.isin(labels, LABELS), ch])
            stats.append({
                "subject": sid, "channel": ch,
                "rest_mean_abs": rest[ch],
                "active_mean_abs": active.mean(),
                "active_rms": np.sqrt(np.mean(emg[np.isin(labels, LABELS), ch] ** 2)),
                "activation_ratio": active.mean() / rest[ch],
            })

        for g in LABELS:
            m = labels == g
            counts.append({"subject": sid, "gesture": g, "name": GESTURES[g],
                           "samples": int(m.sum()),
                           "seconds": float(m.sum()) / FS})

            # channel activation pattern, rest-normalised then z-scored so
            # subjects are comparable regardless of overall gain
            v = np.abs(emg[m]).mean(0) / rest
            patterns.setdefault(g, {})[sid] = (v - v.mean()) / (v.std() + 1e-12)

            f, p = welch(emg[m], fs=FS, axis=0,
                         nperseg=min(256, int(m.sum())))
            spectra.setdefault(g, []).append(p.mean(axis=1))

    return (pd.DataFrame(stats), pd.DataFrame(counts), patterns,
            (f, {g: np.mean(v, axis=0) for g, v in spectra.items()}))


def intersubject(patterns):
    """How consistent is a gesture's spatial pattern across subjects?

    Mean pairwise correlation between subjects' channel-activation patterns.
    High means the gesture looks the same on everyone, so a cross-subject model
    has a chance; low is where LOSO accuracy will be lost.
    """
    rows = []
    for g, by_subject in patterns.items():
        subs = sorted(by_subject)
        M = np.array([by_subject[s] for s in subs])
        cors = [np.corrcoef(M[i], M[j])[0, 1]
                for i in range(len(subs)) for j in range(i + 1, len(subs))]
        rows.append({"gesture": g, "name": GESTURES[g],
                     "mean_pairwise_corr": float(np.mean(cors)),
                     "min_pairwise_corr": float(np.min(cors)),
                     "std_pairwise_corr": float(np.std(cors))})
    return pd.DataFrame(rows).sort_values("mean_pairwise_corr")


def plots(stats, counts, spec, inter):
    freqs, psd = spec

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    piv = stats.pivot(index="subject", columns="channel",
                      values="activation_ratio")
    im = ax[0].imshow(piv.to_numpy(), aspect="auto", cmap="viridis")
    ax[0].set(xlabel="channel", ylabel="subject",
              title="activation ratio (active / rest)")
    ax[0].set_yticks(range(len(piv)), piv.index)
    fig.colorbar(im, ax=ax[0], fraction=0.046)

    tot = counts.groupby("name")["seconds"].sum().sort_values()
    ax[1].barh(tot.index, tot.to_numpy())
    ax[1].set(xlabel="seconds across all subjects", title="class distribution")
    fig.tight_layout()
    fig.savefig(RESULTS / "eda_overview.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for g in sorted(psd):
        ax.semilogy(freqs, psd[g], label=GESTURES[g], lw=1.3)
    ax.set(xlabel="frequency (Hz)", ylabel="PSD",
           title="mean power spectrum per gesture")
    ax.axvline(50, ls=":", c="grey", lw=1)
    ax.text(51, ax.get_ylim()[1], " 50 Hz mains", va="top", fontsize=8,
            color="grey")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "eda_spectra.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(inter["name"], inter["mean_pairwise_corr"])
    ax.set(xlabel="mean pairwise inter-subject correlation",
           title="spatial pattern consistency across subjects")
    fig.tight_layout()
    fig.savefig(RESULTS / "eda_intersubject.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    stats, counts, patterns, spec = collect(args.raw)
    inter = intersubject(patterns)

    stats.to_csv(RESULTS / "eda_channel_stats.csv", index=False)
    counts.to_csv(RESULTS / "eda_class_distribution.csv", index=False)
    inter.to_csv(RESULTS / "eda_intersubject.csv", index=False)
    plots(stats, counts, spec, inter)

    print("channel activation ratio, pooled over subjects:")
    print(stats.groupby("channel")["activation_ratio"]
               .agg(["mean", "std"]).to_string())
    print("\nclass balance (seconds, all subjects):")
    print(counts.groupby("name")["seconds"].sum().to_string())
    print("\ninter-subject consistency, least consistent first:")
    print(inter.to_string(index=False))
    print(f"\nwrote tables and figures to {RESULTS}/")
