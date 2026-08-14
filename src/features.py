import os
from functools import lru_cache

import numpy as np
import pywt
from scipy.stats import kurtosis, skew

FS = 200
N_CHANNELS = 16
DWT_WAVELET = 'db4'
DWT_LEVEL = 3
# Energy ratios are amplitude-invariant, and the ablation showed amplitude is
# where the discriminative signal lives -- ratios alone scored near chance.
# Set true to emit log absolute subband energy instead. Env-overridable so a
# variant feature table can be built without editing the file.
DWT_ABSOLUTE = os.environ.get("DWT_ABSOLUTE", "0") == "1"

# Zero-crossing / slope-sign-change deadzone, in units of resting RMS (the
# signal is rest-normalised upstream, so 1.0 is roughly resting amplitude).
# Without it these two count noise crossings. Tune if electrodes change.
ZC_EPS = 0.1

TD_NAMES = ["mav", "rms", "wl", "zc", "ssc", "iemg", "var", "skew", "kurt"]
FD_NAMES = ["mnf", "mdf", "sent", "dom"]
DWT_NAMES = ["dwt_a3", "dwt_d3", "dwt_d2", "dwt_d1"]   # wavedec order

FEATURE_NAMES = (
    [f"ch{c:02d}_{n}" for c in range(N_CHANNELS) for n in TD_NAMES]
    + [f"ch{c:02d}_{n}" for c in range(N_CHANNELS) for n in FD_NAMES]
    + [f"ch{c:02d}_{n}" for c in range(N_CHANNELS) for n in DWT_NAMES]
)

FEATURE_GROUPS = {
    "TD": [f"ch{c:02d}_{n}" for c in range(N_CHANNELS) for n in TD_NAMES],
    "FD": [f"ch{c:02d}_{n}" for c in range(N_CHANNELS) for n in FD_NAMES],
    "DWT": [f"ch{c:02d}_{n}" for c in range(N_CHANNELS) for n in DWT_NAMES],
}

# Every individual statistic is selectable too, so `--groups mav,rms` picks the
# 16 or 32 column subset without a second mechanism. Worth having because the
# families are not the right granularity: mav alone (16 columns) scores 0.854
# against TD's 0.842, so the useful subset is smaller than any family.
FEATURE_GROUPS.update({
    n: [f"ch{c:02d}_{n}" for c in range(N_CHANNELS)]
    for n in TD_NAMES + FD_NAMES + DWT_NAMES
})


def mav(x): return np.mean(np.abs(x))
def rms(x): return np.sqrt(np.mean(x ** 2))
def wl(x): return np.sum(np.abs(np.diff(x)))


def zc(x, eps=ZC_EPS):
    """Zero crossings with a deadzone. `s[:-1]*s[1:] < 0` also drops the
    spurious pair np.sign() produces at exact zeros."""
    s = np.sign(x)
    return int(np.sum((s[:-1] * s[1:] < 0) & (np.abs(np.diff(x)) >= eps)))


def ssc(x, eps=ZC_EPS):
    """Slope sign changes with the same deadzone."""
    d = np.diff(x)
    big = np.maximum(np.abs(d[:-1]), np.abs(d[1:])) >= eps
    return int(np.sum((d[:-1] * d[1:] < 0) & big))


@lru_cache(maxsize=8)
def _hamming(n):
    return np.hamming(n)


def td_features(w):
    f = []
    for ch in range(w.shape[1]):
        x = w[:, ch]
        f += [
            mav(x),
            rms(x),
            wl(x),
            zc(x),
            ssc(x),
            np.sum(np.abs(x)),      # IEMG
            np.var(x),
            skew(x),
            kurtosis(x),
        ]
    return f


def fd_features(w):
    """Spectral descriptors from the one-sided spectrum.

    Uses rfft deliberately. np.fft.fft + fftfreq returns negative frequencies
    for the upper half, and because a real signal's PSD is symmetric the halves
    cancel: MNF collapses to ~0 and MDF lands on a negative frequency about
    half the time. A Hamming window is applied first, per synopsis Sec. V.C.
    """
    win = _hamming(w.shape[0])
    freqs = np.fft.rfftfreq(w.shape[0], 1 / FS)

    f = []
    for ch in range(w.shape[1]):
        psd = np.abs(np.fft.rfft(w[:, ch] * win)) ** 2
        total = np.sum(psd) + 1e-12

        mnf = np.sum(freqs * psd) / total
        mdf = freqs[np.searchsorted(np.cumsum(psd), total / 2)]
        p = psd / total
        sent = -np.sum(p * np.log2(p + 1e-12))
        dom = freqs[np.argmax(psd)]

        f += [mnf, mdf, sent, dom]
    return f


def dwt_features(w):
    """Level-3 db4 subband energy ratios.

    mode='periodization' keeps the coefficient count equal to the window length.
    The default symmetric padding grows 40 samples into 60 coefficients, so a
    third of the "energy" would come from padding. Note that level 3 still
    exceeds pywt.dwt_max_level(40, 'db4') == 2, so the coarsest subbands carry
    boundary effects; the clean fix is a window longer than 56 samples, which
    conflicts with the 200 ms specified in the synopsis.
    """
    f = []
    for ch in range(w.shape[1]):
        coeffs = pywt.wavedec(w[:, ch], DWT_WAVELET, level=DWT_LEVEL,
                              mode='periodization')
        energy = np.array([np.sum(c ** 2) for c in coeffs])
        if DWT_ABSOLUTE:
            f += list(np.log(energy + 1e-12))
        else:
            f += list(energy / (energy.sum() + 1e-12))
    return f


def extract_all(w):
    return td_features(w) + fd_features(w) + dwt_features(w)


if __name__ == "__main__":
    n = 40
    t = np.arange(n) / FS
    w = np.tile(np.sin(2 * np.pi * 60 * t)[:, None], (1, N_CHANNELS))

    feats = extract_all(w)
    assert len(feats) == len(FEATURE_NAMES) == 272, len(feats)
    assert np.all(np.isfinite(feats)), "non-finite feature"

    named = dict(zip(FEATURE_NAMES, feats))
    # the bug this replaced put MNF near 0 Hz and MDF negative
    assert 40 < named["ch00_mnf"] < 80, named["ch00_mnf"]
    assert 40 < named["ch00_mdf"] < 80, named["ch00_mdf"]
    assert abs(named["ch00_dom"] - 60) <= FS / n, named["ch00_dom"]

    ratios = [named[f"ch00_{k}"] for k in DWT_NAMES]
    if not DWT_ABSOLUTE:
        assert abs(sum(ratios) - 1.0) < 1e-6, sum(ratios)

    # deadzone: pure noise below eps must not register crossings
    quiet = np.zeros((n, N_CHANNELS)) + 1e-4
    quiet[::2] *= -1
    assert zc(quiet[:, 0]) == 0, "deadzone not suppressing sub-threshold noise"

    families = {"TD", "FD", "DWT"}
    stats = set(TD_NAMES + FD_NAMES + DWT_NAMES)
    assert set(FEATURE_GROUPS) == families | stats
    assert not (families & stats), "a statistic must not shadow a family name"

    # the three families still partition the 272 columns
    assert sum(len(FEATURE_GROUPS[f]) for f in families) == 272
    # every per-statistic group is one column per channel, and selecting all of
    # them is the same set as selecting all three families
    assert all(len(FEATURE_GROUPS[s]) == N_CHANNELS for s in stats)
    assert ({c for s in stats for c in FEATURE_GROUPS[s]}
            == {c for f in families for c in FEATURE_GROUPS[f]})
    assert len(FEATURE_GROUPS["mav"]) + len(FEATURE_GROUPS["rms"]) == 32

    print("features self-check ok:", len(feats), "features")
