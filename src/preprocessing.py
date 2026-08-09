import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

FS = 200            # NinaPro DB5 sampling rate
STAGES = ("bandpass", "notch", "normalize")


def bandpass(x, fs=FS, low=20, high=95, order=4):
    """Zero-phase Butterworth bandpass.

    The synopsis specifies 20-450 Hz, which is unreachable here: DB5 samples at
    200 Hz, so Nyquist is 100 Hz. 95 Hz is the practical ceiling. The clamp
    below keeps the filter valid if fs is ever changed.
    """
    nyq = fs / 2
    high = min(high, nyq - 1e-3)
    low = max(low, 1e-3)
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, x, axis=0)


def notch(x, fs=FS, f0=50, q=30):
    """Power-line notch. DB5 was recorded in Europe, so 50 Hz."""
    f0 = min(f0, fs / 2 - 1e-3)
    b, a = iirnotch(f0, q, fs)
    return filtfilt(b, a, x, axis=0)


def normalize(x, rest_mask=None):
    """Scale each channel by the RMS of that subject's rest baseline.

    Deviation from synopsis Sec. V.B, which specifies the first 500 ms of each
    trial: this uses every rest-labelled sample in the recording instead. Same
    intent, far more stable estimate. Pass a narrower mask to match the text.
    """
    if rest_mask is None or int(np.sum(rest_mask)) == 0:
        ref = x         # ponytail: no rest labelled, fall back to the recording
    else:
        ref = x[rest_mask]

    rms = np.sqrt(np.mean(ref ** 2, axis=0))
    return x / (rms + 1e-8)


def preprocess(x, rest_mask=None, stages=STAGES):
    """Apply the preprocessing chain. `stages` selects a subset, for ablation.

    Must be given the continuous recording. Masking gestures out first splices
    together non-adjacent segments and the filters then ring at every join.
    """
    if "bandpass" in stages:
        x = bandpass(x)
    if "notch" in stages:
        x = notch(x)
    if "normalize" in stages:
        x = normalize(x, rest_mask)
    return x


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    t = np.arange(2000) / FS
    # 60 Hz burst in band, 50 Hz interference, DC drift out of band
    sig = (np.sin(2 * np.pi * 60 * t) + 2 * np.sin(2 * np.pi * 50 * t) + 5.0)
    x = np.column_stack([sig, sig * 0.5]) + rng.standard_normal((2000, 2)) * 0.01
    rest = np.zeros(2000, dtype=bool)
    rest[:400] = True

    def power_at(sig1d, f):
        spec = np.abs(np.fft.rfft(sig1d))
        return spec[np.argmin(np.abs(np.fft.rfftfreq(len(sig1d), 1 / FS) - f))]

    y = preprocess(x, rest_mask=rest)
    assert y.shape == x.shape
    assert power_at(y[:, 0], 0) < power_at(x[:, 0], 0) * 0.01, "DC not removed"
    assert power_at(y[:, 0], 50) < power_at(x[:, 0], 50) * 0.5, "50 Hz not notched"
    assert np.all(np.isfinite(y))

    # stage selection actually changes the output
    assert not np.allclose(preprocess(x, rest, ("bandpass",)), y)
    assert np.allclose(preprocess(x, rest, ()), x), "empty stage list must be a no-op"

    print("preprocessing self-check ok")
