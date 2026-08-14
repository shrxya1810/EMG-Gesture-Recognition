"""Rest / no-gesture gate for real-time use (PROGRESS.md 7.2 item 2).

    rest_mav = calibrate(emg, labels == 0)
    if is_gesture(window, rest_mav):
        pred = model.predict(extract_all(window))

The models are trained on six gestures only -- `build_features.py` drops every
window whose majority label is not in GESTURES -- so a resting arm gets
classified as one of the six with no way to say "nothing". This gates that,
without a seventh class.

**Why a gate and not a seventh class.** Rest is not rare on DB5: 73,431 rest
windows against 8,288 gesture windows, 8.9x the whole gesture set and ~50x any
single class. Training on it means subsampling, and it renumbers the problem --
7-class chance is 14.3% against 16.7%, and rest is the easiest class, so
accuracy rises for reasons unrelated to the model. Every figure in PROGRESS.md
would need restating. The gate leaves all of them intact.

**Why amplitude.** The feature ablation (PROGRESS.md 5.3) established that the
discriminative signal here is almost entirely amplitude -- mav, rms, iemg, wl,
var -- and rest is *defined* by its absence. The same measurement that makes the
frequency and wavelet features useless makes this gate work.

**Why a multiple of the rest baseline, not an absolute threshold.** Resting MAV
varies 1.59 to 3.21 across the ten DB5 subjects and gesture MAV varies 7.2 to
20.6, so one absolute number fits nobody well. Measured on all 10 subjects, all
81,719 windows:

    threshold                    rest rejected   gestures kept   balanced
    absolute, 5.0                    0.906           0.964         0.935
    2.5 x subject rest baseline      0.928           0.972       **0.950**

Taking the threshold as a multiple also makes it invariant to preprocessing and
to electrode gain: calibrate on the same signal you gate, and the ratio holds.
That is the knob to turn for a different armband, not K itself.
"""
import numpy as np

# Threshold in multiples of resting MAV. 2.5 is the balanced optimum measured
# above; the curve is flat between 2.25 and 2.75, so trade within that range.
# Lower keeps more gestures and lets more rest through (dashboard flickers);
# higher rejects more rest and starts dropping weak gestures.
K = 2.5

# Fallback when no rest baseline is available -- the median resting MAV across
# the ten DB5 subjects. A real session should always calibrate instead: this is
# a different subject's arm and different electrode placement.
DEFAULT_REST_MAV = 2.06


def amplitude(window):
    """Mean absolute value over every channel and sample of one window.

    Identical to the mean of the 16 chNN_mav features, so a caller holding a
    TD feature vector can gate on that instead. Computing it from the raw
    window means rest windows can be gated *before* feature extraction, which
    is where 12.3 of the 12.5 ms of LDA's per-window latency goes (5.12).
    """
    return float(np.mean(np.abs(window)))


def calibrate(emg, rest_mask):
    """Resting MAV for one recording or session.

    `rest_mask` must come from the untrimmed labels. `trim_transients()`
    relabels the first and last 15% of every repetition as rest, which is
    11,153 windows of gesture onset and offset -- one in seven of the trimmed
    rest set. Calibrating on those inflates the baseline and the gate then
    swallows weak gestures.
    """
    rest_mask = np.asarray(rest_mask, dtype=bool)
    if not rest_mask.any():
        return DEFAULT_REST_MAV
    return float(np.mean(np.abs(emg[rest_mask])))


def is_gesture(window, rest_mav=DEFAULT_REST_MAV, k=K):
    """True if this window is active enough to be worth classifying."""
    return amplitude(window) >= k * rest_mav


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, ch = 40, 16

    rest = rng.standard_normal((4000, ch)) * 2.0
    quiet = rng.standard_normal((n, ch)) * 2.0
    active = rng.standard_normal((n, ch)) * 12.0

    base = calibrate(rest, np.ones(len(rest), bool))
    assert 1.5 < base < 1.7, base            # E|N(0,2)| = 2*sqrt(2/pi) ~ 1.60

    assert not is_gesture(quiet, base)
    assert is_gesture(active, base)

    # Scale invariance is the whole design: a 10x gain change on both the
    # baseline and the signal must not move the decision.
    assert is_gesture(active * 10, calibrate(rest * 10, np.ones(len(rest), bool)))
    assert not is_gesture(quiet * 10, calibrate(rest * 10, np.ones(len(rest), bool)))

    # An empty baseline falls back rather than dividing by nothing.
    assert calibrate(rest, np.zeros(len(rest), bool)) == DEFAULT_REST_MAV

    # k orders monotonically: raising it can only reject more.
    amp = amplitude(active)
    ks = [k for k in (1.0, 2.0, 4.0, 8.0) if amp >= k * base]
    assert ks == sorted(ks) and len(ks) < 4

    # The feature-vector shortcut must agree with the raw-window computation.
    assert np.isclose(amplitude(active), np.mean([np.mean(np.abs(active[:, c]))
                                                  for c in range(ch)]))

    print("rest_gate self-check ok")
