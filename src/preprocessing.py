from scipy.signal import butter, filtfilt, iirnotch
import numpy as np

def bandpass(x, fs=200, low=20, high=95, order=4):
    nyq = fs / 2
    high = min(high, nyq - 1e-3)   # keep safely below Nyquist
    low = max(low, 1e-3)
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, x, axis=0)

def notch(x, fs=200, f0=50, q=30):
    f0 = min(f0, fs / 2 - 1e-3)
    b, a = iirnotch(f0, q, fs)
    return filtfilt(b, a, x, axis=0)

def normalize(x, rest_samples=200):
    rest_samples = min(rest_samples, len(x))
    rms = np.sqrt(np.mean(x[:rest_samples] ** 2, axis=0))
    return x / (rms + 1e-8)

def preprocess(x):
    x = bandpass(x)
    x = notch(x)
    x = normalize(x)
    return x
