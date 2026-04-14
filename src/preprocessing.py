from scipy.signal import butter, filtfilt, iirnotch
import numpy as np

def bandpass(x, fs=200):
    low, high = 20, 450
    nyq = fs/2
    b,a = butter(4,[low/nyq,high/nyq],'band')
    return filtfilt(b,a,x,axis=0)

def notch(x,fs=200):
    b,a = iirnotch(50,30,fs)
    return filtfilt(b,a,x,axis=0)

def normalize(x):
    rms = np.sqrt(np.mean(x[:200]**2,axis=0))
    return x/(rms+1e-8)

def preprocess(x):
    x=bandpass(x)
    x=notch(x)
    x=normalize(x)
    return xx
