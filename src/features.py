import numpy as np
import pywt
from scipy.fft import fft
from scipy.stats import skew,kurtosis

def mav(x): return np.mean(np.abs(x))
def rms(x): return np.sqrt(np.mean(x**2))
def wl(x): return np.sum(np.abs(np.diff(x)))
def zc(x): return np.sum(np.diff(np.sign(x))!=0)
def ssc(x): return np.sum(np.diff(np.sign(np.diff(x)))!=0)

def td_features(w):
    f=[]
    for ch in range(w.shape[1]):
        x=w[:,ch]

        f+= [
            mav(x),
            rms(x),
            wl(x),
            zc(x),
            ssc(x),
            np.sum(np.abs(x)),
            np.var(x),
            skew(x),
            kurtosis(x)
        ]
    return f


def fd_features(w):
    f=[]
    for ch in range(w.shape[1]):

        x=w[:,ch]
        X=np.abs(fft(x))
        psd=X**2

        freqs=np.fft.fftfreq(len(x),1/200)

        mnf=np.sum(freqs*psd)/np.sum(psd)
        mdf=freqs[np.where(np.cumsum(psd)>=np.sum(psd)/2)[0][0]]

        p=psd/np.sum(psd)
        se=-np.sum(p*np.log2(p+1e-8))

        f+=[mnf,mdf,se,np.argmax(psd)]

    return f


def dwt_features(w):

    f=[]
    for ch in range(w.shape[1]):

        coeffs=pywt.wavedec(w[:,ch],'db4',level=3)

        energy=[np.sum(c**2) for c in coeffs]
        total=sum(energy)

        f+= [e/total for e in energy]

    return f


def extract_all(w):

    return td_features(w)+fd_features(w)+dwt_features(w)
