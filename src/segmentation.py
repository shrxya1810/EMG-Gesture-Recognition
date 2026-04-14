import numpy as np

def sliding_window(emg, labels):
    fs=200
    W=40
    step=20

    windows=[]
    y=[]

    for i in range(0,len(emg)-W,step):

        seg=emg[i:i+W]
        label=labels[i+W//2]

        if label>0:
            windows.append(seg)
            y.append(label)

    return np.array(windows),np.array(y)
