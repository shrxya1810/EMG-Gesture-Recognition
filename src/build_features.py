import glob
import pandas as pd
from data_loader import *
from preprocessing import *
from segmentation import *
from features import *

rows=[]

files=glob.glob("data/raw/**/*.mat",recursive=True)

for f in files:

    emg,labels=load_subject(f)
    emg,labels=filter_gestures(emg,labels)

    emg=preprocess(emg)

    windows,y=sliding_window(emg,labels)

    for w,l in zip(windows,y):

        feats=extract_all(w)
        rows.append(feats+[l])

df=pd.DataFrame(rows)

df.to_csv("features.csv",index=False)
