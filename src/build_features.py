from pathlib import Path
import numpy as np
import pandas as pd

from data_loader import load_subject, filter_gestures
from preprocessing import preprocess
from features import extract_all

RAW_DIR = Path("data/raw")
WINDOW_SIZE = 40
STEP = 20
MIN_PURITY = 0.90   # keep only mostly-pure windows

rows = []

mat_files = sorted(RAW_DIR.rglob("*E2*.mat"))   # ONLY exercise 2

for mat_path in mat_files:
    print(f"Processing {mat_path}")

    emg, labels = load_subject(mat_path)
    emg, labels = filter_gestures(emg, labels)

    labels = labels.astype(int)
    emg = preprocess(emg)

    for start in range(0, len(emg) - WINDOW_SIZE + 1, STEP):
        window = emg[start:start + WINDOW_SIZE]
        window_labels = labels[start:start + WINDOW_SIZE]

        # dominant label in this window
        label = np.bincount(window_labels).argmax()
        purity = np.mean(window_labels == label)

        # skip transition windows
        if purity < MIN_PURITY:
            continue

        feats = extract_all(window)
        rows.append(feats + [label])

df = pd.DataFrame(rows)
df.to_csv("features.csv", index=False)
print("Saved features.csv with shape:", df.shape)
