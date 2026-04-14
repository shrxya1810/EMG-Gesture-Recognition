# EMG Gesture Recognition (sEMG + Machine Learning)

A complete end-to-end **surface EMG gesture recognition pipeline** built using the **NinaPro DB5 dataset**, featuring preprocessing, segmentation, feature extraction, and high-accuracy classical ML models.

This project trains gesture classifiers using **multi-domain EMG features** and achieves **~84% accuracy** on 6 gesture classes.

---

# Features

* EMG signal preprocessing (filtering + normalization)
* Sliding window segmentation
* Time-domain EMG features
* Frequency-domain EMG features
* Wavelet (DWT) features
* Feature scaling & cleaning
* Model training & evaluation
* Automatic accuracy comparison
* Model export for real-time inference

---

# Dataset

This project uses **NinaPro DB5** surface EMG dataset.

Dataset is **NOT included** in the repo due to size.

Place dataset in:

```
data/raw/
```

Example:

```
data/raw/s1/
data/raw/s2/
data/raw/s3/
...
data/raw/s10/
```

Each folder should contain `.mat` files from NinaPro.

---

# Project Structure

```
EMG-Gesture-Recognition/
│
├── src/
│   ├── build_features.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── features.py
│   └── train.py
│
├── data/
│   └── raw/           # NinaPro dataset (not committed)
│
├── models/            # saved trained models
├── results/           # metrics / confusion matrix
│
├── requirements.txt
└── README.md
```

---

# Installation

Clone repo

```
git clone https://github.com/shrxya1810/EMG-Gesture-Recognition.git
cd EMG-Gesture-Recognition
```

Create virtual environment

```
python3 -m venv venv
source venv/bin/activate
```

Install dependencies

```
pip install -r requirements.txt
```

---

# Step 1 — Build Features

This extracts EMG features and creates `features.csv`

```
python3 src/build_features.py
```

Output:

```
features.csv
Saved features.csv with shape: (12756, 273)
```

---

# Step 2 — Train Models

```
python3 src/train.py
```

Example Output

```
Loading features...
Cleaning data...
Scaling...

===== Training SVM =====
SVM Accuracy: 0.72

===== Training RF =====
RF Accuracy: 0.82

===== Training ExtraTrees =====
ExtraTrees Accuracy: 0.84

Done.
```

---

# Current Results

| Model         | Accuracy |
| ------------- | -------- |
| Linear SVM    | ~72%     |
| Random Forest | ~82%     |
| Extra Trees   | ~84%     |

Best Model: **ExtraTrees**

---

# Feature Extraction

Each window extracts:

### Time Domain

* Mean Absolute Value (MAV)
* RMS
* Waveform Length
* Zero Crossing
* Slope Sign Changes
* Variance
* Skewness
* Kurtosis

### Frequency Domain

* Mean Frequency
* Median Frequency
* Spectral Entropy
* Dominant Frequency

### Wavelet Features

* DWT Level 3
* Subband Energy
* Energy Ratios

Total Features ≈ **272 per window**

---

# Windowing

* Window size: 200 ms
* Overlap: 50%
* Channels: 16
* Dataset: NinaPro DB5 Exercise 2

---

# Models Used

* Linear SVM
* Random Forest
* Extra Trees (Best)

All models automatically:

* clean NaN
* normalize
* train
* evaluate
* print accuracy

---

# Output

After training:

Models saved to:

```
models/
```

Results saved to:

```
results/
```

---

# How to Reproduce

Run in order:

```
python3 src/build_features.py
python3 src/train.py
```

---

# Future Work

* CNN on raw EMG
* Real-time inference
* Streamlit dashboard
* Online prediction
* Subject-independent training
* Deep learning models
* Hardware integration

---

# Tech Stack

Python
NumPy
SciPy
Scikit-learn
PyWavelets
Pandas
Matplotlib

---

# Author

Shreya Agarwal
EMG Gesture Recognition using Machine Learning


