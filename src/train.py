import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

print("Loading features...")
df = pd.read_csv("features.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print("Cleaning data...")
X[np.isinf(X)] = np.nan
X = np.nan_to_num(X)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# -------------------------
# SVM
# -------------------------
print("\n===== Training SVM =====")

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

svm = SVC(kernel="rbf", C=10, gamma="scale")

scores = cross_val_score(svm, Xs, y, cv=cv, n_jobs=-1)
print("SVM Accuracy:", scores.mean())

svm.fit(Xs, y)

joblib.dump(svm, "models/svm.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# -------------------------
# Random Forest
# -------------------------
print("\n===== Training RF =====")

rf = RandomForestClassifier(
    n_estimators=600,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)

scores = cross_val_score(rf, X, y, cv=cv, n_jobs=-1)
print("RF Accuracy:", scores.mean())

rf.fit(X, y)
joblib.dump(rf, "models/rf.pkl")

# -------------------------
# ExtraTrees (usually best)
# -------------------------
print("\n===== Training ExtraTrees =====")

et = ExtraTreesClassifier(
    n_estimators=600,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)

scores = cross_val_score(et, X, y, cv=cv, n_jobs=-1)
print("ExtraTrees Accuracy:", scores.mean())

et.fit(X, y)
joblib.dump(et, "models/extratrees.pkl")

print("\nDone.")
