import re
from pathlib import Path

import numpy as np
import scipy.io

# NinaPro DB5 Exercise B (files *E2*.mat) numbers hand postures 1-8 and wrist
# movements 9-17. These six are the gestures named in the synopsis, Sec. V.A.
# Do not renumber without re-checking against the Atzori movement table --
# labels 1-6 are all hand postures and do not match the synopsis.
GESTURES = {
    5:  "open hand (finger abduction)",
    6:  "closed fist",
    9:  "forearm supination",
    10: "forearm pronation",
    13: "wrist flexion",
    14: "wrist extension",
}


def load_subject(path):
    """Return (emg, restimulus, rerepetition) for one NinaPro recording.

    The signal is returned whole, rest samples included: preprocessing has to
    see the continuous recording, and the rest samples are the normalisation
    baseline.
    """
    # ponytail: DB5_Preproc ships MATLAB v5 throughout; add mat73 only if a
    # future NinaPro database (DB2/DB7) brings in v7.3 files.
    data = scipy.io.loadmat(path)

    emg = np.asarray(data['emg'], dtype=np.float64)
    labels = np.asarray(data['restimulus']).flatten().astype(int)
    reps = np.asarray(data['rerepetition']).flatten().astype(int)

    return emg, labels, reps


def subject_id(path):
    """Pull the subject number out of a filename like S7_E2_A1.mat."""
    m = re.search(r"[Ss](\d+)_E", Path(path).name)
    if m is None:
        raise ValueError(f"cannot read subject id from {path}")
    return int(m.group(1))


if __name__ == "__main__":
    path = sorted(Path("data/raw").rglob("*E2*.mat"))[0]
    emg, labels, reps = load_subject(path)

    assert emg.ndim == 2 and emg.shape[1] == 16, emg.shape
    assert len(emg) == len(labels) == len(reps)
    assert set(GESTURES) <= set(np.unique(labels).tolist()), "missing gestures"
    assert subject_id(path) >= 1

    print(f"{path.name}: {emg.shape[0]} samples, subject {subject_id(path)}")
    for g, name in GESTURES.items():
        n = int((labels == g).sum())
        nrep = len(set(reps[labels == g].tolist()))
        print(f"  {g:2d} {name:28s} {n:6d} samples, {nrep} reps")
