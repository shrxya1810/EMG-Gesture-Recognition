import mat73
import numpy as np

def load_subject(path):
    data = mat73.loadmat(path)

    emg = np.array(data['emg'])
    labels = np.array(data['restimulus']).flatten()

    return emg, labels


def filter_gestures(emg, labels):
    gestures = [1,2,3,4,5,6]
    mask = np.isin(labels, gestures)

    return emg[mask], labels[mask]
