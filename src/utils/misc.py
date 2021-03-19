import numpy as np


def std_error(v: np.ndarray):
    sample_mean = v.mean()
    temp = 0
    for i in np.nditer(v):
        temp += (i - sample_mean) * (i - sample_mean)
    return np.sqrt(temp / (v.size - 1)) / np.sqrt(v.size)
