import numpy as np

def eud_dis(x, y):
    return np.linalg.norm(x, y)

def cos_dis(x, y):
    return x.reshape((1, -1)) @ y.reshape((-1, 1)) / (np.linalg.norm(x) * np.linalg.norm(y))