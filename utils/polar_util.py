import numpy as np

def dop(s0, s1, s2):
    return np.sqrt(s1**2 + s2**2) / (s0 + 1e-6)

def aop(s1, s2):
    return 0.5 * np.arctan2(s2, s1)
