import numpy as np

def zero_to_nan(x):
    if isinstance(x, (int, float)) and x == 0:
        return np.nan
    return x
