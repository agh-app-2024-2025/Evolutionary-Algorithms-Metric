import numpy as np


def radar_function(x):
    x = np.array(x)
    d = x.shape[0]

    m = 2 * d - 1
    M = 2 * m
    hsum = np.zeros(M)

    for kk in range(1, m + 1):
        if kk % 2 != 0:
            i = (kk + 1) // 2
            hsum[kk - 1] = 0
            for j in range(i, d + 1):
                summ = np.sum(x[abs(2 * i - j - 1):j])
                hsum[kk - 1] += np.cos(summ)
        else:
            i = kk // 2
            hsum[kk - 1] = 0
            for j in range(i + 1, d + 1):
                summ = np.sum(x[abs(2 * i - j):j])
                hsum[kk - 1] += np.cos(summ)
            hsum[kk - 1] += 0.5

    hsum[m:M] = -hsum[:m]
    return np.max(hsum)