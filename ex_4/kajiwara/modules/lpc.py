import numpy as np


def levinson_durbin_method(wave_data, dim):
    ac = np.correlate(wave_data, wave_data, mode='full')
    ac = ac[len(wave_data)-1:len(wave_data)+dim]

    a = np.zeros(dim+1)
    a[0] = 1
    a[1] = -ac[1]/ac[0]

    e = ac[0] + (ac[1]*a[1])

    for i in range(2, dim + 1):
        k = -np.sum(a[:i] * ac[i:0:-1]) / e
        a[:i+1] += k * a[:i+1][::-1]
        e *= 1 - k * k

    return a, e
