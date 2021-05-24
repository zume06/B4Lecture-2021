import numpy as np
import scipy as sp

from .autocorrelation import autocorrelation


def toeplitz_solver(ac, dim):
    '''
    toeplitz_solver solve toeplitz matrix.

    Parameters
    ----------
    ac: ndarray
        autocorrelation of sequence data
    dim: int
        dimention of coef

    Returns
    -------
    a: ndarray
        predicted coefficient
    e: float
        prediction error
    '''

    a = np.zeros(dim+1)
    a[0] = 1
    a[1] = -ac[1]/ac[0]

    e = ac[0] + (ac[1]*a[1])

    for i in range(2, dim + 1):
        k = -np.sum(a[:i] * ac[i:0:-1]) / e
        a[:i+1] += k * a[:i+1][::-1]
        e *= 1 - k * k

    return a, e


def lpc_method(wave_data, dim=100, clip_size=None):
    '''
    lpc_method extract spectrum envelope by LPC method

    Parameters
    ----------
    wave_data: ndarray (1d)
        audio data
    dim: int
        dimention of coef
    clip_size: int
        length of clipping data
        if None, length of wave_data

    Returns
    -------
    lpc_env: ndarray (1d)
        spectrum envelope
    '''

    if clip_size is None:
        clip_size = len(wave_data)

    ac = autocorrelation(wave_data)
    a, e = toeplitz_solver(ac, dim)
    _, h = sp.signal.freqz(np.sqrt(e), a, clip_size, "whole")
    lpc_env = 20 * np.log10(np.abs(h))

    return lpc_env
