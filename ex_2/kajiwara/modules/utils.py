import numpy as np


def conv1d(x, y):
    '''
    conv1d calculate convolution of x and y (1d array)
    x * y

    Parameters
    ----------
    x: 1d array
    y: 1d array

    Returns
    -------
    conv: ndarray
    '''

    length = len(x) + len(y) - 1

    x_spec = np.fft.rfft(x, n=length)
    y_spec = np.fft.rfft(y, n=length)
    conv_spec = x_spec * y_spec
    conv = np.fft.irfft(conv_spec)

    return conv
