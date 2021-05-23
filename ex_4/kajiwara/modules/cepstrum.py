import numpy as np


def get_cepstrum(input):
    spec = np.fft.rfft(input)
    spec_db = np.log(np.abs(spec))
    ceps_db = np.fft.irfft(spec_db).real

    return ceps_db
