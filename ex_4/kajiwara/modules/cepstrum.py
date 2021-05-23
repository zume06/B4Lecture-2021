import numpy as np

from .utils import get_framing_data


def get_cepstrum(input):
    framing_data = get_framing_data(input, 1024, 0.5)
    spec = np.fft.rfft(framing_data)
    spec_db = np.log(np.abs(spec))
    ceps_db = np.fft.irfft(spec_db)

    return ceps_db


def get_ceps_peaks(ceps, sr, max=200, min=50):
    # ケプストラムの最大次数、最小次数
    max_cep_order = int(np.floor(sr / min))
    min_cep_order = int(np.floor(sr / max))

    # ピーク位置の検出
    peak_index = np.argmax(ceps[:, min_cep_order:max_cep_order], axis=1)
    peak_index = peak_index + min_cep_order

    return peak_index


def get_envelope(ceps, coef=20):
    ceps_liftered = ceps.copy()
    ceps_liftered[coef:len(ceps)-coef] = 0

    envelope = np.fft.irfft(ceps_liftered, axis=0)

    return envelope
