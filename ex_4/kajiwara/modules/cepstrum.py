import numpy as np

from .utils import get_framing_data


def get_cepstrum(input, is_clipping, is_framing,  clip_size=2048, win_size=1024, overlap=0.5):
    assert not (is_clipping and is_framing), 'The two values (is_clipping, is_framing) ​​must be different'

    if is_clipping:
        spec = np.fft.rfft(input, clip_size)

    if is_framing:
        framing_data = get_framing_data(input, win_size, overlap)
        spec = np.fft.rfft(framing_data)

    spec_db = 20 * np.log10(np.abs(spec))
    ceps_db = np.fft.irfft(spec_db)

    return ceps_db.real


def get_ceps_peaks(ceps, sr, max=200, min=50):
    max_cep_order = int(np.floor(sr / min))
    min_cep_order = int(np.floor(sr / max))

    peak_index = np.argmax(ceps[:, min_cep_order:max_cep_order], axis=1)
    peak_index = peak_index + min_cep_order

    return peak_index


def get_envelope(ceps, coef=20):
    ceps_liftered = ceps.copy()
    ceps_liftered[coef:-coef] = 0

    envelope = np.fft.rfft(ceps_liftered)

    return envelope
