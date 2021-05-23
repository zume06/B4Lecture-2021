import numpy as np

from .utils import get_framing_data
# from .stft import faster_stft, instft


def autocorrelation(input, win_size, sr):
    framing_data = get_framing_data(input, win_size)
    ac = []
    for frame in framing_data:
        fft_data = np.fft.fft(frame)
        power = np.abs(fft_data)**2
        ac.append(np.fft.ifft(power, axis=0).real)

    return np.array(ac, dtype='float')


def get_ac_peaks(ac):
    peaks = []
    for ac_frame in ac:
        peak_val_list = []
        peak_idx_list = []
        for i in range(2, ac_frame.size):
            if ac_frame[i-1] - ac_frame[i-2] >= 0 and ac_frame[i] - ac_frame[i-1] < 0:
                peak_val_list.append(ac_frame[i-1])
                peak_idx_list.append(i-1)
        max_idx = peak_val_list.index(max(peak_val_list))
        peaks.append(peak_idx_list[max_idx])

    return np.array(peaks)
