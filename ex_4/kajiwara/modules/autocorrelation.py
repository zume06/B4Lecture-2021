import numpy as np
# import scipy

from .utils import get_framing_data


def autocorrelation(input, win_size, sr):
    framing_data = get_framing_data(input, win_size)
    ac = []
    for frame in framing_data:
        fft_data = np.fft.fft(frame)
        power = np.abs(fft_data)**2
        _ac = np.fft.ifft(power, axis=0).real
        # _ac /= np.max(_ac)
        ac.append(_ac)

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

        # ac_frame = ac_frame[int(len(ac_frame)/2):]
        # relmax_index = scipy.signal.argrelmax(ac_frame)[0]
        # peak_index = np.argmax(ac_frame[relmax_index])
        # peaks.append(relmax_index[peak_index])

    return np.array(peaks)
