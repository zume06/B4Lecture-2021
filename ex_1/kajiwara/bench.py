import time

import scipy as sp
import numpy as np
import librosa

from main import stft, instft


def calc_process_time(args, func):
    start = time.perf_counter()
    func(**args)
    end = time.perf_counter()

    return end - start


def my_spectrogram(data):
    # stft
    cs = stft(data, win_size=1024, overlap=0.5)

    # extract magnitude (amplitude spectrum) and phase (phase spectrum)
    # amplitude, phase = librosa.magphase(cs)
    # amplitude -> db
    db = librosa.amplitude_to_db(np.abs(cs))

    return db


def main():
    fake_data = 2 * np.random.rand(22050*3) - 1

    # scipy
    sp_res = calc_process_time(
        {'x': fake_data, 'mode': 'complex'}, sp.signal.spectrogram)
    print("scipy.signal.spectrogram result: {}".format(sp_res))

    # mine
    my_res = calc_process_time({'data': fake_data}, my_spectrogram)
    print("my spectrogram result: {}".format(my_res))


main()
