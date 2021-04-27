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


def librosa_spectrogram(data):
    cv = librosa.stft(data)
    # s, phase = librosa.magphase(cv)
    db = librosa.amplitude_to_db(np.abs(cv))

    return db


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

    res = {
        'scipy': [],
        'librosa': [],
        'mine': [],
    }

    for i in range(10):
        # scipy
        sp_res = calc_process_time(
            {'x': fake_data, 'mode': 'complex'}, sp.signal.spectrogram)
        res['scipy'].append(sp_res)
        print("scipy.signal.spectrogram result: {}".format(sp_res))

        # librosa
        lib_res = calc_process_time({'data': fake_data}, librosa_spectrogram)
        res['librosa'].append(lib_res)
        print("using librosa dtft result: {}".format(lib_res))

        # mine
        my_res = calc_process_time({'data': fake_data}, my_spectrogram)
        res['mine'].append(my_res)
        # print("my spectrogram result: {}".format(my_res))

    for k, v in res.items():
        print(k, sum(v)/len(v))


main()
