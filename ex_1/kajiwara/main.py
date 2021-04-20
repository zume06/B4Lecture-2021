import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import sys
import argparse
from pathlib import Path


def stft(data, win_size=1024, overlap=0.5):
    '''
    stft transform data by short-time Fourier transform

    Parameters
    ----------
    data: ndarray
        waveform data
    win_size: int
        window size
    overlap: float
        overlap size

    Returns
    -------
    cv_spec: ndarray
        complex-valued spectrogram
    '''

    data_length = len(data)
    shift_size = int(win_size*overlap)
    ite = int((data_length - shift_size) / shift_size)
    window = np.hamming(win_size)

    cv_spec = []
    for j in range(0,  data_length, shift_size):
        x = data[j:j+win_size]
        if win_size > len(x):
            break
        x = window * x
        x = np.fft.fft(x)
        cv_spec.append(x)

    # result = []
    # for i in range(ite):
    #     x = data[i*shift_size:i*shift_size+win_size]
    #     x = window * x
    #     x = np.fft.fft(x)
    #     result.append(x)

    # shape: (ite, complex-valued spectrogram) -> (complex-valued spectrogram, ite)
    cv_spec = np.array(cv_spec).T
    return cv_spec


def instft(data, win_size=1024, overlap=0.5):
    '''
    instft transform data by inverse short-time Fourier transform

    Parameters
    ----------
    data: ndarray
        complex-valued spectrogram
    win_size: int
        window size
    overlap: float
        overlap size

    Returns
    -------
    wave_data: ndarray
        waveform data
    '''

    shift_size = int(win_size*overlap)
    ite = data.shape[0]
    window = np.hamming(win_size)

    wave_data = np.zeros(ite*shift_size+win_size)
    for i in range(ite):
        x = data[i]
        x = np.fft.ifft(x).real * win_size
        wave_data[i*shift_size:i*shift_size+win_size] = wave_data[
            i * shift_size:i*shift_size+win_size] + x

    return wave_data


def main(args):
    data_path = args.data_path
    win_size = args.win_size
    overlap = args.overlap

    wave_data, sr = librosa.load(data_path)

    cs = stft(wave_data)

    # extract magnitude (amplitude spectrum) and phase (phase spectrum)
    amplitude, phase = librosa.magphase(cs)
    # amplitude -> db
    db = librosa.amplitude_to_db(np.abs(amplitude))

    inv_data = instft(cs.T)

    fig, ax = plt.subplots(nrows=3, ncols=1)
    fig.subplots_adjust(hspace=0.5)

    librosa.display.waveplot(wave_data, sr=sr, x_axis='time', ax=ax[0])
    ax[0].set(title='Original', xlabel="Time [s]", ylabel="Magnitude")
    ax[0].label_outer()

    spec_img = librosa.display.specshow(
        db, sr=sr, x_axis='time', y_axis='log', ax=ax[1])
    fig.colorbar(spec_img, ax=ax[1])
    ax[1].set(title='Spectrum', xlabel="Time [s]", ylabel="Frequency [Hz]")
    ax[1].label_outer()

    librosa.display.waveplot(inv_data, sr=sr, x_axis='time', ax=ax[2])
    ax[2].set(title='Inversed', xlabel="Time [s]", ylabel="Magnitude")
    ax[2].label_outer()

    plt.savefig("./result/result.pdf")


if __name__ == "__main__":
    description = 'Example: python main.py sample.wav 1024 0.5'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('data_path', help='path of data')
    parser.add_argument('win_size', help='Window size')
    parser.add_argument('overlap', help='overlap size')

    args = parser.parse_args()

    main(args)
