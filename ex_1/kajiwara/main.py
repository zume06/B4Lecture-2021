import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

TIME_TEMPLATE = '%Y%m%d%H%M%S'


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
    window = np.hamming(win_size)

    cv_spec = []
    for j in range(0,  data_length, shift_size):
        x = data[j:j+win_size]
        if win_size > len(x):
            break
        x = window * x
        x = np.fft.fft(x)
        cv_spec.append(x)

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
    data_path = Path(args.data_path)
    result_path = Path(args.save_path)

    assert data_path.exists(), '{} is not exist'.format(data_path)
    assert result_path.exists(), '{} is not exist'.format(result_path)

    # load data
    wave_data, sr = librosa.load(data_path)

    # stft
    cs = stft(wave_data, win_size=1024, overlap=0.5)

    # extract magnitude (amplitude spectrum) and phase (phase spectrum)
    # amplitude, phase = librosa.magphase(cs)
    # amplitude -> db
    db = librosa.amplitude_to_db(np.abs(cs))

    # inverse conversion
    inv_data = instft(cs.T)/1000

    # plotting
    fig, ax = plt.subplots(nrows=3, ncols=1)
    fig.subplots_adjust(hspace=01.0)

    librosa.display.waveplot(wave_data, sr=sr, x_axis='time', ax=ax[0])
    ax[0].set(title='Original', xlabel="Time [s]", ylabel="Magnitude")
    ax[0].label_outer()

    spec_img = librosa.display.specshow(
        db, sr=sr, x_axis='time', y_axis='log', ax=ax[1])
    fig.colorbar(spec_img, ax=ax[1])
    ax[1].set(title='Spectrum', xlabel="Time [s]", ylabel="Frequency [Hz]")
    ax[1].yaxis.set_ticks([0, 128, 512, 2048, 8192])
    ax[1].label_outer()

    librosa.display.waveplot(inv_data, sr=sr, x_axis='time', ax=ax[2])
    ax[2].set(title='Inversed', xlabel="Time [s]", ylabel="Magnitude")
    ax[2].label_outer()

    # graph positioning
    ax_pos_0 = ax[0].get_position()
    ax_pos_1 = ax[1].get_position()
    ax_pos_2 = ax[2].get_position()
    ax[0].set_position(
        [ax_pos_0.x0, ax_pos_0.y0, ax_pos_1.width, ax_pos_1.height])
    ax[2].set_position(
        [ax_pos_2.x0, ax_pos_2.y0, ax_pos_1.width, ax_pos_1.height])

    # save result
    timestamp = datetime.now().strftime(TIME_TEMPLATE)
    plt.savefig(result_path.joinpath(timestamp+'-result.pdf'))
    plt.savefig(result_path.joinpath(timestamp+'-result.png'))


if __name__ == "__main__":
    description = 'Example: python main.py ./sample.wav'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('data_path', help='path of data')
    parser.add_argument('-s', '--save_path', default='./',
                        help='path to save the result')

    args = parser.parse_args()

    main(args)
