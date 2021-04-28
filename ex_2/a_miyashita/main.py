import numpy as np
import librosa
import librosa.display
from matplotlib import pyplot as plt

import ex1

def convolution(input, filter):
    result = np.zeros(input.size + filter.size)
    for i in range(input.size):
        result[i : i + filter.size] = result[i : i + filter.size] + input[i] * filter
    return result

class BSF:
    def __init__(self, fmin, fmax, win_size, sr):
        wmin = 2.0 * fmin / sr
        wmax = 2.0 * fmax / sr
        ite = np.arange(-win_size // 2,(win_size + 1) // 2)
        self.win_size = win_size
        self.filter = wmin * np.sinc(wmin * ite) - wmax * np.sinc(wmax * ite)
        self.filter[win_size // 2] = self.filter[win_size // 2] + 1.0
        self.filter = self.filter * np.hamming(win_size)
        self.status = np.zeros(win_size)

    def __call__(self, inputs=None):
        if inputs is None:
            status = self.status
            self.status = np.zeros(self.win_size)
            return status
        else:
            inputs = np.array(inputs)
            conv = convolution(inputs, self.filter)
            conv[:self.win_size] = conv[:self.win_size] + self.status
            self.status = conv[inputs.size:]
            return conv[:inputs.size]

if __name__ == "__main__":
    bsf = BSF(2000, 4000, 4000, 16000)

    # load
    origin_signal, rate = librosa.load('sample.wav', sr=16000)

    filtered_signal = bsf(origin_signal)
    filtered_signal = np.concatenate([filtered_signal, bsf()])

    n_fft = 512

    # stft
    spectrogram = ex1.stft(filtered_signal, n_fft)

    spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram))

    # istft
    resynthesized_signal = ex1.istft(spectrogram)

    # plot
    fig, ax = plt.subplots(3, 1)
    fig.subplots_adjust(hspace=1.0)
    
    librosa.display.waveplot(filtered_signal, sr=rate, x_axis='time', ax=ax[0])
    ax[0].set(title='Original signal', xlabel='Time[sec]', ylabel='Magnitude')
    
    img = librosa.display.specshow(spectrogram_db, sr=rate, hop_length=n_fft//2, x_axis='time', y_axis='linear', ax=ax[1])
    ax[1].set(title='Spectrogram', xlabel='Time[sec]', ylabel='Frequency[Hz]')
    fig.colorbar(img, ax=ax[1])

    librosa.display.waveplot(resynthesized_signal, sr=rate, x_axis='time', ax=ax[2])
    ax[2].set(title='Re-synthesized signal', xlabel='Time[sec]', ylabel='Magnitude')

    ax_pos_0 = ax[0].get_position()
    ax_pos_1 = ax[1].get_position()
    ax_pos_2 = ax[2].get_position()
    ax[0].set_position([ax_pos_0.x0, ax_pos_0.y0, ax_pos_1.width, ax_pos_1.height])
    ax[2].set_position([ax_pos_2.x0, ax_pos_2.y0, ax_pos_1.width, ax_pos_1.height])
    