# include flake8, black

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
from scipy import signal


def stft(y, hop=0.5, win_length=1024):
    hop_length = int(win_length * hop)
    ynum = y.shape[0]
    window = np.hamming(win_length)

    F = []
    for i in range(int((ynum - hop_length) / hop_length)):
        tmp = y[i * hop_length : i * hop_length + win_length]
        tmp = tmp * window
        tmp = np.fft.fft(tmp)
        F.append(tmp)

    F = np.array(F).T
    return F


def istft(F, hop=0.5, win_length=1024):
    F = F.T
    hop_length = int(win_length * hop)
    Fnum = F.shape[0]
    window = np.hamming(win_length)

    for i in range(Fnum):
        tmp = F[i]
        tmp = np.fft.ifft(tmp).real
        tmp = tmp

    return y


def main():
    # load audio file
    # get current working directory
    dir = os.path.dirname(__file__) + "/"
    # dir = "C:/Users/yamam/Desktop/lab/2021/B4Lecture-2021/ex_1/t_yamamoto/"
    audio_path = dir + "recording_b4lec_ex1.wav"
    wav, sr = librosa.load(audio_path)

    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    plt.subplots_adjust(hspace=0.6)
    # params = {"axes.labelsize": 16, "axes.titlesize": 20}
    # plt.rcParams.update(params)
    # plt.rcParams["font.family"] = "Times New Roman"
    # plt.rcParams["font.size"] = 12

    # draw original wave plot
    librosa.display.waveplot(wav, sr=sr, ax=ax[0])
    ax[0].set(title="Original signal", xlabel=None, ylabel="Magnitude")
    ax[0].label_outer()

    amp = stft(wav)
    db = librosa.amplitude_to_db(np.abs(amp), ref=np.max)
    # db = librosa.amplitude_to_db(np.abs(librosa.stft(wav)), ref=np.max)
    img = librosa.display.specshow(db, sr=sr, y_axis="log", ax=ax[1])
    ax[1].set(title="Spectrogram", xlabel=None, ylabel="Frequency [kHz]")
    fig.colorbar(img, ax=ax[1], format="%+2.f dB")

    # draw inverse wave plot
    inv_wav = istft(amp)
    librosa.display.waveplot(inv_wav, sr=sr, ax=ax[2])
    ax[2].set(title="Re-synthesized signal", xlabel="Time [s]", ylabel="Magnitude")

    plt.savefig(dir + "ex1_result.png")
    plt.show()


if __name__ == "__main__":
    main()
