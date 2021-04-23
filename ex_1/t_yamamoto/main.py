# include flake8, black

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os

"""
def stft():
    """ """


def istft():
    """ """

"""


def main():
    # load audio file
    # get current working directory
    dir = os.path.dirname(__file__) + "/"
    # dir = "C:/Users/yamam/Desktop/lab/2021/B4Lecture-2021/ex_1/t_yamamoto/"
    audio_path = dir + "recording_b4lec_ex1.wav"
    wav, sr = librosa.load(audio_path)

    fig, ax = plt.subplots(nrows=3, sharex=True)
    # draw original wave plot
    librosa.display.waveplot(wav, sr=sr, ax=ax[0])
    ax[0].set(title="Original signal", xlabel="Time [s]", ylabel="Magnitude")

    # alist = stft()
    # alistdb = librosa.amplitude_to_db(np.abs(alist))
    # librosa.display.specshow(Sdb, sr, x_axis="time", y_axis="log")

    plt.savefig(dir + "ex1_result.png")
    plt.show()


if __name__ == "__main__":
    main()
