import numpy as np
import sys
import matplotlib.pyplot as plt
import librosa
import scipy
from scipy.io.wavfile import read
import wave


def convolution(x, h, N, M):
    y = np.zeros(N + M)
    for i in range(N):
        y[i : i + M] += x[i] * h
    return y


def plot_wave(s, title):
    plt.plot(s)
    plt.xlabel("time[s]")
    plt.ylabel("amplitude")
    plt.suptitle(title)
    plt.grid()
    # plt.show()
    plt.savefig(title)
    plt.clf()
    plt.close()


def plot_freq(s, fs, title):
    S = np.abs(np.fft.fft(s))
    f = np.fft.fftfreq(s.shape[0], d=1.0 / fs)
    plt.plot(f, S)
    plt.xlim(0, 2000)
    plt.xlabel("frequency[Hz]")
    plt.ylabel("amplitude")
    plt.suptitle(title)
    plt.grid()
    # plt.show()
    plt.savefig(title)
    plt.clf()
    plt.close()


def sinc(x):
    return np.where(x == 0, 1, np.sin(x) / x)


def save(data, fs, bit, filename):
    """output to WaveFile"""
    wf = wave.open(filename, "w")
    wf.setnchannels(1)
    wf.setsampwidth(bit / 8)
    wf.setframerate(fs)
    wf.writeframes(data)
    wf.close()


def main():
    # LPF
    """
    fs : Smapling rate
    N : Length of original signal
    M : Degree of Filter
    fe : Cut off frequency
    h : LPF
    """
    args = sys.argv
    wav_filename = args[1]
    fs, origin_signal = read(wav_filename)
    N = len(origin_signal)
    M = 255
    fe = 1000
    h = np.zeros(M)

    title1 = "originalwave"
    plot_wave(origin_signal, title1)
    title2 = "originalwave_frequency"
    plot_freq(origin_signal, fs, title2)

    h = 2 * fe / fs * sinc((2 * np.pi * fe * np.arange(-M // 2, M // 2 + 1)) / fs)

    window = np.hanning(M + 1)
    h *= window

    title3 = "LPF_time"
    plot_wave(h, title3)
    title4 = "LPF_frequency"
    plot_freq(h, fs, title4)

    y = convolution(origin_signal, h, N, M + 1)

    title5 = "CutWave"
    plot_wave(y, title5)
    title6 = "CutWave_frequency"
    plot_freq(y, fs, title6)

    f, t, Sxx = scipy.signal.spectrogram(y, fs, nperseg=512)

    plt.figure()
    plt.pcolormesh(t, f, Sxx, cmap="GnBu")
    plt.xlim([0, 10])
    plt.ylim([0, 2000])
    plt.xlabel("Time [sec]")
    plt.ylabel("Freq [Hz]")
    plt.colorbar()
    plt.savefig("spec")

    save(y, fs, 16, "sample_LPF.wav")


if __name__ == "__main__":
    main()
