import numpy as np
from matplotlib import pyplot as plt
import matplotlib.collections as collections
from scipy.signal import find_peaks
import argparse
import librosa
import soundfile


def autocorr(wave_array, sr):
    m = 1000
    temp = 0
    r = np.array([])
    for i in range(m):
        temp = 0
        for j in range(len(wave_array) - m - 1):
            temp += wave_array[j] * wave_array[j+i]
        r = np.append(r, temp)

    peaks_r, _ = find_peaks(r)

    max_peak = 0
    n_max_peak = 0
    for i in range(len(peaks_r)):
        if max_peak < r[peaks_r[i]] and i > 1:
            max_peak = r[peaks_r[i]]
            n_max_peak = peaks_r[i]

    f0 = sr/n_max_peak
    print("autocorr f0 = ", f0)

    plt.title("autocorrelation")
    plt.plot(r, label='r(m)')
    plt.plot(peaks_r, r[peaks_r], "x", label='peak')
    label = 'max peak, $f0$ = {:.2f} Hz'.format(f0)
    plt.plot(n_max_peak, max_peak, "x", color='r', label=label)
    plt.xlabel("m")
    plt.ylabel("r")
    plt.legend()
    plt.savefig("autocorr.png")
    plt.clf()

    return f0


def main():
    parser = argparse.ArgumentParser(
        description='Program for applying digital filter.\nFile name, filter type, filtering frequency are required.')
    parser.add_argument("-f", dest="filename", help='Filename', required=True)
    parser.add_argument("-fs", dest="frame_size", type=int,
                        help='Frame size (optional default = 2048)', required=False, default=2048)
    args = parser.parse_args()
    frame_size = args.frame_size
    wave_array, sr = librosa.load(args.filename, sr=22050)

    time_vector = np.arange(frame_size) / sr
    dt = 1./sr
    freq_vector = np.fft.rfftfreq(frame_size, d=dt)

    signal = wave_array[frame_size:frame_size*2]
    windowed_signal = np.hamming(frame_size) * signal

    spec = np.fft.rfft(signal)
    log_spec = np.log(np.abs(spec))

    plt.plot(freq_vector, log_spec)
    plt.xlabel('frequency (Hz)')
    plt.title('Fourier spectrum')
    plt.savefig("Fourier spectrum.png")
    plt.clf()

    
    fig, ax = plt.subplots()
    cepstrum = np.fft.rfft(log_spec)
    df = freq_vector[1] - freq_vector[0]
    quefrency_vector = np.fft.rfftfreq(log_spec.size, df)

    plt.plot(quefrency_vector, np.abs(cepstrum))

    peaks_ceps, _ = find_peaks(np.abs(cepstrum), threshold=10)

    max_peak = 0
    n_max_peak = 0
    for i in range(len(peaks_ceps)):
        if max_peak < cepstrum[peaks_ceps[i]] and i > 1:
            max_peak = cepstrum[peaks_ceps[i]]
            n_max_peak = peaks_ceps[i]

    f0 = 1/(n_max_peak * quefrency_vector[1])
    label = 'max peak, $f0$ = {:.2f} Hz'.format(f0)

    
    plt.plot(peaks_ceps * quefrency_vector[1],
             np.abs(cepstrum[peaks_ceps]), "x", label='peak')
    plt.plot(n_max_peak * quefrency_vector[1],
             np.real(max_peak), "x", color='r', label=label)
    valid = (quefrency_vector > 1/640) & (quefrency_vector <= 1/82)
    collection = collections.BrokenBarHCollection.span_where(
        quefrency_vector, ymin=0, ymax=np.abs(cepstrum).max(), where=valid, facecolor='green', alpha=0.5, label='valid pitches')
    ax.add_collection(collection)
    plt.xlabel('Quefrency (s)')
    plt.title('Cepstrum')
    plt.legend()
    plt.savefig("cepstrum.png")
    plt.clf()

    #autocorr(wave_array, sr)


if __name__ == "__main__":
    main()
