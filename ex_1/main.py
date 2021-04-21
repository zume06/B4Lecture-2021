import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy

import pdb


def calc_fft(data, samplerate):
    spectrum = scipy.fftpack.fft(data)
    amp = np.sqrt((spectrum.real ** 2) + (spectrum.imag ** 2))
    amp = amp / (len(data) / 2)
    phase = np.arctan2(spectrum.imag, spectrum.real)
    phase = np.degrees(phase)
    freq = np.linspace(0, samplerate, len(data))
    return spectrum, amp, phase, freq

def stft(data, win_len):
    start = 0
    window_n = np.hanning(win_len)
    start_points = [int(start)]

    slices = []
    while start < len(data):
        # start_points.append(int(start))
        if len(data) - start >= win_len:
            slices.append(data[start: start+win_len])
        else:
            valid_signal = data[start: -1]
            slices.append(np.pad(valid_signal, (0, win_len-len(valid_signal)), 'constant'))
        start += win_len // 2
    slices = np.array(slices)
    # windowing
    windowed_slices = [window_n * slices[i] for i in range(len(slices[0:]))]

    spec = []
    for short_slice in slices:
        fft_result = np.fft.fft(short_slice)
        spec.append(np.abs(fft_result))    
    freqs = np.fft.fftfreq(len(fft_result)) 
    spec = np.array(spec)
    pdb.set_trace()
    return spec

if __name__ == "__main__":
    input_wav = '/home/kevingeng/nas01/home/JNAS/WAVES_DT/F002/NP/NF002001_DT.wav'
    y, sr = librosa.load(input_wav)
    win_len = 1024
    spec = stft(y, win_len=win_len)
    pdb.set_trace()
    # amp = np.abs(a)
    # N = len(amp)
    # amp_normal = amp / (N / 2)
    # amp_normal[0] /= 2
    # # pdb.set_trace()

    # # plt.imshow(amp_normal)
    # amp_normal_db = librosa.amplitude_to_db(amp_normal, ref=np.max)
    # librosa.display.specshow(amp_normal_db, sr=sr,
    #                          hop_length=16000, y_axis='linear')
    # # plt.plot(y)

    # plt.show()
