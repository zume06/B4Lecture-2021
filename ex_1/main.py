import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy

import pdb

def specplot(mat, sr, frames):
    plt.imshow(mat, aspect="auto", extent=[0, frames / sr, 0, sr//2])

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
        spec.append(fft_result)
        # spec.append(np.abs(fft_result))    
    spec = np.array(spec).T
    abs_spec = np.abs(spec)
    return spec, abs_spec

def istft(spec, win_len, wav_len):
    ifft_slices = [] 
    start = 0
    signal = np.zeros(wav_len)
    for slice in spec.T:
        # pdb.set_trace()
        ifft_result = np.fft.ifft(slice)
        if start+win_len <= wav_len:
            signal[start: start+win_len] += np.real(ifft_result)
        else:
            signal[start: wav_len] += np.real(ifft_result[0: wav_len - start])
        start += win_len // 2
    return signal


    

if __name__ == "__main__":
    # input_wav = '/home/kevingeng/nas01/home/JNAS/WAVES_DT/F002/NP/NF002001_DT.wav'
    input_wav = '/home/kevingeng/B4Lecture-2021/ex_1/arctic_b0340.wav'
    y, sr = librosa.load(input_wav)
    plt.subplot(311)
    plt.plot(y)

    win_len = 512
    spec, abs_spec = stft(y, win_len=win_len)    
    abs_spec = abs_spec[len(abs_spec[0]//2): -1] # 対称のため半分しか取らない
    mag_spec = 20 * np.log10(abs_spec)
    plt.subplot(312)
    specplot(mag_spec,sr=sr, frames=len(y))
    
    signal= istft(spec, win_len=win_len, wav_len=len(y))    
    plt.subplot(313)
    plt.plot(signal)
    plt.show()