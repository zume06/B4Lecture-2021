import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

import sys
from pathlib import Path

def stft(data, win_size=1024, overlap=0.5):
    '''
    stft transform data by short-time Fourier transform

    Args
    data: ndarray
    win_size: int
    overlap: float
    '''

    data_length = data.shape[0]
    shift_size = int(win_size*overlap)
    ite = int((data_length - shift_size) / shift_size)
    window = np.hamming(win_size)

    result = []
    for i in range(ite):
        x = data[i*shift_size:i*shift_size+win_size]
        x = window * x
        x = np.fft.fft(x)
        result.append(x)
    
    # shape: (ite, 複素信号) > (複素信号, ite)
    return np.array(result).T

def main(data_path, win_size, overlap):
    wave_data, sr = librosa.load(data_path)
    fig = plt.figure()
    librosa.display.waveplot(wave_data, sr=sr)
    fig.savefig("./result/original.png")

    cs = stft(wave_data)
    print(cs.shape)

    # 振幅スペクトルと位相スペクトルの抽出
    mag, phase = librosa.magphase(cs)  
    # 振幅スペクトルをdB単位に変換
    mag_db = librosa.amplitude_to_db(np.abs(mag))
    fig = plt.figure()
    librosa.display.specshow(mag_db, sr=sr, x_axis='time', y_axis='log')
    fig.savefig("./result/spec.png")

if __name__ == "__main__":
    args = sys.argv
    data_path = args[1]
    win_size = args[2]
    overlap = args[3]

    main(data_path, win_size, overlap)