import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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

    data_length = len(data)
    shift_size = int(win_size*overlap)
    ite = int((data_length - shift_size) / shift_size)
    window = np.hamming(win_size)

    result = []
    for j in range(0,  data_length, shift_size):
        x = data[j:j+win_size]
        if win_size > len(x):
            break
        x = window * x
        x = np.fft.fft(x)
        result.append(x)

    # result = []
    # for i in range(ite):
    #     x = data[i*shift_size:i*shift_size+win_size]
    #     x = window * x
    #     x = np.fft.fft(x)
    #     result.append(x)
    
    # shape: (ite, 複素信号) > (複素信号, ite)
    return np.array(result).T

def instft(data, win_size=1024, overlap=0.5):
    shift_size = int(win_size*overlap)
    ite = data.shape[0]
    window = np.hamming(win_size)

    result = np.zeros(ite*shift_size+win_size)
    for i in range(ite):
        x = data[i]
        x = np.fft.ifft(x).real * win_size
        result[i*shift_size:i*shift_size+win_size] = result[i*shift_size:i*shift_size+win_size] + x

    return result

def main(data_path, win_size, overlap):
    wave_data, sr = librosa.load(data_path)
    plt.figure()
    librosa.display.waveplot(wave_data, sr=sr)

    cs = stft(wave_data)
    # 振幅スペクトルと位相スペクトルの抽出
    mag, phase = librosa.magphase(cs)  
    # 振幅スペクトルをdB単位に変換
    mag_db = librosa.amplitude_to_db(np.abs(mag))
    plt.figure()
    librosa.display.specshow(mag_db, sr=sr, x_axis='time', y_axis='log')

    inv_data = instft(cs.T)
    plt.figure()
    librosa.display.waveplot(inv_data, sr=sr)

    # save images for pdf
    pdf = PdfPages('./result/result.pdf')
    fignums = plt.get_fignums()
    for fignum in fignums:
        plt.figure(fignum)
        pdf.savefig()
    pdf.close()


if __name__ == "__main__":
    args = sys.argv
    data_path = args[1]
    win_size = args[2]
    overlap = args[3]

    main(data_path, win_size, overlap)