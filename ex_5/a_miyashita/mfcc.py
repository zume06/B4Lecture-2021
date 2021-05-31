import argparse

from scipy.fft import dct
import librosa
import librosa.display
import numpy as np
from matplotlib import pyplot as plt
import soundfile

import ex_1.a_miyashita.main as ex1
import ex_4.a_miyashita.main as ex4


def f2mel(f, f0=None, m0=None):
    if m0 is None:
        if f0 is None:
            f0 = 700
        m0 = 1000/np.log(1000/f0+1)
    else:
        f0 = 1000/(np.exp(1000/m0)-1)
    return m0*np.log(f/f0+1)


def mel2f(m, f0=None, m0=None):
    if m0 is None:
        if f0 is None:
            f0 = 700
        m0 = 1000/np.log(1000/f0+1)
    else:
        f0 = 1000/(np.exp(1000/m0)-1)
    return f0*(np.exp(m/m0)-1)
    

def melfilterbank(bank_size, fft_size, sr):
    maxmel = f2mel(sr/2)
    mel = np.linspace(0, maxmel, bank_size+2)
    f = (mel2f(mel)*fft_size)/sr
    f = np.round(f).astype(np.int)
    filterbank = np.zeros((bank_size+2, fft_size//2+1))
    for i in range(bank_size+1):
        line = np.linspace(0, 1, f[i+1]-f[i], endpoint=False)
        filterbank[i, f[i]:f[i+1]] = 1-line
        filterbank[i+1, f[i]:f[i+1]] = line
    return filterbank[1:-1]


def get_mfcc(signal, bank_size, fft_size, sr):
    spec = np.abs(ex1.stft(signal, fft_size))
    filter = melfilterbank(bank_size, fft_size, sr)
    melspec = filter @ spec
    melspec_db = librosa.amplitude_to_db(melspec)
    ceps = dct(melspec_db, axis=0)
    mfcc = ceps[1:13]
    return mfcc

def delta(input, neighbor):
    frames = ex4.frame(input, 2*neighbor+1, 1)
    ite = np.arange(-neighbor, neighbor+1)
    d = ite @ frames / (neighbor*(neighbor+1)*(2*neighbor+1)/3)
    return d


def main():
    # process args
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("sc", type=str, help="input filename with extension .wav")
    args = parser.parse_args()

    # load
    signal, sr = soundfile.read(args.sc)

    bank_size = 20
    fft_size = 1024

    spec = ex1.stft(signal, fft_size)
    spec_db = librosa.amplitude_to_db(np.abs(spec))

    mfcc = get_mfcc(signal, bank_size=bank_size, fft_size=fft_size, sr=sr)
    dmfcc = delta(mfcc.T, 2)
    ddmfcc = delta(dmfcc, 2)
    dmfcc = dmfcc.T
    ddmfcc = ddmfcc.T

    # plot
    fig, ax = plt.subplots(4, 1)
    fig.subplots_adjust(hspace=1.0)

    img = ax[0].imshow(spec_db, cmap='rainbow', aspect='auto', origin='lower')
    ax[0].set(title='Spectrogram', ylabel='Frequency[Hz]')
    fig.colorbar(img, ax=ax[0])

    img= ax[1].imshow(mfcc, cmap='rainbow', aspect='auto', origin='lower')
    ax[1].set(title='MFCC')
    fig.colorbar(img, ax=ax[1])
    
    img = ax[2].imshow(dmfcc, cmap='rainbow', aspect='auto', origin='lower')
    ax[2].set(title='$\Delta$MFCC')
    fig.colorbar(img, ax=ax[2])
    
    img = ax[3].imshow(ddmfcc, cmap='rainbow', aspect='auto', origin='lower')
    ax[3].set(title='$\Delta\Delta$MFCC', xlabel='Time[s]')
    fig.colorbar(img, ax=ax[3])
    
    plt.show()

if __name__ == "__main__":
    main()