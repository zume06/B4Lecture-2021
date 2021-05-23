import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import scipy as sp
import librosa
import librosa.display
import matplotlib.pyplot as plt

from modules.autocorrelation import autocorrelation, get_ac_peaks
from modules.cepstrum import get_cepstrum, get_ceps_peaks, get_envelope
from modules.spectrogram import spectrogram
from modules.lpc import levinson_durbin_method

TIME_TEMPLATE = '%Y%m%d%H%M%S'


def main(args):
    audio_path = Path(args.audio_path)

    result_path = Path(args.save_path)
    timestamp = datetime.now().strftime(TIME_TEMPLATE)
    result_path = result_path/timestamp
    if not result_path.exists():
        try:
            result_path.mkdir(parents=True)
        except Exception as err:
            print(err)

    wave_data, sr = librosa.load(audio_path)
    win_size = 1024

    db_spec = spectrogram(
        wave_data, win_size=1024, overlap=0.5, mode='faster', scale='db')

    ac = autocorrelation(wave_data, win_size, sr)
    peaks = get_ac_peaks(ac)
    f0 = sr / peaks
    times = np.linspace(0, (wave_data.size - win_size) / sr, f0.size)
    librosa.display.specshow(
        db_spec, sr=sr, x_axis='time', y_axis='linear', cmap='rainbow')
    plt.plot(times, f0)
    plt.ylim(0, 8000)
    plt.title("F0 AC")
    plt.xlabel("Time [sec]")
    plt.ylabel("Frequency [Hz]")
    plt.savefig(result_path/'f0_ac.png')
    plt.clf()
    plt.close()

    ceps_db = get_cepstrum(wave_data, is_clipping=False, is_framing=True)
    peak_index = get_ceps_peaks(ceps_db, sr)
    f0 = sr / peak_index
    times = np.linspace(0, (wave_data.size - win_size) / sr, f0.size)
    librosa.display.specshow(
        db_spec, sr=sr, x_axis='time', y_axis='linear', cmap='rainbow')
    plt.plot(times, f0)
    plt.ylim(0, 8000)
    plt.title("F0 Cepstrum")
    plt.xlabel("Time[sec]")
    plt.ylabel("Frequency[Hz]")
    plt.savefig(result_path/'f0_ceps.png')
    plt.clf()
    plt.close()

    clip_size = 8192
    ceps_db = get_cepstrum(wave_data, is_clipping=True, is_framing=False, clip_size=clip_size)
    ceps_env = get_envelope(ceps_db, 100)

    a, e = levinson_durbin_method(wave_data, 100)
    h = sp.signal.freqz(np.sqrt(e), a, clip_size, "whole")[1]
    lpc_env = 20 * np.log10(np.abs(h))

    freq = np.fft.rfft(wave_data, clip_size)
    amp = 20 * np.log10(np.abs(freq))

    fscale = np.fft.fftfreq(clip_size, d=1.0 / sr)
    # freq_nyquist = sr / 2
    # fscale = np.linspace(0, freq_nyquist, freq.shape[0]) / 1000

    plt.plot(fscale[:clip_size//2], amp[:clip_size//2], label='spectrum')
    plt.plot(fscale[:clip_size//2], ceps_env[:clip_size//2], label='cepstrum')
    plt.plot(fscale[:clip_size//2], lpc_env[:clip_size//2], label='lpc')
    plt.title("amplitude characteristic")
    plt.xlabel("Frequency[Hz]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.savefig(result_path/'enveloop.png')
    plt.clf()
    plt.close()


if __name__ == "__main__":
    description = 'Example: python main.py ./audio/sample.wav -s ./result'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('audio_path', help='path to audio data')
    parser.add_argument('-s', '--save-path', default='./result', help='path to save the result')

    args = parser.parse_args()

    main(args)
