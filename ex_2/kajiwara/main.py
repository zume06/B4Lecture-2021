import argparse
from pathlib import Path
from datetime import datetime

import librosa.display
import matplotlib.pyplot as plt
import soundfile

from modules.spectrogram import spectrogram
from modules.filters import *
from modules.utils import *

TIME_TEMPLATE = '%Y%m%d%H%M%S'


def main(args):
    data_path = Path(args.data_path)
    result_path = Path(args.save_path)

    assert data_path.exists(), '{} is not exist'.format(data_path)
    assert result_path.exists(), '{} is not exist'.format(result_path)

    wave_data, sr = librosa.load(data_path)
    spec = spectrogram(wave_data)

    l_cutoff, h_cutoff = 1000, 3000
    bpf = band_pass_filter(sr, h_cutoff, l_cutoff, 255)
    wave_data_filtered = conv1d(wave_data, bpf)
    spec_filtered = spectrogram(wave_data_filtered)

    fig, ax = plt.subplots(nrows=2, ncols=2)

    librosa.display.waveplot(wave_data, sr=sr, x_axis='time', ax=ax[0][0])
    ax[0][0].set(title='Original wave', xlabel="Time [s]", ylabel="Magnitude")
    ax[0][0].label_outer()

    spec_img_1 = librosa.display.specshow(
        spec, sr=sr, x_axis='time', y_axis='log', ax=ax[0][1])
    fig.colorbar(spec_img_1, ax=ax[0][1])
    ax[0][1].set(title='Spectrum', xlabel="Time [s]", ylabel="Frequency [Hz]")
    ax[0][1].yaxis.set_ticks([0, 128, 512, 2048, 8192])
    ax[0][1].label_outer()

    librosa.display.waveplot(wave_data_filtered, sr=sr,
                             x_axis='time', ax=ax[1][0])
    ax[1][0].set(title='Filterd wave', xlabel="Time [s]", ylabel="Magnitude")
    ax[1][0].label_outer()

    spec_img_2 = librosa.display.specshow(
        spec_filtered, sr=sr, x_axis='time', y_axis='log', ax=ax[1][1])
    fig.colorbar(spec_img_2, ax=ax[1])
    ax[1][1].set(title='Spectrum', xlabel="Time [s]", ylabel="Frequency [Hz]")
    ax[1][1].yaxis.set_ticks([0, 128, 512, 2048, 8192])
    ax[1][1].label_outer()

    # save result
    timestamp = datetime.now().strftime(TIME_TEMPLATE)
    # plt.savefig(result_path.joinpath(timestamp+'-result.pdf'))
    plt.savefig(result_path.joinpath(timestamp+'-result.png'))
    soundfile.write(result_path.joinpath(
        timestamp+'-result.wav'), wave_data_filtered, sr)


if __name__ == "__main__":
    description = 'Example: python main.py ./sample.wav'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('data_path', help='path of data')
    parser.add_argument('-s', '--save_path', default='./',
                        help='path to save the result')

    args = parser.parse_args()

    main(args)
