import os
import librosa
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

out_dir = "./out"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# 音声ファイルの読み込み
fname = "./sound/source.aifc"
y, sr = librosa.load(fname)
total_time = np.arange(0, len(y)) / sr
# 窓幅・窓関数の設定
window = 1024
win_func = signal.hamming(window)

# フーリエ変換
spec = []
fft_data = []
for idx in range(0, len(y), 512):
    frame = y[idx:idx+1024]
    if len(frame) == 1024:
        frame = frame * win_func
        fft_res = np.fft.rfft(frame)
        # 逆変換用のデータを準備
        fft_data.append(fft_res)
        res = np.log(np.abs(fft_res) ** 2)
        spec.append(res)

# 逆変換
ifft_res = []
for i in range(len(fft_data)):
    f = np.fft.irfft(fft_data[i])
    if i == 0:
        ifft_res.extend(f)
    else:
        ifft_res.extend(f[512:1024])
# 欠損値をゼロ埋め
for i in range(len(total_time) - len(ifft_res)):
    ifft_res.append(0)

# オリジナルの信号と逆変換後の信号を描画・保存
fig1 = plt.figure(tight_layout=True)
plt.subplot(2, 1, 1)
plt.plot(total_time, y)
plt.title("Original signal")
plt.xlabel("Time[s]")
plt.ylabel("Magnitude")
plt.subplot(2, 1, 2)
plt.plot(total_time, ifft_res)
plt.title("Re-synthesized signal")
plt.xlabel("Time[s]")
plt.ylabel("Magnitude")
plt.show(block=True)
fig1.savefig(f"{out_dir}/Signal.png")

# スペクトログラムを描画・保存
fig2 = plt.figure()
plt.imshow(np.array(spec).T, extent=[0, total_time[-1], 0, (sr/2) / 1000], aspect="auto")
plt.title("Spectrogram")
plt.xlabel("Time[s]")
plt.ylabel("Frequency[kHz]")
plt.colorbar()
plt.show(block=True)
fig2.savefig(f"{out_dir}/Spectrogram.png")
