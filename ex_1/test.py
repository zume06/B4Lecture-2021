import matplotlib.pyplot as plt
import numpy as np

'''超簡単な正弦波を作る'''
f = 10 # 周波数f(frequency),単位はHz(サイクル毎秒)
f_s = 100 # サンプリングレート。1秒あたりの測定数

t = np.linspace(0, 2, 2 * f_s, endpoint = False)# 時間を0から2秒に指定しサンプリング分割を行う(終点は要素に含まない)
x = np.sin(f * 2 * np.pi * t) # x=sin(2πf)の波形作成

'''グラフに表示'''
fig, ax= plt.subplots()
ax.plot(t, x) # 横軸にt, 縦軸にxの値をプロットする
ax.set_xlabel('Time[s]')
ax.set_ylabel('Signal amplitude')

'''フーリエ変換'''
X = np.fft.fft(x) # 波形のフーリエ変換
freqs = np.fft.fftfreq(len(x)) * f_s # サンプリング周波数を返す

'''グラフに表示'''
fig2, ax2= plt.subplots()
ax2.stem(freqs, np.abs(X), use_line_collection=True) # フーリエ変換の結果をステム(茎)プロットする
ax2.set_xlabel('Frequency in Hertz[Hz]')
ax2.set_ylabel('Frequency Domain (Spectrum) Magnitude')
ax2.set_xlim(- f_s / 2, f_s / 2)
ax2.set_ylim(-5, 110)

plt.show()