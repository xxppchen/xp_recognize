import numpy as np
from Features.features import *
import pandas as pd
import matplotlib.pyplot as plt


def cut_data(data):
    data = np.array(data)
    cut_data = []
    eval_per_num = int(sampling_frequency * eval_per_time)
    used_num = int(sampling_frequency / power_frequency * eval_used_periods_num)
    for i in range(int((len(data) - used_num) / eval_per_num) + 1):
        cut_data.append(data[i * eval_per_num:i * eval_per_num + used_num])
    return cut_data


def plot_I_data(data):
    max_value = np.max([np.max(data), -np.min(data)])
    plt.ylim(ymin=-2 * max_value, ymax=2 * max_value)
    plt.plot(I_cutted_data[0], 'k')
    plt.show()


power_frequency = 50
sampling_frequency = 20000
eval_per_time = 0.2
eval_used_periods_num = 10
path = r"D:\Desktop\项目\负荷识别部\数据大全\1120数据csv\\"
filename = "ZFB_1_20201030_171108.CSV"
source_data = pd.read_csv(path + filename, skiprows=int(10 * 20000 + 9), names=["T", "U", "I"])
I_cutted_data = cut_data(source_data["I"])
U_cutted_data = cut_data(source_data["U"])
plot_I_data(I_cutted_data[0])
plt.scatter(U_cutted_data[0][:400], I_cutted_data[0][:400])
plt.show()
# fft计算
I_HM_result = []
I_HP_result = []
freq = np.fft.fftfreq(np.size(I_cutted_data[0], 0), 1 / sampling_frequency)
freq = freq[:eval_used_periods_num*7+1]
for i in range(len(I_cutted_data)):
    fft_result = np.fft.fft(I_cutted_data[i], np.size(I_cutted_data[i], 0), axis=0) / np.size(I_cutted_data[i], 0) * 2
    fft_result = fft_result[:eval_used_periods_num * 7 + 1]
    i_hm = np.abs(fft_result) / np.sqrt(2)
    i_hp = np.angle(fft_result) / np.pi * 180
    # hm = fft_result / np.sqrt(2)
    # if hp[0] > 90:
    #     hp[0] -= 180
    #     hm[0] = -hm[0]
    I_HM_result.append(i_hm)
    I_HP_result.append(i_hp)
plt.bar(freq, I_HM_result[0])
plt.show()
I_HM_result = pd.DataFrame(I_HM_result)
I_HP_result = pd.DataFrame(I_HP_result)

U_HM_result = []
U_HP_result = []
freq = np.fft.fftfreq(np.size(U_cutted_data[0], 0), 1 / sampling_frequency)
freq = freq[:eval_used_periods_num*7+1]
for i in range(len(U_cutted_data)):
    fft_result = np.fft.fft(U_cutted_data[i], np.size(U_cutted_data[i], 0), axis=0) / np.size(U_cutted_data[i], 0) * 2
    fft_result = fft_result[:eval_used_periods_num * 7 + 1]
    u_hm = np.abs(fft_result) / np.sqrt(2)
    u_hp = np.angle(fft_result) / np.pi * 180
    # hm = fft_result / np.sqrt(2)
    # if hp[0] > 90:
    #     hp[0] -= 180
    #     hm[0] = -hm[0]
    U_HM_result.append(u_hm)
    U_HP_result.append(u_hp)
plt.bar(freq, U_HM_result[0])
plt.show()
U_HM_result = pd.DataFrame(U_HM_result)
U_HP_result = pd.DataFrame(U_HP_result)

x = np.arange(0, 4000, 1) / 20000
y1 = I_HM_result[0][10] * np.cos(2 * np.pi * 50 * x + I_HP_result[0][10] / 180 * np.pi)
y2 = I_HM_result[0][20] * np.cos(2 * np.pi * 50 * 2 * x + I_HP_result[0][20] / 180 * np.pi)
y3 = I_HM_result[0][30] * np.cos(2 * np.pi * 50 * 3 * x + I_HP_result[0][30] / 180 * np.pi)
y4 = I_HM_result[0][40] * np.cos(2 * np.pi * 50 * 4 * x + I_HP_result[0][40] / 180 * np.pi)
y5 = I_HM_result[0][50] * np.cos(2 * np.pi * 50 * 5 * x + I_HP_result[0][50] / 180 * np.pi)
plt.plot(x[:400], y1[:400])
# plt.plot(x[:400], y1[:400]+y2[:400])
# plt.plot(x[:400], y1[:400]+y2[:400]+y3[:400])
# plt.plot(x[:400], y1[:400]+y2[:400]+y3[:400]+y4[:400])
plt.plot(x[:400], y2[:400])
plt.plot(x[:400], y3[:400])
plt.plot(x[:400], y4[:400])
plt.plot(x[:400], y5[:400])
plt.plot(x[:400], y1[:400]+y2[:400]+y3[:400]+y4[:400]+y5[:400])
plt.show()
aa = U_HP_result-I_HP_result
features = Features(sampling_frequency=sampling_frequency, power_frequency=power_frequency, is_fft=True)
features(source_data["I"], source_data["U"])
print("XXX")
