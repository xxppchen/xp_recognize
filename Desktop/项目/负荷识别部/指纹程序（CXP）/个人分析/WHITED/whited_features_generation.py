"""
所需特征：
峰值            峰峰值             峰峰值比        峰均比       波形因数              功率           功率因素
THD            低频谐波含量        0次电流含量      2-7次电流谐波含量          1-7次电流相位
开启时长        开启时间占比
"""

from Features.features import *
import os
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt

power_frequency1 = 50
power_frequency2 = 60
sampling_frequency = 44100
data_len_per_file = 1  # 每个文件所取的数据量

path = 'D:/Desktop/项目/负荷识别部/dataset/____WHITED/WHITEDv1.1/'
source_dir = 'CSVData/'
process_dir = path
csv_dir = os.listdir(os.path.join(path, source_dir))
MK = {
    "MK1": [1033.64, 61.4835],
    "MK2": [861.15, 60.200],
    "MK3": [988.926, 60.9562]
}

feature1 = Features(sampling_frequency=sampling_frequency,
                    power_frequency=power_frequency1,
                    is_fft=True,
                    use_periods=10,
                    eval_per=5 / 50)
feature2 = Features(sampling_frequency=sampling_frequency,
                    power_frequency=power_frequency2,
                    is_fft=True,
                    use_periods=10,
                    eval_per=5 / 60)
All_Data = pd.DataFrame()
for i, file in enumerate(csv_dir):
    Data_factor = MK[file.split("_")[3]]
    source_data = pd.read_csv(os.path.join(path, source_dir, file),
                              header=0,
                              names=["U", "I"])
    source_data.iloc[:, 0] = source_data.iloc[:, 0] * Data_factor[0]
    source_data.iloc[:, 1] = source_data.iloc[:, 1] * Data_factor[1]
    # plt.plot(source_data.iloc[:8820, 0])
    # plt.plot(source_data.iloc[:8820, 1])
    # plt.show()
    if file.split("_")[2] == "r8":
        feature2(source_data['I'], source_data['U'])
        feature = feature2
    else:
        feature1(source_data['I'], source_data['U'])
        feature = feature1
    dataframe = pd.concat(
        [
            pd.DataFrame({'load_type': data_len_per_file * [file.split("_")[0]]}),
            pd.DataFrame({'brand': data_len_per_file * [file.split("_")[1]]}),
            pd.DataFrame({'file_number': data_len_per_file * [str(i+1)]}),
            pd.DataFrame({'file_name': data_len_per_file * [file[0:-4]]}),
            pd.DataFrame({'i_max': feature.data_i_max_list[-data_len_per_file - 3:-3]}),
            pd.DataFrame({'i_pp': feature.data_i_pp_list[-data_len_per_file - 3:-3]}),
            pd.DataFrame({'P': feature.P_list[-data_len_per_file - 3:-3]}),
            pd.DataFrame({'i_pp_ratio': feature.data_i_pp_ratio_list[-data_len_per_file - 3:-3]}),
            pd.DataFrame({'i_pp_rms': feature.data_i_pp_rms_list[-data_len_per_file - 3:-3]}),
            pd.DataFrame({'i_wave_factor': feature.data_i_wave_factor_list[-data_len_per_file - 3:-3]}),
            pd.DataFrame({'P_F': feature.P_F_list[-data_len_per_file - 3:-3]}),
            pd.DataFrame({'i_thd': feature.data_i_thd_list[-data_len_per_file - 3:-3]}),
            pd.DataFrame({'low_hd': feature.data_low_hd_list[-data_len_per_file - 3:-3]}),
        ],
        axis=1)

    i_hm = np.array(feature.u_i_fft_list['I_hm']).transpose()
    i_hp = np.array(feature.u_i_fft_list['I_hp']).transpose()

    ihm = pd.DataFrame({'i_hm0/i_hm1': np.array(i_hm[0, -data_len_per_file-3:-3]) / np.array(i_hm[1, -data_len_per_file-3:-3])})
    dataframe = pd.concat(
        [
            dataframe,
            ihm,
        ],
        axis=1)

    for times in range(2, 8):
        ihm_ = np.array(i_hm[times, -data_len_per_file-3:-3]) / np.array(i_hm[1, -data_len_per_file-3:-3])
        ihm = pd.DataFrame({'i_hm{}/i_hm1'.format(times): ihm_})
        dataframe = pd.concat(
            [
                dataframe,
                ihm,
            ],
            axis=1)
    for times in range(1, 8):
        ihp = pd.DataFrame({'i_hp{}'.format(times): i_hp[times, -data_len_per_file-3:-3]})
        dataframe = pd.concat(
            [
                dataframe,
                ihp,
            ],
            axis=1)
    All_Data = pd.concat([All_Data, dataframe], axis=0)
    print('正在处理第{}个数据'.format(i+1))

All_Data.to_csv(process_dir + "Feature_Data_0719(-3).csv", index=False, sep=',')
