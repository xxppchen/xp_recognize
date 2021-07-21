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

power_frequency = 50
sampling_frequency = 20000
data_len_per_file = 10  # 每个文件所取的数据量

path = 'D:/Desktop/项目/负荷识别部/dataset/公司数据/'
source_dir = '数据集（公司电器）/单电器csv/'
process_dir = path
csv_dir = os.listdir(os.path.join(path, source_dir))
with open(path+"single.json") as data_file:
    meta = json.load(data_file)
feature = Features(sampling_frequency=sampling_frequency,
                   power_frequency=power_frequency,
                   is_fft=True,
                   use_periods=10,
                   eval_per=5 / 50)
All_Data = pd.DataFrame()
for i, file in enumerate(csv_dir):
    source_data = pd.read_csv(os.path.join(path, source_dir, file),
                              names=["U", "I"],
                              skiprows=9,
                              usecols=[1, 2])
    file_id = file.split("_")[0]
    # plt.plot(source_data.iloc[:4000, 0])
    # plt.plot(source_data.iloc[:4000, 1])
    # plt.show()
    feature(source_data['I'], source_data['U'])
    dataframe = pd.concat(
        [
            pd.DataFrame({'load_type': data_len_per_file * [meta[file_id]["type"]]}),
            pd.DataFrame({'brand': data_len_per_file * [meta[file_id]["brand"]]}),
            pd.DataFrame({'xinghao': data_len_per_file * [meta[file_id]["xinghao"]]}),
            pd.DataFrame({'file_number': data_len_per_file * [file_id]}),
            pd.DataFrame({'i_max': feature.data_i_max_list[-data_len_per_file-10:-10]}),
            pd.DataFrame({'i_pp': feature.data_i_pp_list[-data_len_per_file-10:-10]}),
            pd.DataFrame({'P': feature.P_list[-data_len_per_file-10:-10]}),
            pd.DataFrame({'i_pp_ratio': feature.data_i_pp_ratio_list[-data_len_per_file-10:-10]}),
            pd.DataFrame({'i_pp_rms': feature.data_i_pp_rms_list[-data_len_per_file-10:-10]}),
            pd.DataFrame({'i_wave_factor': feature.data_i_wave_factor_list[-data_len_per_file-10:-10]}),
            pd.DataFrame({'P_F': feature.P_F_list[-data_len_per_file-10:-10]}),
            pd.DataFrame({'i_thd': feature.data_i_thd_list[-data_len_per_file-10:-10]}),
            pd.DataFrame({'low_hd': feature.data_low_hd_list[-data_len_per_file-10:-10]}),
        ],
        axis=1)

    i_hm = np.array(feature.u_i_fft_list['I_hm']).transpose()
    i_hp = np.array(feature.u_i_fft_list['I_hp']).transpose()

    ihm = pd.DataFrame({'i_hm0/i_hm1': np.array(i_hm[0, -data_len_per_file-10:-10]) / np.array(i_hm[1, -data_len_per_file-10:-10])})
    dataframe = pd.concat(
        [
            dataframe,
            ihm,
        ],
        axis=1)

    for times in range(2, 8):
        ihm_ = np.array(i_hm[times, -data_len_per_file-10:-10]) / np.array(i_hm[1, -data_len_per_file-10:-10])
        ihm = pd.DataFrame({'i_hm{}/i_hm1'.format(times): ihm_})
        dataframe = pd.concat(
            [
                dataframe,
                ihm,
            ],
            axis=1)
    for times in range(1, 8):
        ihp = pd.DataFrame({'i_hp{}'.format(times): i_hp[times, -data_len_per_file-10:-10]})
        dataframe = pd.concat(
            [
                dataframe,
                ihp,
            ],
            axis=1)
    All_Data = pd.concat([All_Data, dataframe], axis=0)
    print('正在处理第{}个数据'.format(i+1))

All_Data.to_csv(process_dir + "Feature_Data_0719(-10).csv", index=False, sep=',')
