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

power_frequency = 60
sampling_frequency = 30000
data_len_per_file = 1  # 每个文件所取的数据量

path = 'D:/Desktop/项目/负荷识别部/Plaid/PLAID 2018/submetered/'
source_dir = 'submetered_new/'
process_dir = path
meta_path = path + 'metadata_submetered.json'
csv_dir = os.listdir(os.path.join(path, source_dir))


feature = Features(sampling_frequency=sampling_frequency,
                   power_frequency=power_frequency,
                   is_fft=True,
                   use_periods=10,
                   eval_per=5 / 60)
with open(meta_path) as data_file:
    meta = json.load(data_file)
All_Data = pd.DataFrame()
for i, file in enumerate(csv_dir):
    source_data = pd.read_csv(os.path.join(path, source_dir, file),
                              names=["I", "U"])
    feature(source_data['I'], source_data['U'])
    load_type = meta[file[0:-4]]["appliance"]["type"]
    brand = meta[file[0:-4]]["appliance"]["brand"]
    model_number = meta[file[0:-4]]["appliance"]["model_number"]
    dataframe = pd.concat(
        [
            pd.DataFrame({'load_type': data_len_per_file * [load_type]}),
            pd.DataFrame({'brand': data_len_per_file * [brand]}),
            pd.DataFrame({'model_number': data_len_per_file * [model_number]}),
            pd.DataFrame({'file_name': data_len_per_file * [file[0:-4]]}),
            pd.DataFrame({'i_max': feature.data_i_max_list[-data_len_per_file-3:-3]}),
            pd.DataFrame({'i_pp': feature.data_i_pp_list[-data_len_per_file-3:-3]}),
            pd.DataFrame({'P': feature.P_list[-data_len_per_file-3:-3]}),
            pd.DataFrame({'i_pp_ratio': feature.data_i_pp_ratio_list[-data_len_per_file-3:-3]}),
            pd.DataFrame({'i_pp_rms': feature.data_i_pp_rms_list[-data_len_per_file-3:-3]}),
            pd.DataFrame({'i_wave_factor': feature.data_i_wave_factor_list[-data_len_per_file-3:-3]}),
            pd.DataFrame({'P_F': feature.P_F_list[-data_len_per_file-3:-3]}),
            pd.DataFrame({'i_thd': feature.data_i_thd_list[-data_len_per_file-3:-3]}),
            pd.DataFrame({'low_hd': feature.data_low_hd_list[-data_len_per_file-3:-3]}),
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
    print('正在处理第{}个数据'.format(i))

All_Data.to_csv(process_dir + "Feature_Data_0603(-3).csv", index=False, sep=',')
