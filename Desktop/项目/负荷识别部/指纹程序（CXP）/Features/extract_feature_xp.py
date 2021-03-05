from Features.features import Features
import os
import pandas as pd
import json
import numpy as np

power_frequency = 60
sampling_frequency = 30000
path = 'D:/Desktop/项目/负荷识别部/Plaid/PLAID 2018/submetered/'
source_dir = 'submetered_new/'
process_dir = path
csv_dir = os.listdir(os.path.join(path, source_dir))
feature = Features(sampling_frequency=sampling_frequency,
                   power_frequency=power_frequency,
                   is_fft=True,
                   eval_per=1 / power_frequency)
all_data = pd.DataFrame()
for i, file in enumerate(csv_dir):
    soucre_data = pd.read_csv(os.path.join(path, source_dir, file),
                              names=["I", "U"])
    # 取数据中100个周波进行计算，若数据没有100周波，则取全部数据
    if len(soucre_data['I']) < 100 * sampling_frequency / power_frequency:
        I_data = soucre_data['I']
        U_data = soucre_data['U']
    else:
        I_data = soucre_data['I'][int(-100 * sampling_frequency / power_frequency):]
        U_data = soucre_data['U'][int(-100 * sampling_frequency / power_frequency):]
    feature(I_data, U_data)
    data_len = feature.data_len
    id_list = data_len * [file[:-4]]
    dataframe = pd.concat([
        pd.DataFrame({'id': id_list}),
        pd.DataFrame({'i_mean': feature.data_i_mean_list}),
        pd.DataFrame({'i_pp': feature.data_i_pp_list}),
        pd.DataFrame({'i_rms': feature.data_i_rms_list}),
        pd.DataFrame({'i_wave_factor': feature.data_i_wave_factor_list}),
        pd.DataFrame({'i_pp_rms': feature.data_i_pp_rms_list}),
        pd.DataFrame({'i_thd': feature.data_i_thd_list}),
        pd.DataFrame({'u_mean': feature.data_u_mean_list}),
        pd.DataFrame({'u_pp': feature.data_u_pp_list}),
        pd.DataFrame({'u_rms': feature.data_u_rms_list}),
        pd.DataFrame({'P': feature.P_list}),
        pd.DataFrame({'Q': feature.Q_list}),
        pd.DataFrame({'S': feature.S_list}),
        pd.DataFrame({'P_F': feature.P_F_list}),
    ], axis=1)

    u_hm = np.array(feature.u_i_fft_list['U_hm']).transpose()
    u_hp = np.array(feature.u_i_fft_list['U_hp']).transpose()
    i_hm = np.array(feature.u_i_fft_list['I_hm']).transpose()
    i_hp = np.array(feature.u_i_fft_list['I_hp']).transpose()
    z_hm = np.array(feature.u_i_fft_list['Z_hm']).transpose()
    z_hp = np.array(feature.u_i_fft_list['Z_hp']).transpose()

    for times in range(1, u_hm.shape[0]):
        uhm = pd.DataFrame({'u_hm{}'.format(times): u_hm[times, :]})
        uhp = pd.DataFrame({'u_hp{}'.format(times): u_hp[times, :]})
        ihm = pd.DataFrame({'i_hm{}'.format(times): i_hm[times, :]})
        ihp = pd.DataFrame({'i_hp{}'.format(times): i_hp[times, :]})
        zhm = pd.DataFrame({'z_hm{}'.format(times): z_hm[times, :]})
        zhp = pd.DataFrame({'z_hp{}'.format(times): z_hp[times, :]})
        dataframe = pd.concat([dataframe, uhm, uhp, ihm, ihp, zhm, zhp],
                              axis=1)
    all_data = pd.concat([all_data, dataframe], axis=0, ignore_index=True)
    print('正在处理第{}个数据'.format(i+1))
    # type = load_dict[file[0:-4]]['appliance']['type']
    # dataframe.to_csv(process_dir + type + '_' + file, index=True, sep=',')
all_data.to_csv(process_dir + 'all_submetered.csv', index=False, sep=',')