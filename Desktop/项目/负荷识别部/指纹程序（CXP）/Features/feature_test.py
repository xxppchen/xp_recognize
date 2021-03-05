import numpy as np
from Features.features import *
import pandas as pd

path = r"D:\Desktop\项目\负荷识别部\暂态特征整理\source_data\\"
filename = "RSH_3_20201025_105528.CSV"
source_data = pd.read_csv(path + filename, skiprows=int(20 * 20000 + 9), names=["T", "U", "I"], nrows=int(0.16 * 20000))
base_features1 = BaseFeatures()
base_features1(source_data["I"])
base_features2 = BaseFeatures(is_fft=True, sampling_frequency=20000)
base_features2(source_data["I"])
base_features3 = BaseFeatures(is_fft=True, sampling_frequency=20000, is_wavelet=True, wt_level=6)
base_features3(source_data["I"])
expert_features1 = ExpertFeatures()
expert_features1(source_data["U"], source_data["I"])
expert_features2 = ExpertFeatures(is_fft=True, sampling_frequency=20000)
expert_features2(source_data["U"], source_data["I"])
features1 = Features(sampling_frequency=20000)
features1(source_data["I"], source_data["U"])
features2 = Features(sampling_frequency=20000, eval_per=0.04)
features2(source_data["I"], source_data["U"])
features3 = Features(sampling_frequency=20000, is_fft=True)
features3(source_data["I"], source_data["U"])
features4 = Features(sampling_frequency=20000, is_wavelet=True, wt_level=6)
features4(source_data["I"], source_data["U"])
features5 = Features(sampling_frequency=20000, is_fft=True)
features5(source_data["I"])

print("XXX")

