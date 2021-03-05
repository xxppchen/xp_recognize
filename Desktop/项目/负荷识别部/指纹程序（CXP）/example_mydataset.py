from MyDataset import *

Dir = 'D:/Desktop/项目/负荷识别部/Plaid/PLAID 2018/submetered/'
meta_path = Dir + 'metadata_submetered2.0.json'
csv_path = Dir + 'all_submetered.csv'
used_feas = ["i_mean", "i_thd", "P", "Q"]
d = MyDataset(meta_path, csv_path, LabelType.Type, used_feas=used_feas)
data, label = d[0]
print("XXX")