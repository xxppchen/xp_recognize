from MyDataset import *

meta_path = 'D:/Desktop/项目/负荷识别部/Plaid/PLAID 2018/submetered/'
meta_name = 'metadata_submetered.json'
csv_path = meta_path + 'submetered_new/'
d = MyDataset(meta_path, meta_name, csv_path, LabelType.Type)
data, label = d[0]
print("XXX")