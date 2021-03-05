from Dataset.MyDataset import *
from torch.utils import data

Dir = 'D:/Desktop/项目/负荷识别部/Plaid/PLAID 2018/submetered/'
meta_path = Dir + 'metadata_submetered2.0.json'
csv_path = Dir + 'all_submetered.csv'
used_feas = ["i_mean", "i_thd", "P", "Q"]
d = MyDataset(meta_path, csv_path, LabelType.Type, used_feas=used_feas)
features, label = d[0]

data_loader = data.DataLoader(d,
                              batch_size=4,
                              shuffle=False)
print(len(d))
for i, (f, labels) in enumerate(data_loader):
    print(f)
    print(labels)
print("XXX")