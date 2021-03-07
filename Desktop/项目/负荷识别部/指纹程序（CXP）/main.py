from Dataset.MyDataset import *
from torch.utils import data
from Model.linear_model import *
import time
import numpy as np

Dir = 'D:/Desktop/项目/负荷识别部/Plaid/PLAID 2018/submetered/'
meta_path = Dir + 'metadata_submetered2.0.json'
csv_path = Dir + 'all_submetered_20.csv'

used_feas = ["i_thd", "P", "Q"]

d = MyDataset(meta_path, csv_path, LabelType.Is_heat, used_feas=used_feas)
data_loader = data.DataLoader(d, batch_size=512, shuffle=True)

# model = SingleLayerModel(len(used_feas), len(d.encode_list))
# model = DoubleLayerModel(len(used_feas), 8, len(d.encode_list))
model = MultiLayerModer(len(used_feas), len(d.encode_list))
model = model.double()

# 构建优化器
optim = torch.optim.SGD(model.parameters(), lr=0.05)
# 损失类
loss_f = nn.CrossEntropyLoss()
epoch_num = 200

epoch_start_time = time.time()
for epoch in range(1, epoch_num + 1):
    train_loss = 0
    train_acc = 0
    for i, (x_epoch, y_epoch) in enumerate(data_loader):
        # 训练集的前向
        optim.zero_grad()
        train_y_p = model(x_epoch)
        train_loss = loss_f(train_y_p, y_epoch)
        train_acc = np.mean((torch.argmax(train_y_p, 1) == y_epoch).numpy())
        # # 验证集的前向
        # val_y_p = model(val_x)
        # val_loss = loss_fn(val_y_p, val_y)
        train_loss.backward()
        optim.step()
    if epoch % 1 == 0:
        print('Epoch {}, Train Loss {}, Train Acc {}, Time {}'.format(epoch, float(train_loss), train_acc, time.time() - epoch_start_time))
print("XXX")

