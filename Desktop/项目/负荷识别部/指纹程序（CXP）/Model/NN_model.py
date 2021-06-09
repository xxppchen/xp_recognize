import torch
import torch.nn as nn
import torch.optim as optim
from Model.linear_model import *
import numpy as np
import pandas as pd
from sklearn import preprocessing
from plot.CM_Polt import *

torch.manual_seed(7)
# 准备数据
source_data = pd.read_csv("Input_Data.csv")
# source_data2 = source_data.copy(deep=True)
# source_data2.sort_values("Label", inplace=True)
X = torch.tensor(np.array(source_data.iloc[:, 1:-1]).astype(np.float32))
Y = source_data.iloc[:, -1]
le = preprocessing.LabelEncoder()
le.fit_transform(Y)
Y = torch.LongTensor(le.transform(Y))
# 训练集划分
n_sample = X.shape[0]
n_val = int(0.2 * n_sample)
n_test = int(0.1 * n_sample)
shuffled_indices = torch.randperm(n_sample)  # 获取随机正整数
train_indices = shuffled_indices[: -n_val-n_test]  # 训练集的索引
val_indices = shuffled_indices[-n_val-n_test:-n_test]  # 验证集的索引
test_indices = shuffled_indices[-n_test:]  # 验证集的索引
Train_X = X[train_indices]
Train_Y = Y[train_indices]
Val_X = X[val_indices]
Val_Y = Y[val_indices]
Test_X = X[test_indices]
Test_Y = Y[test_indices]

# 构建模型
model = MultiLayerModel(X.shape[1], len(le.classes_))
# 构建优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# 损失类
loss_f = nn.CrossEntropyLoss()


# 训练
def training_loop(n_epochs, optimizer, model, loss_fn, train_x, val_x, train_y, val_y):
    for epoch in range(1, n_epochs + 1):
        # 训练集的前向
        train_y_p = model(train_x)
        train_loss = loss_fn(train_y_p, train_y)
        # 验证集的前向
        val_y_p = model(val_x)
        val_loss = loss_fn(val_y_p, val_y)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print('Epoch {}, Train Loss {}, Val Loss {}'.format(epoch, float(train_loss), float(val_loss)))


def test(X_data, Y_data, Is_plot=False):
    out_put = model(X_data)
    test_loss = F.nll_loss(out_put, Y_data).data.item()
    pred = out_put.data.max(1, keepdim=True)[1]
    if Is_plot:
        cm_plot(np.array(Y_data), np.array(pred))
    correct = pred.eq(Y_data.data.view_as(pred)).sum()
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(Y_data),
        100. * correct / len(Y_data)))


training_loop(
    n_epochs=2500,
    optimizer=optimizer,
    model=model,
    loss_fn=loss_f,
    train_x=Train_X,
    val_x=Val_X,
    train_y=Train_Y,
    val_y=Val_Y
)
out = model(X)
pred = out.data.max(1, keepdim=True)[1]
source_data["y"] = pd.DataFrame(np.array(Y))
source_data["y_pre"] = pd.DataFrame(np.array(pred)[:, 0])

test(Train_X, Train_Y)
test(Val_X, Val_Y)
test(Test_X, Test_Y)
test(X, Y, Is_plot=True)

print("XXX")
