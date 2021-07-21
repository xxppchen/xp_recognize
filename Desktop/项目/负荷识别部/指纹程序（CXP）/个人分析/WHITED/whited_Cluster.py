import numpy as np
import pandas as pd
from sklearn.externals import joblib
import random
import matplotlib.pyplot as plt

random.seed(12)

# 聚类模型读取
model1 = joblib.load('../../Model/cluster_model1.pkl')
model2 = joblib.load('../../Model/cluster_model2.pkl')
model3 = joblib.load('../../Model/cluster_model3.pkl')
model4 = joblib.load('../../Model/cluster_model4.pkl')
# 原始数据读取，后续新增标签数据，
feature_data = pd.read_csv('D:/Desktop/项目/负荷识别部/dataset/____WHITED/WHITEDv1.1/Feature_Data_0719(-3).csv')
feature_data.sort_values("file_name", inplace=True)
feature_data = feature_data[feature_data["P"] >= 3]


def my_point(a, b):
    return a, b


def my_point2(a, b, c, d):
    return a, b, c, d


unique_brand = dict(feature_data.groupby("load_type")["brand"].unique())

"""
聚类1---Low_hd--稳定型/时变型
"""
# 聚类1标签生成
X1 = np.array(feature_data["low_hd"]).reshape(-1, 1)
yhat1 = model1.predict(X1)
feature_data["cluster1"] = yhat1

"""
聚类2---i_hp1--对齐型/偏移型
"""
# 聚类2标签生成
X2 = np.array(feature_data["i_hp1"]).reshape(-1, 1)
X2 = np.cos(X2 / 180 * np.pi)
yhat2 = model2.predict(X2)
feature_data["cluster2"] = yhat2

"""
聚类3---偶次谐波含量--是否对称
"""
X3_ = np.array(feature_data[["i_hm2/i_hm1", "i_hm4/i_hm1", "i_hm6/i_hm1"]])
X3 = np.sqrt(np.sum(np.square(X3_), axis=1)).reshape(-1, 1)
yhat3 = model3.predict(X3)
feature_data["cluster3"] = yhat3

"""
聚类4---奇次谐波含量--是否较多波形毛刺
"""
X4_ = np.array(feature_data[["i_hm3/i_hm1", "i_hm5/i_hm1", "i_hm7/i_hm1"]])
X4 = np.sqrt(np.sum(np.square(X4_), axis=1)).reshape(-1, 1)
yhat4 = model4.predict(X4)
feature_data["cluster4"] = yhat4


"""
聚类结果画图
"""
clusters = np.unique(yhat4)
cluster_file = {}
cluster_nums = {}
fig1, ax1 = plt.subplots()
for cluster in clusters:
    row_ix = np.where(yhat4 == cluster)
    plt.scatter(X4[row_ix, 0], (X4[row_ix, 0]).shape[1]*[0], label=cluster)
    cluster_file['cluster_{}'.format(cluster)] = np.unique(np.array(feature_data)[row_ix, 3])
    cluster_nums['cluster_{}'.format(cluster)] = np.unique(np.array(feature_data)[row_ix, 2])
plt.legend()
plt.show()

for cluster_name, file_names in cluster_file.items():
    i = 0
    if cluster_name == "cluster_0":
        continue
    for file_name in file_names:
        MK = {
            "MK1": [1033.64, 61.4835],
            "MK2": [861.15, 60.200],
            "MK3": [988.926, 60.9562]
        }
        Data_factor = MK[file_name.split("_")[3]]
        path = 'D:/Desktop/项目/负荷识别部/dataset/____WHITED/WHITEDv1.1/'
        source_dir = 'CSVData/'
        data = pd.read_csv(path + source_dir + file_name + '.csv',
                           header=0,
                           names=["U", "I"])
        data.iloc[:, 0] = data.iloc[:, 0] * Data_factor[0]
        data.iloc[:, 1] = data.iloc[:, 1] * Data_factor[1]
        data = data.iloc[-30 * 882:-20 * 882, :]

        # if file_name == "SolderingIron_80W_r1_MK2_20151012113036":
        #     data_temp = np.array(data["I"])
        #     index = 10
        #     x = np.fft.fft(data_temp, np.size(data_temp, 0), axis=0) / np.size(data_temp, 0) * 2
        #     hm = np.abs(x) / np.sqrt(2)
        #     lh = np.mean(hm[1:index]) / hm[index]


        # ########## 作图 ##########
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        T = np.arange(0, 0.2, 0.2 / 8820)
        line2, = ax1.plot(data["U"], 'b')
        line3, = ax2.plot(data["I"], 'r', linewidth=2)
        plt.ylim(ymin=-np.max(data["I"]) * 2, ymax=np.max(data["I"]) * 2)
        plt.legend([line3, line2], ['Current[A]', 'Voltage[V]'], loc='upper right')
        ax1.set_xlabel(u'Time[s]')
        ax1.set_ylabel(u'Voltage[V]')
        ax2.set_ylabel(u'Current[A]')
        # 分两类
        plt.savefig("D:\Desktop\项目\负荷识别部\指纹程序（CXP）\Cluster_Result_Whited\cluster4\\" + str(cluster_name) + "\\" + str(file_name) + ".jpg")
        plt.close()
        i += 1
        print("正在处理" + cluster_name + "类中的第" + str(i) + "条数据...")
print("XXX")
