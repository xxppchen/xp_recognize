import numpy as np
import pandas as pd
from sklearn.externals import joblib
import random

random.seed(12)

# 聚类模型读取
model1 = joblib.load('../cluster_model1.pkl')
model2 = joblib.load('../cluster_model2.pkl')
model3 = joblib.load('../cluster_model3.pkl')
model4 = joblib.load('../cluster_model4.pkl')
# 原始数据读取，后续新增标签数据，
feature_data = pd.read_csv(r'D:/Desktop/项目/负荷识别部/Plaid/PLAID 2018/submetered/Feature_Data_0603(-3).csv')
feature_data.sort_values("file_name", inplace=True)
feature_data = feature_data[feature_data["P"] >= 3]


def my_point(a, b):
    return a, b


def my_point2(a, b, c, d):
    return a, b, c, d


feature_data['(brand，model_number)'] = feature_data.apply(lambda row: str(my_point(row['brand'], row['model_number'])),
                                                          axis=1)
unique_brand = dict(feature_data.groupby("load_type")["(brand，model_number)"].unique())

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

# 聚类总标签
feature_data['cluster1-4'] = feature_data.apply(lambda row: str(my_point2(row['cluster1'], row['cluster2'], row['cluster3'], row['cluster4'])), axis=1)
data_temp = pd.DataFrame({
    "load_type": feature_data["load_type"],
    "cluster1-4": feature_data['cluster1-4']
})
unique_cluster = data_temp.groupby("load_type")["cluster1-4"].unique()

"""
转为onehot编码
"""
onehot_encoded = list()
for value in feature_data["cluster4"]:
    letter = [0 for _ in range(3)]
    letter[value] = 1
    onehot_encoded.append(letter)

"""
幅值特征分箱
"""
P_list = list(feature_data["P"])
cut = pd.qcut(P_list, 5)
P_bins = cut.codes
# (3.017, 18.951], (18.951, 41.474], (41.474, 130.388], (130.388, 572.396], (572.396, 1636.301]
"""
提取出某些型号
"""
feature_data["Is_test"] = 0
for brand in unique_brand:
    if len(unique_brand[brand]) > 1:
        a = len(unique_brand[brand])
        feature_data.loc[
            feature_data["(brand，model_number)"] == unique_brand[brand][random.randrange(2, a)], "Is_test"] = 1

# 相位预处理
hp = np.array(feature_data[["i_hp1", "i_hp2", "i_hp3", "i_hp4", "i_hp5"]])
hp_cos = np.cos(hp[:, 1] - hp[:, 0])

input_data = pd.DataFrame({
    "file_name": feature_data["file_name"],
    "P_bins": P_bins,
    "P": feature_data["P"],
    "cluster1": feature_data["cluster1"],
    "cluster2": feature_data["cluster2"],
    "cluster3": feature_data["cluster3"],
    "cluster4": feature_data["cluster4"],
    "i_hm2/i_hm1": feature_data["i_hm2/i_hm1"],
    "i_hm3/i_hm1": feature_data["i_hm3/i_hm1"],
    "i_hm4/i_hm1": feature_data["i_hm4/i_hm1"],
    "i_hm5/i_hm1": feature_data["i_hm5/i_hm1"],
    "hf_2_1": np.cos(hp[:, 1] - hp[:, 0]),
    "hf_3_1": np.cos(hp[:, 2] - hp[:, 0]),
    "hf_4_1": np.cos(hp[:, 3] - hp[:, 0]),
    "hf_5_1": np.cos(hp[:, 4] - hp[:, 0]),
    "i_thd": feature_data["i_thd"],
    "P_F": feature_data["P_F"],
    "Label": feature_data["load_type"],
    "Is_test": feature_data["Is_test"]
})
input_data.to_csv("../Input_Data_PLAID_0720.csv", index=False, sep=',')
print("XXX")
