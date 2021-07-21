import numpy as np
import pandas as pd
from sklearn.externals import joblib
import random

random.seed(12)

# 聚类模型读取
model1 = joblib.load('../cluster_model1.pkl')
model2_1 = joblib.load('../cluster_model2_1.pkl')
model2_3 = joblib.load('../cluster_model2_3.pkl')
model2_4 = joblib.load('../cluster_model2_4.pkl')
# 原始数据读取，后续新增标签数据，
feature_data = pd.read_csv(r'D:/Desktop/项目/负荷识别部/Plaid/PLAID 2018/submetered/Feature_Data_0603(-3).csv')
feature_data.sort_values("file_name", inplace=True)
feature_data = feature_data[feature_data["P"] >= 3]


def my_point(a, b):
    return a, b


feature_data['(brand，model_number)'] = feature_data.apply(lambda row: str(my_point(row['brand'], row['model_number'])),
                       axis=1)
unique_brand = dict(feature_data.groupby("load_type")["(brand，model_number)"].unique())

"""
第一层聚类
0————有功主导（R）
1————电力电子（EL）
2————超低频谐波（LHD）
3————无功主导（X）
"""
# 第一层聚类标签生成
X1 = np.array(feature_data[["i_thd", "i_hp1", "low_hd"]])
X1[:, 1] = np.cos(X1[:, 1] / 180 * np.pi)
X1 = (X1 - np.mean(X1, axis=0)) / np.std(X1, axis=0)
yhat = model1.predict(X1)
feature_data["cluster1"] = yhat

"""
第二层聚类
0————啥也不是————”N“
1————偏基次波————”H1“
2————偏三次波————”H3“
3————偏五次波————”H5“
4————不对称类————”NS“
5————偏电子类————”E“
"""


# ##########处理函数#############
def get_type(yhat, pretype):
    """
    yhat:数据原始的标签
    pretype:数据原始的各标签对应类型
    """
    type2code = {"N": 0, "H1": 1, "H3": 2, "H5": 3, "NS": 4, "E": 5}
    return type2code[pretype[int(yhat)]]


feature_data["cluster2"] = feature_data.shape[0] * [-1]
# 类别1
cluster_data1 = feature_data[feature_data["cluster1"] == 0]
X2_1 = np.array(cluster_data1[["i_hm2/i_hm1", "i_hm4/i_hm1", "i_pp_rms", "i_thd", "i_hm3/i_hm1", "i_hm5/i_hm1"]])
X2_1 = (X2_1 - np.mean(X2_1, axis=0)) / np.std(X2_1, axis=0)
yhat2 = model2_1.predict(X2_1)
cluster_data1["yhat"] = yhat2
cluster_data1["cluster2"] = cluster_data1.apply(lambda x: get_type(x["yhat"], ["H3", "H5", "NS", "H1"]), axis=1)
# 类别2
cluster_data2 = feature_data[feature_data["cluster1"] == 1]
cluster_data2["cluster2"] = cluster_data2.shape[0] * [0]
# 类别3
cluster_data3 = feature_data[feature_data["cluster1"] == 2]
X2_3 = np.array(cluster_data3[["i_thd", "i_hm3/i_hm1", "i_hm5/i_hm1"]])
X2_3 = (X2_3 - np.mean(X2_3, axis=0)) / np.std(X2_3, axis=0)
yhat3 = model2_3.predict(X2_3)
cluster_data3["yhat"] = yhat3
cluster_data3["cluster2"] = cluster_data3.apply(lambda x: get_type(x["yhat"], ["H5", "E", "H1"]), axis=1)
# 类别4
cluster_data4 = feature_data[feature_data["cluster1"] == 3]
X2_4 = np.array(cluster_data4[["i_thd", "i_hm3/i_hm1", "i_hm5/i_hm1"]])
X2_4 = (X2_4 - np.mean(X2_4, axis=0)) / np.std(X2_4, axis=0)
yhat4 = model2_4.predict(X2_4)
cluster_data4["yhat"] = yhat4
cluster_data4["cluster2"] = cluster_data4.apply(lambda x: get_type(x["yhat"], ["H3", "E", "H5"]), axis=1)
# 聚类2总标签
cluster2_result = pd.concat(
    [cluster_data1["cluster2"], cluster_data2["cluster2"], cluster_data3["cluster2"], cluster_data4["cluster2"]],
    axis=0)
feature_data["cluster2"] = cluster2_result

feature_data['(cluster1，cluster2)'] = feature_data.apply(lambda row: str(my_point(row['cluster1'], row['cluster2']))
                                                         , axis=1)
data_temp = pd.DataFrame({
    "load_type": feature_data["load_type"],
    "(cluster1，cluster2)": feature_data['(cluster1，cluster2)']
})
unique_cluster = data_temp.groupby("load_type")["(cluster1，cluster2)"].unique()

"""
转为onehot编码
"""
onehot_encoded1 = list()
for value in feature_data["cluster1"]:
    letter = [0 for _ in range(4)]
    letter[value] = 1
    onehot_encoded1.append(letter)
onehot_encoded2 = list()
for value in feature_data["cluster2"]:
    letter = [0 for _ in range(5)]
    if value != 0:
        letter[value - 1] = 1
    onehot_encoded2.append(letter)

"""
幅值特征分箱
"""
P_list = list(feature_data["P"])
cut = pd.qcut(P_list, 5)
P_bins = cut.codes

"""
提取出某些型号
"""
feature_data["Is_test"] = 0
for brand in unique_brand:
    if len(unique_brand[brand]) > 1:
        a = len(unique_brand[brand])
        feature_data.loc[feature_data["(brand，model_number)"] == unique_brand[brand][random.randrange(2, a)], "Is_test"] = 1

input_data = pd.DataFrame({
    "file_name": feature_data["file_name"],
    "P_bins": P_bins,
    "P": feature_data["P"],
    "cluster1": feature_data["cluster1"],
    "cluster2": feature_data["cluster2"],
    "is_R": np.array(onehot_encoded1)[:, 0],
    "is_EL": np.array(onehot_encoded1)[:, 1],
    "is_LHD": np.array(onehot_encoded1)[:, 2],
    "is_X": np.array(onehot_encoded1)[:, 3],
    "is_H1": np.array(onehot_encoded2)[:, 0],
    "is_H3": np.array(onehot_encoded2)[:, 1],
    "is_H5": np.array(onehot_encoded2)[:, 2],
    "is_NS": np.array(onehot_encoded2)[:, 3],
    "is_E": np.array(onehot_encoded2)[:, 4],
    "i_hm2/i_hm1": feature_data["i_hm2/i_hm1"],
    "i_hm3/i_hm1": feature_data["i_hm3/i_hm1"],
    "i_hm4/i_hm1": feature_data["i_hm4/i_hm1"],
    "i_hm5/i_hm1": feature_data["i_hm5/i_hm1"],
    "i_thd": feature_data["i_thd"],
    "P_F": feature_data["P_F"],
    "Label": feature_data["load_type"],
    "Is_test": feature_data["Is_test"]
})
input_data.to_csv("../Input_Data.csv", index=False, sep=',')
print("XXX")
