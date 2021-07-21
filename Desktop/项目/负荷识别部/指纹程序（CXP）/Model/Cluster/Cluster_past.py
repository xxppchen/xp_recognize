import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.externals import joblib
import random

random.seed(12)
feature_data = pd.read_csv(r'D:/Desktop/项目/负荷识别部/Plaid/PLAID 2018/submetered/Feature_Data_0603(-3).csv')
with open('D:/Desktop/项目/负荷识别部/Plaid/PLAID 2018/submetered/metadata_submetered.json') as data_file:
    meta = json.load(data_file)
file2type = []
for i, x in meta.items():
    file2type.append(x["appliance"]["type"])

# 标准化处理
# # 多类使用的特征
# X = np.array(feature_data.iloc[:, 5:])
# 两类使用的特征
feature_data = feature_data[feature_data["P"] >= 3]  # 进行功率阈值过滤
X = np.array(feature_data[["i_thd", "i_hp1", "low_hd"]])
X[:, 0] = X[:, 0] / 3
X[:, 1] = np.cos(X[:, 1] / 180 * np.pi)
X[:, 2] = X[:, 2] * 5

"""
聚类分析
"""
model = KMeans(n_clusters=4, random_state=9)
model.fit(X)
yhat = model.predict(X)
feature_data["yhat"] = yhat
clusters = np.unique(yhat)
cluster_file = {}
cluster_type = {}
cluster_type_unique = {}
fig1, ax1 = plt.subplots()
for cluster in clusters:
    row_ix = np.where(yhat == cluster)
    plt.scatter(X[row_ix, 0], X[row_ix, 2], label=cluster)
    cluster_file['cluster_{}'.format(cluster)] = np.unique(np.array(feature_data)[row_ix, 3])
    cluster_type['cluster_{}'.format(cluster)] = np.array(file2type)[
        np.unique(np.array(feature_data)[row_ix, 3]).astype(int) - 1]
    cluster_type_unique['cluster_{}'.format(cluster)] = np.unique(np.array(feature_data)[row_ix, 0])
plt.legend()
plt.show()
# joblib.dump(model, '../cluster_model1.pkl')
# model__1 = joblib.load('./cluster_model1.pkl')

print("XXX")

# from sklearn.cluster import MeanShift
# model2 = MeanShift()
# yhat2 = model2.fit_predict(X)
# clusters2 = np.unique(yhat2)
# cluster_file2 = {}
# fig2, ax2 = plt.subplots()
# for cluster2 in clusters2:
#     row_ix2 = np.where(yhat2 == cluster2)
#     plt.scatter(X[row_ix2, 2], X[row_ix2, 1])
#     cluster_file2['cluster_{}'.format(cluster2)] = np.unique(np.array(feature_data)[row_ix2, 1])
# plt.show()

# print("XXX")

"""
聚类结果画图
"""
# for cluster_name, cluster_nums in cluster_file.items():
#     i = 0
#     for num in cluster_nums:
#         data = pd.read_csv('D:/Desktop/项目/负荷识别部/Plaid/PLAID 2018/submetered/submetered_new/' + str(num) + '.csv',
#                            names=["I", "U"])
#         data = data.iloc[-30 * 500:-20 * 500, :]
#         # ########## 作图 ##########
#         fig, ax1 = plt.subplots(figsize=(10, 6))
#         ax2 = ax1.twinx()
#         plt.title(u'Voltage,Current-Time---' + str(file2type[num - 1]))
#         T = np.arange(0, 0.16666, 0.16666 / 5000)
#         line2, = ax1.plot(data["U"], 'b')
#         line3, = ax2.plot(data["I"], 'r', linewidth=2)
#         plt.ylim(ymin=-np.max(data["I"]) * 2, ymax=np.max(data["I"]) * 2)
#         plt.legend([line3, line2], ['Current[A]', 'Voltage[V]'], loc='upper right')
#         ax1.set_xlabel(u'Time[s]')
#         ax1.set_ylabel(u'Voltage[V]')
#         ax2.set_ylabel(u'Current[A]')
#         # 分两类
#         plt.savefig("D:\Desktop\项目\负荷识别部\指纹程序（CXP）\Cluster_Result1\\" + str(cluster_name) + "\\" + str(num) + "_" + str(
#             file2type[num - 1]) + ".jpg")
#         # # 分多类
#         # plt.savefig("D:\Desktop\项目\负荷识别部\指纹程序（CXP）\Cluster_Result2\\" + str(cluster_name) + "\\" + str(num) + "_" + str(
#         #     file2type[num - 1]) + ".jpg")
#         plt.close()
#         i += 1
#         print("正在处理" + cluster_name + "类中的第" + str(i) + "条数据...")


"""
二次聚类1
"""
cluster_data = feature_data[feature_data["yhat"] == 0]
cluster_data.sort_values("file_name", inplace=True)
X2 = np.array(cluster_data[["i_hm2/i_hm1", "i_hm4/i_hm1", "i_pp_rms", "i_thd", "i_hm3/i_hm1", "i_hm5/i_hm1"]])
model2 = KMeans(n_clusters=4, random_state=9)
model2.fit(X2)
yhat2 = model2.predict(X2)
cluster_data["yhat2"] = yhat2
clusters2 = np.unique(yhat2)
cluster_file2 = {}
cluster_type2 = {}
cluster_type_unique2 = {}
fig3, ax3 = plt.subplots()
for cluster in clusters2:
    row_ix = np.where(yhat2 == cluster)
    plt.scatter(X2[row_ix, 0], X2[row_ix, 4], label=cluster)
    cluster_file2['cluster_{}'.format(cluster)] = np.unique(np.array(cluster_data)[row_ix, 1])
    cluster_type2['cluster_{}'.format(cluster)] = np.array(file2type)[
        np.unique(np.array(cluster_data)[row_ix, 1]).astype(int) - 1]
    cluster_type_unique2['cluster_{}'.format(cluster)] = np.unique(np.array(cluster_data)[row_ix, 0])
plt.legend()
plt.show()
joblib.dump(model2, './cluster_model2_1.pkl')

# for cluster_name, cluster_nums in cluster_file2.items():
#     i = 0
#     for num in cluster_nums:
#         data = pd.read_csv('D:/Desktop/项目/负荷识别部/Plaid/PLAID 2018/submetered/submetered_new/' + str(num) + '.csv',
#                            names=["I", "U"])
#         data = data.iloc[-30 * 500:-20 * 500, :]
#         # ########## 作图 ##########
#         fig, ax1 = plt.subplots(figsize=(10, 6))
#         ax2 = ax1.twinx()
#         plt.title(u'Voltage,Current-Time---' + str(file2type[num - 1]))
#         T = np.arange(0, 0.16666, 0.16666 / 5000)
#         line2, = ax1.plot(T, data["U"], 'b')
#         line3, = ax2.plot(T, data["I"], 'r', linewidth=2)
#         plt.ylim(ymin=-np.max(data["I"]) * 2, ymax=np.max(data["I"]) * 2)
#         plt.legend([line3, line2], ['Current[A]', 'Voltage[V]'], loc='upper right')
#         ax1.set_xlabel(u'Time[s]')
#         ax1.set_ylabel(u'Voltage[V]')
#         ax2.set_ylabel(u'Current[A]')
#         # 分多类
#         plt.savefig("D:\Desktop\项目\负荷识别部\指纹程序（CXP）\Cluster_Result1\cluster_0\\" + str(cluster_name) + "\\" + str(
#             num) + "_" + str(
#             file2type[num - 1]) + ".jpg")
#         plt.close()
#         i += 1
#         print("正在处理" + cluster_name + "类中的第" + str(i) + "条数据...")

"""
二次聚类3
"""
cluster_data3 = feature_data[feature_data["yhat"] == 2]
cluster_data3.sort_values("file_name", inplace=True)
X3 = np.array(cluster_data3[["i_thd", "i_hm3/i_hm1", "i_hm5/i_hm1"]])
model3 = KMeans(n_clusters=3, random_state=9)
model3.fit(X3)
yhat3 = model3.predict(X3)
cluster_data3["yhat2"] = yhat3
clusters3 = np.unique(yhat3)
cluster_file3 = {}
cluster_type3 = {}
cluster_type_unique3 = {}
fig3, ax3 = plt.subplots()
for cluster in clusters3:
    row_ix = np.where(yhat3 == cluster)
    plt.scatter(X3[row_ix, 0], X3[row_ix, 1], label=cluster)
    cluster_file3['cluster_{}'.format(cluster)] = np.unique(np.array(cluster_data3)[row_ix, 1])
    cluster_type3['cluster_{}'.format(cluster)] = np.array(file2type)[
        np.unique(np.array(cluster_data3)[row_ix, 1]).astype(int) - 1]
    cluster_type_unique3['cluster_{}'.format(cluster)] = np.unique(np.array(cluster_data3)[row_ix, 0])
plt.legend()
plt.show()
joblib.dump(model3, './cluster_model2_3.pkl')

# for cluster_name, cluster_nums in cluster_file3.items():
#     i = 0
#     for num in cluster_nums:
#         data = pd.read_csv('D:/Desktop/项目/负荷识别部/Plaid/PLAID 2018/submetered/submetered_new/' + str(num) + '.csv',
#                            names=["I", "U"])
#         data = data.iloc[-30 * 500:-20 * 500, :]
#         # ########## 作图 ##########
#         fig, ax1 = plt.subplots(figsize=(10, 6))
#         ax2 = ax1.twinx()
#         plt.title(u'Voltage,Current-Time---' + str(file2type[num - 1]))
#         T = np.arange(0, 0.16666, 0.16666 / 5000)
#         line2, = ax1.plot(data["U"], 'b')
#         line3, = ax2.plot(data["I"], 'r', linewidth=2)
#         plt.ylim(ymin=-np.max(data["I"]) * 2, ymax=np.max(data["I"]) * 2)
#         plt.legend([line3, line2], ['Current[A]', 'Voltage[V]'], loc='upper right')
#         ax1.set_xlabel(u'Time[s]')
#         ax1.set_ylabel(u'Voltage[V]')
#         ax2.set_ylabel(u'Current[A]')
#         # 分多类
#         plt.savefig("D:\Desktop\项目\负荷识别部\指纹程序（CXP）\Cluster_Result1\cluster_2\\" + str(cluster_name) + "\\" + str(
#             num) + "_" + str(
#             file2type[num - 1]) + ".jpg")
#         plt.close()
#         i += 1
#         print("正在处理" + cluster_name + "类中的第" + str(i) + "条数据...")
print("XXX")



"""
二次聚类4
"""
cluster_data4 = feature_data[feature_data["yhat"] == 3]
cluster_data4.sort_values("file_name", inplace=True)
X4 = np.array(cluster_data4[["i_thd", "i_hm3/i_hm1", "i_hm5/i_hm1"]])
# X4 = np.array(cluster_data4[["i_thd", "i_hm3/i_hm1", "i_hm5/i_hm1", "i_hp3", "i_hp5", "i_hp1"]])
# X4[:, -3] = np.cos((X4[:, -3]-X4[:, -1])/180*np.pi)
# X4[:, -2] = np.cos((X4[:, -2]-X4[:, -1])/180*np.pi)
# X4 = X4[:, :-1]
model4 = KMeans(n_clusters=3, random_state=9)
model4.fit(X4)
yhat4 = model4.predict(X4)
cluster_data4["yhat2"] = yhat4
clusters4 = np.unique(yhat4)
cluster_file4 = {}
cluster_type4 = {}
cluster_type_unique4 = {}
fig4, ax4 = plt.subplots()
for cluster in clusters4:
    row_ix = np.where(yhat4 == cluster)
    plt.scatter(X4[row_ix, 0], X4[row_ix, 1], label=cluster)
    cluster_file4['cluster_{}'.format(cluster)] = np.unique(np.array(cluster_data4)[row_ix, 1])
    cluster_type4['cluster_{}'.format(cluster)] = np.array(file2type)[
        np.unique(np.array(cluster_data4)[row_ix, 1]).astype(int) - 1]
    cluster_type_unique4['cluster_{}'.format(cluster)] = np.unique(np.array(cluster_data4)[row_ix, 0])
plt.legend()
plt.show()
joblib.dump(model4, './cluster_model2_4.pkl')

# for cluster_name, cluster_nums in cluster_file4.items():
#     i = 0
#     for num in cluster_nums:
#         data = pd.read_csv('D:/Desktop/项目/负荷识别部/Plaid/PLAID 2018/submetered/submetered_new/' + str(num) + '.csv',
#                            names=["I", "U"])
#         data = data.iloc[-30 * 500:-20 * 500, :]
#         # ########## 作图 ##########
#         fig, ax1 = plt.subplots(figsize=(10, 6))
#         ax2 = ax1.twinx()
#         plt.title(u'Voltage,Current-Time---' + str(file2type[num - 1]))
#         T = np.arange(0, 0.16666, 0.16666 / 5000)
#         line2, = ax1.plot(data["U"], 'b')
#         line3, = ax2.plot(data["I"], 'r', linewidth=2)
#         plt.ylim(ymin=-np.max(data["I"]) * 2, ymax=np.max(data["I"]) * 2)
#         plt.legend([line3, line2], ['Current[A]', 'Voltage[V]'], loc='upper right')
#         ax1.set_xlabel(u'Time[s]')
#         ax1.set_ylabel(u'Voltage[V]')
#         ax2.set_ylabel(u'Current[A]')
#         # 分多类
#         plt.savefig("D:\Desktop\项目\负荷识别部\指纹程序（CXP）\Cluster_Result1\cluster_3\\" + str(cluster_name) + "\\" + str(
#             num) + "_" + str(
#             file2type[num - 1]) + ".jpg")
#         plt.close()
#         i += 1
#         print("正在处理" + cluster_name + "类中的第" + str(i) + "条数据...")
print("XXX")