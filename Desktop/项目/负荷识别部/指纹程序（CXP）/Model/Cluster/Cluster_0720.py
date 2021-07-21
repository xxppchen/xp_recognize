#!/usr/bin/env python
#--coding=utf-8--
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.externals import joblib
import random
from sklearn.cluster import MeanShift


random.seed(12)
feature_data = pd.read_csv(r'D:/Desktop/项目/负荷识别部/Plaid/PLAID 2018/submetered/Feature_Data_0603(-3).csv')
with open(r'D:/Desktop/项目/负荷识别部/Plaid/PLAID 2018/submetered/metadata_submetered.json') as data_file:
    meta = json.load(data_file)
file2type = []
for i, x in meta.items():
    file2type.append(x["appliance"]["type"])

feature_data = feature_data[feature_data["P"] >= 3]  # 进行功率阈值过滤

"""
聚类1---Low_hd--稳定型/时变型
"""
# X1 = np.array(feature_data["low_hd"]).reshape(-1, 1)
# # model1 = MeanShift()
# model1 = KMeans(n_clusters=2, random_state=9)
# model1.fit(X1)
# yhat1 = model1.predict(X1)
# feature_data["yhat1"] = yhat1
# clusters1 = np.unique(yhat1)
# cluster_file1 = {}
# fig1, ax1 = plt.subplots()
# for cluster1 in clusters1:
#     row_ix = np.where(yhat1 == cluster1)
#     plt.scatter(X1[row_ix, 0], (X1[row_ix, 0]).shape[1]*[0], label=cluster1)
#     cluster_file1['cluster_{}'.format(cluster1)] = np.unique(np.array(feature_data)[row_ix, 3])
# plt.legend()
# plt.show()
# # joblib.dump(model1, '../cluster_model1.pkl')
# # model_1 = joblib.load('./cluster_model1.pkl')
#
# print("XXX")

"""
聚类结果画图
"""
# for cluster_name, cluster_nums in cluster_file1.items():
#     i = 0
#     # if cluster_name == "cluster_0":
#     #     continue
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
#         plt.savefig("D:\Desktop\项目\负荷识别部\指纹程序（CXP）\Cluster_Result_PLAID\cluster1\\" + str(cluster_name) + "\\" + str(num) + "_" + str(
#             file2type[num - 1]) + ".jpg")
#         plt.close()
#         i += 1
#         print("正在处理" + cluster_name + "类中的第" + str(i) + "条数据...")


"""
聚类2---i_hp1--对齐型/偏移型
"""
# X2 = np.array(feature_data["i_hp1"]).reshape(-1, 1)
# X2 =  np.cos(X2 / 180 * np.pi)
# model2 = KMeans(n_clusters=2, random_state=9)
# model2.fit(X2)
# yhat2 = model2.predict(X2)
# feature_data["yhat2"] = yhat2
# clusters2 = np.unique(yhat2)
# cluster_file2 = {}
# fig2, ax2 = plt.subplots()
# for cluster2 in clusters2:
#     row_ix = np.where(yhat2 == cluster2)
#     plt.scatter(X2[row_ix, 0], (X2[row_ix, 0]).shape[1]*[0], label=cluster2)
#     cluster_file2['cluster_{}'.format(cluster2)] = np.unique(np.array(feature_data)[row_ix, 3])
# plt.legend()
# plt.show()
# joblib.dump(model2, '../cluster_model2.pkl')
# # model_2 = joblib.load('./cluster_model2.pkl')
#
# print("XXX")
#
# """
# 聚类结果画图
# """
# for cluster_name, cluster_nums in cluster_file2.items():
#     i = 0
#     if cluster_name == "cluster_0":
#         continue
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
#         plt.savefig("D:\Desktop\项目\负荷识别部\指纹程序（CXP）\Cluster_Result_PLAID\cluster2\\" + str(cluster_name) + "\\" + str(num) + "_" + str(
#             file2type[num - 1]) + ".jpg")
#         plt.close()
#         i += 1
#         print("正在处理" + cluster_name + "类中的第" + str(i) + "条数据...")


"""
聚类3---偶次谐波含量--是否对称
"""
# X3_ = np.array(feature_data[["i_hm2/i_hm1", "i_hm4/i_hm1", "i_hm6/i_hm1"]])
# X3 = np.sqrt(np.sum(np.square(X3_), axis=1)).reshape(-1, 1)
# model3 = KMeans(n_clusters=2, random_state=9)
# model3.fit(X3)
# yhat3 = model3.predict(X3)
# feature_data["yhat3"] = yhat3
# clusters3 = np.unique(yhat3)
# cluster_file3 = {}
# fig3, ax3 = plt.subplots()
# for cluster3 in clusters3:
#     row_ix = np.where(yhat3 == cluster3)
#     plt.scatter(X3[row_ix, 0], (X3[row_ix, 0]).shape[1]*[0], label=cluster3)
#     cluster_file3['cluster_{}'.format(cluster3)] = np.unique(np.array(feature_data)[row_ix, 3])
# plt.legend()
# plt.show()
# joblib.dump(model3, '../cluster_model3.pkl')
# # model_3 = joblib.load('./cluster_model3.pkl')
#
# print("XXX")
#
# """
# 聚类结果画图
# """
# for cluster_name, cluster_nums in cluster_file3.items():
#     i = 0
#     if cluster_name == "cluster_0":
#         continue
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
#         plt.savefig("D:\Desktop\项目\负荷识别部\指纹程序（CXP）\Cluster_Result_PLAID\cluster3\\" + str(cluster_name) + "\\" + str(num) + "_" + str(
#             file2type[num - 1]) + ".jpg")
#         plt.close()
#         i += 1
#         print("正在处理" + cluster_name + "类中的第" + str(i) + "条数据...")
#
# print("XXX")


"""
聚类4---奇次谐波含量--是否较多波形毛刺
"""
X4_ = np.array(feature_data[["i_hm3/i_hm1", "i_hm5/i_hm1", "i_hm7/i_hm1"]])
X4 = np.sqrt(np.sum(np.square(X4_), axis=1)).reshape(-1, 1)
model4 = KMeans(n_clusters=3, random_state=9)
model4.fit(X4)
yhat4 = model4.predict(X4)
feature_data["yhat4"] = yhat4
clusters4 = np.unique(yhat4)
cluster_file4 = {}
fig4, ax4 = plt.subplots()
for cluster4 in clusters4:
    row_ix = np.where(yhat4 == cluster4)
    plt.scatter(X4[row_ix, 0], (X4[row_ix, 0]).shape[1]*[0], label=cluster4)
    cluster_file4['cluster_{}'.format(cluster4)] = np.unique(np.array(feature_data)[row_ix, 3])
plt.legend()
plt.show()
joblib.dump(model4, '../cluster_model4.pkl')
# model_4 = joblib.load('./cluster_model4.pkl')


"""
聚类结果画图
"""
for cluster_name, cluster_nums in cluster_file4.items():
    i = 0
    if cluster_name == "cluster_0":
        continue
    for num in cluster_nums:
        data = pd.read_csv('D:/Desktop/项目/负荷识别部/Plaid/PLAID 2018/submetered/submetered_new/' + str(num) + '.csv',
                           names=["I", "U"])
        data = data.iloc[-30 * 500:-20 * 500, :]
        # ########## 作图 ##########
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        plt.title(u'Voltage,Current-Time---' + str(file2type[num - 1]))
        T = np.arange(0, 0.16666, 0.16666 / 5000)
        line2, = ax1.plot(data["U"], 'b')
        line3, = ax2.plot(data["I"], 'r', linewidth=2)
        plt.ylim(ymin=-np.max(data["I"]) * 2, ymax=np.max(data["I"]) * 2)
        plt.legend([line3, line2], ['Current[A]', 'Voltage[V]'], loc='upper right')
        ax1.set_xlabel(u'Time[s]')
        ax1.set_ylabel(u'Voltage[V]')
        ax2.set_ylabel(u'Current[A]')
        # 分两类
        plt.savefig("D:\Desktop\项目\负荷识别部\指纹程序（CXP）\Cluster_Result_PLAID\cluster4\\" + str(cluster_name) + "\\" + str(num) + "_" + str(
            file2type[num - 1]) + ".jpg")
        plt.close()
        i += 1
        print("正在处理" + cluster_name + "类中的第" + str(i) + "条数据...")

print("XXX")