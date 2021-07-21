from Model.linear_model import *
import numpy as np
import pandas as pd
from sklearn import preprocessing
from plot.CM_Polt import *
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from plot.CM_Polt import *

torch.manual_seed(7)
"""
聚类1 Low_hd  -- 0:稳定型    1:时变型
聚类2 i_hp1   -- 0:对齐型    1:偏移型
聚类3 偶含量   -- 0:对称型    1:不对称型
聚类4 奇含量   -- 0:少毛刺型   1:电子型     2:多毛刺型
"""
# 准备数据
source_data1 = pd.DataFrame()
source_data1 = pd.read_csv("../../Model/Input_Data_PLAID_0720.csv", encoding="utf-8")
source_data1.loc[source_data1["Label"] == "Air Conditioner", "Label"] = "AC"
source_data1.loc[source_data1["Label"] == "Coffee maker", "Label"] = "CoffeeMachine"
source_data1.loc[source_data1["Label"] == "Hairdryer", "Label"] = "HairDryer"
source_data1.loc[source_data1["Label"] == "Soldering Iron", "Label"] = "SolderingIron"
source_data1.loc[source_data1["Label"] == "Vacuum", "Label"] = "VacuumCleaner"
source_data1.loc[source_data1["Label"] == "Washing Machine", "Label"] = "WashingMachine"
source_data1.loc[source_data1["Label"] == "Water kettle", "Label"] = "WaterHeater"
source_data2 = pd.read_csv("../WHITED/Input_Data0719.csv")
source_data2.loc[source_data2["Label"] == "Hairdryer", "Label"] = "HairDryer"
source_data = pd.concat([source_data1, source_data2], axis=0)
dataset_app_unique = np.unique(np.array(source_data["Label"]))
X = source_data.loc[:, [
                           "P_bins",
                           # "P",
                           "cluster1", "cluster2", "cluster3", "cluster4",
                           "i_hm2/i_hm1", "i_hm3/i_hm1", "i_hm4/i_hm1", "i_hm5/i_hm1",
                           "hf_2_1", "hf_3_1", "hf_4_1", "hf_5_1",
                           # "i_thd",
                           # "P_F",
                       ]]
Y = source_data["Label"]
le = preprocessing.LabelEncoder()
le.fit_transform(Y)
Y = le.transform(Y)

source_data_self = pd.read_csv("../Hioki/Input_Data0720.csv", encoding="utf-8")
hioki_app_unique = np.unique(np.array(source_data_self["Label"]))
X_self = source_data_self.loc[:, [
                                     "P_bins",
                                     # "P",
                                     "cluster1", "cluster2", "cluster3", "cluster4",
                                     "i_hm2/i_hm1", "i_hm3/i_hm1", "i_hm4/i_hm1", "i_hm5/i_hm1",
                                     "hf_2_1", "hf_3_1", "hf_4_1", "hf_5_1",
                                     # "i_thd",
                                     # "P_F",
                                 ]]

# ['AC' 'AirPump' 'BenchGrinder' 'Blender' 'CFL' 'CableModem', 'CableReceiver' 'Charger' 'CoffeeMachine' 'Compact
# Fluorescent Lamp', 'DeepFryer' 'DesktopPC' 'Desoldering' 'DrillingMachine' 'Fan' 'FanHeater', 'FlatIron' 'Fridge'
# 'GameConsole' 'GuitarAmp' 'HIFI' 'Hair Iron', 'HairDryer' 'Halogen' 'HalogenFluter' 'Heater' 'Incandescent Light
# Bulb', 'Iron' 'JigSaw' 'JuiceMaker' 'Kettle' 'KitchenHood' 'LEDLight' 'Laptop', 'LaserPrinter' 'LightBulb'
# 'Massage' 'Microwave' 'Mixer' 'Monitor', 'MosquitoRepellent' 'MultiTool' 'PowerSupply' 'Projector' 'RiceCooker',
# 'SandwichMaker' 'SewingMachine' 'ShoeWarmer' 'Shredder' 'SolderingIron', 'Stove' 'TV' 'Toaster' 'Treadmill'
# 'VacuumCleaner' 'WashingMachine', 'WaterHeater' 'WaterPump']
# ['AC' '气泵' '台式研磨机' '搅拌机' 'CFL' '电缆调制解调器',
# '电缆接收器' '充电器' '咖啡机' '紧凑型荧光灯', '油炸锅' '台式电脑' '拆焊机''钻孔机''风扇''风扇加热器' '扁铁''冰箱''游戏机''吉他放大器''HIFI''电熨斗'
# '吹风机''卤素''卤素灯''加热器' '白炽灯泡' '熨斗' '曲线锯' '果汁机' '水壶' '厨房油烟机' 'LED 灯' '笔记本电脑' '激光打印机' '灯泡' '按摩' '微波炉' '搅拌机' '显示器' '驱蚊剂'
# '多功能工具' '电源' '投影仪' '电饭煲' '三明治机' '缝纫机' '暖鞋器' '碎纸机' '烙铁' '炉子' '电视机' '烤面包机''跑步机''吸尘器''洗衣机' '热水器''水泵']

# 训练集划分
Train_X = pd.DataFrame(X)
Train_Y = pd.DataFrame(Y)

rf0 = RandomForestClassifier(oob_score=True, random_state=10)
rf0.fit(Train_X, Train_Y.values.ravel())
print(rf0.oob_score_)

best_score = 0.0
min_samples_split = 10
min_samples_leaf = 2
for n_estimator in range(10, 26, 2):
    for depth_ in range(6, 14, 2):
        estimator = RandomForestClassifier(n_estimators=n_estimator,
                                           min_samples_split=min_samples_split,
                                           min_samples_leaf=min_samples_leaf,
                                           max_depth=depth_,
                                           max_features='sqrt',
                                           random_state=10)
        estimator.fit(Train_X, Train_Y.values.ravel())
        score = estimator.score(Train_X, Train_Y)
        if score > best_score:
            best_score = score
            best_n_estimator = n_estimator
            best_depth = depth_
        print("n_estimator: %02d -- depth: %02d ---- train_score: %f "
              % (n_estimator, depth_, score))
print("best_score: n_estimator:%02d  depth:%02d  score: %f" % (best_n_estimator, best_depth, best_score))
gbc0 = RandomForestClassifier(n_estimators=best_n_estimator,
                              min_samples_split=min_samples_split,
                              min_samples_leaf=min_samples_leaf,
                              max_depth=best_depth,
                              max_features='sqrt',
                              random_state=10)
gbc0.fit(Train_X, Train_Y.values.ravel())
print("Train data best_score -- %f" % gbc0.score(Train_X, Train_Y.values.ravel()))


def result_plot(X_data, Y_data):
    out_put = gbc0.predict(X_data)
    cm_plot(np.array(Y_data), np.array(out_put))


result_plot(Train_X, Train_Y)

pred_Y = gbc0.predict(X_self)
source_data_self["pred"] = le.inverse_transform(pred_Y)
print("XXX")
