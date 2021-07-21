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
# 准备数据
source_data = pd.read_csv("Input_Data.csv")
X = source_data.loc[:, [
                           # "P_bins",
                           "P",
                           # "cluster1", "cluster2",
                           # "is_R", "is_EL", "is_LHD", "is_X", "is_H1", "is_H3", "is_H5", "is_NS", "is_H5",
                           # "i_thd",
                           # "P_F",
                           "i_hm2/i_hm1", "i_hm3/i_hm1", "i_hm4/i_hm1", "i_hm5/i_hm1"
                       ]]
Y = source_data["Label"]
le = preprocessing.LabelEncoder()
le.fit_transform(Y)
Y = le.transform(Y)
# 训练集划分
train_indices = source_data["Is_test"] == 0  # 训练集的索引
test_indices = source_data["Is_test"] == 1  # 验证集的索引
Train_X = X[train_indices]
Train_Y = Y[train_indices]
Test_X = X[test_indices]
Test_Y = Y[test_indices]

Train_X = pd.DataFrame(Train_X)
Train_Y = pd.DataFrame(Train_Y)
Test_X = pd.DataFrame(Test_X)
Test_Y = pd.DataFrame(Test_Y)

rf0 = RandomForestClassifier(oob_score=True, random_state=10)
rf0.fit(Train_X, Train_Y.values.ravel())
print(rf0.oob_score_)
print(rf0.score(Test_X, Test_Y))

best_score = 0.0
# prep_onehot--best--7,2  train:0.937107 test:0.906977
# prep-not onehot -- 7,2  train:0.915380 test:0.941860
# no prep         --10,2  train:0.942253 test:0.906977
# no P_bin no prep-- 7,2  train:0.973462 test:0.711864
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
        train_score = estimator.score(Train_X, Train_Y)
        test_score = estimator.score(Test_X, Test_Y)
        score = (train_score + test_score) / 2 - 0.2 * abs(train_score - test_score)
        if score > best_score:
            best_score = score
            best_n_estimator = n_estimator
            best_depth = depth_
        print("n_estimator: %02d -- depth: %02d ---- train_score: %f -- test_score: %f"
              % (n_estimator, depth_, train_score, test_score))
print("best_score: n_estimator:%02d  depth:%02d  score: %f" % (best_n_estimator, best_depth, best_score))
gbc0 = RandomForestClassifier(n_estimators=best_n_estimator,
                              min_samples_split=min_samples_split,
                              min_samples_leaf=min_samples_leaf,
                              max_depth=best_depth,
                              max_features='sqrt',
                              random_state=10)
gbc0.fit(Train_X, Train_Y.values.ravel())
print("Train data best_score -- %f" % gbc0.score(Train_X, Train_Y.values.ravel()))
print("Test data score ------- %f" % gbc0.score(Test_X, Test_Y.values.ravel()))


def test_plot(X_data, Y_data):
    out_put = gbc0.predict(X_data)
    cm_plot(np.array(Y_data), np.array(out_put))


test_plot(Test_X, Test_Y)
test_plot(Train_X, Train_Y)

# y_trainval_pred = gbc0.predict(Train_X)
# y_trainval_predprob = gbc0.predict_proba(Train_X)
# Label_fit = LabelBinarizer()
# y_trainval_one_hot = Label_fit.fit_transform(Train_Y)
# y_test_pred = gbc0.predict(Test_X)
# y_test_predprob = gbc0.predict_proba(Test_X)
# y_test_one_hot = Label_fit.transform(Test_Y)
# print("train best score: %f" % best_score)
# print("Accuracy : %.4g" % metrics.accuracy_score(Test_Y, y_test_pred))
# print("AUC Score (train): %f" %
#       metrics.roc_auc_score(y_trainval_one_hot, y_trainval_predprob, average='micro'))
# print("AUC Score (test): %f" %
#       metrics.roc_auc_score(y_test_one_hot, y_test_predprob, average='micro'))
# print('best_n_estimator: %02d' % best_n_estimator)

print("XXX")
