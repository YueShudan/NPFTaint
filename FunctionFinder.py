# 相比于RF2，不做数据归一化处理，发现生成的结果更好。
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
from joblib import load
import os
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer
# 读取数据集

print("当前工作目录:", os.getcwd())
positive_data = pd.read_csv('TaintSource/functionsample/Positive1.csv')
negative_data = pd.read_csv('TaintSource/functionsample/Negative1.csv')

# 添加标签
positive_data['label'] = 1
negative_data['label'] = 0

# 合并数据集
data = pd.concat([positive_data, negative_data], ignore_index=True)

# 特征和标签分离(含有程序特征)
X = data.drop(columns=['label', 'src', 'fname', 'funcaddr']) #删除标签列
# X = data.drop(columns=['label', 'src', 'fname', 'keyimportapiinfo', 'netkeywords' , 'funcaddr', 'n_memcmp', 'n_net', 'n_API', 'out_degree']) #删除标签列
y = data['label']

# 检查数据类型并转换为数值类型
for column in X.columns:   # 此时是包含data中的所有列
    if X[column].dtype == 'object':  # 检查当前列的数据类型是否为object，代指字符串或类别数据。
        X[column] = pd.to_numeric(X[column], errors='coerce')  # 尝试将该列转换为数值类型。
# 处理缺失值，填充为0或者其他策略
X.fillna(0, inplace=True)

# 数据标准化
# scaler = StandardScaler()
# X_selected = scaler.fit_transform(X)

# 定义训练集和验证集的比例（例如，80%训练，20%验证）
train_size = 0.8
# 划分数据集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1 - train_size, random_state=42)


# 训练
RF = RandomForestClassifier(n_estimators=100, random_state=42)
RF.fit(X_train, y_train)

# 输出参数
best_params = RF.get_params()
print(f"Best parameters for Random Forest: {best_params}")

# 定义保存模型的目录路径
save_directory = 'F:/gadgets/TaintSource/res'  # 替换为实际的目录路径
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# 保存模型
model_filename = os.path.join(save_directory, 'RF3_1.joblib')
joblib.dump(RF, model_filename)

# 使用训练好的模型对测试数据集 X_test 进行预测，得到预测结果 y_pred
y_pred = RF.predict(X_val)

# 计算并保存 F1 分数
stacking_pre = precision_score(y_val, y_pred, average='macro')
stacking_recall = recall_score(y_val, y_pred, average='macro')
stacking_f1 = f1_score(y_val, y_pred, average= 'macro')

print("pre", stacking_pre, "recall", stacking_recall, "f1", stacking_f1)
# pre 0.9497626582278481 recall 0.9101731601731602 f1 0.9283559577677224

# 可视化最佳模型的ROC曲线
y_prob_best = RF.predict_proba(X_val)[:, 1]
fpr_best, tpr_best, _ = roc_curve(y_val, y_prob_best)
roc_auc_best = auc(fpr_best, tpr_best)

plt.figure(figsize=(10, 6))
plt.plot(fpr_best, tpr_best, label=f'Best Model (AUC = {roc_auc_best:.2f})', color='blue')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Best Model')
plt.legend()
plt.show()


# Best parameters for Random Forest: {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None,
# 'criterion': 'gini', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None,
# 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2,
# 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': None, 'oob_score': False, 'random_state': 42,
# 'verbose': 0, 'warm_start': False}
# pre 0.9497626582278481 recall 0.9101731601731602 f1 0.9283559577677224