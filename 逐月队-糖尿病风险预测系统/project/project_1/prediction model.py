# -*- coding: utf-8 -*-
import sys
import os
if os.name == 'nt':  # Windows 系统
    os.environ['PYTHONLEGACYWINDOWSSTDIO'] = '1'  # 允许使用 Unicode 字符

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# 读取数据集
data = pd.read_csv('data-01.csv')

# 分离特征和标签
X = data.drop('Outcome', axis=1)  # 特征
y = data['Outcome']  # 标签

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 特征缩放
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")

# 对训练集进行 SMOTE 过采样以处理类别不平衡
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 使用随机森林分类器
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

joblib.dump(rf_model, 'random_forest_model.pkl')

# 在测试集上做出预测
y_pred_rf = rf_model.predict(X_test)

# 输出随机森林分类的报告和混淆矩阵
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

