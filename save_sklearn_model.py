#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-08-13 15:46:49
# @Author  : ${guoshengkang} (${kangguosheng1@huokeyi.com})

import os
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib
import pickle 

# 鸢尾花数据集
iris = load_iris()
keys = iris.keys() # dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names'])
X=iris.data 	# (150, 4)
y=iris.target 	# (150,)

# 波士顿房价数据集
boston=load_boston()
keys = boston.keys() # dict_keys(['data', 'target', 'feature_names', 'DESCR'])
X=boston.data 	# (506, 13)
y=boston.target # (506,)


retval = os.getcwd()
print("当前工作目录为:%s" % retval)


clf = MLPRegressor(solver='lbfgs', alpha=0.001, activation='tanh',hidden_layer_sizes=(13,5), random_state=1)
clf.fit(X,y)
X_test=X[-5:,:]
print(clf.predict(X_test))
model_root_path="C://Users//Kang//Desktop//model_management//saved_models"
model_name="sklearn_MLP_model.m"
joblib_path=os.path.join(model_root_path,model_name)
# 将训练好的模型保存到train_model.m中
joblib.dump(clf, joblib_path)
model_name="sklearn_MLP_model.pkl"
pickle_path=os.path.join(model_root_path,model_name)
# 将训练好的模型保存到train_model.pkl中
pickle.dump(clf,open(pickle_path,"wb"))
# 模型的加载
# clf = joblib.load(model_path)