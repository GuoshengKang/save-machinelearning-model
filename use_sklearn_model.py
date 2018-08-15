#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-08-13 16:12:07
# @Author  : ${guoshengkang} (${kangguosheng1@huokeyi.com})

import os
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
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

X_test=X[-5:,:]
print(y[-5:])
model_path="C://Users//Kang//Desktop//model_management//saved_models//sklearn_MLP_model.pkl"
# 模型的加载
# clf = joblib.load(model_path)
clf=pickle.load(open(model_path,"rb"))
print('模型的数据类型：',type(clf))
#  <class 'sklearn.neural_network.multilayer_perceptron.MLPRegressor'>
predited_y=clf.predict(X_test)
print(predited_y)
