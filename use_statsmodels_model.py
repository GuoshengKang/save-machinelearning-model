#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-08-13 15:46:49
# @Author  : ${guoshengkang} (${kangguosheng1@huokeyi.com})

import os
import pandas as pd
from sklearn.externals import joblib
import pickle 
import statsmodels.api as sm
#####################################################
'''
# 测试指数平滑
index=pd.date_range('5/1/2018',periods=20,freq='d')
ts=pd.Series([1.0,2,3,4,3,6,3,7,3,5,1,2,3,4,3,6,3,7,3,5],index=index)
# model = sm.tsa.ExponentialSmoothing(ts).fit(optimized=True)
# y_hat=model.forecast(7)
# y_fit=model.predict()
model_path="C://Users//Kang//Desktop//model_management//models//statsmodels_ES_model.m"
# 模型的加载
model = joblib.load(model_path)
# model=pickle.load(open(model_path,"rb"))
y_hat=model.forecast(7)
y_fit=model.predict()
print(y_hat)
print(y_fit)
'''
############################################
# 测试指数平滑
index=pd.date_range('5/1/2018',periods=20,freq='d')
ts=pd.Series([1.0,2,3,4,3,6,3,7,3,5,1,2,3,4,3,6,3,7,3,5],index=index)
model_path="C://Users//Kang//Desktop//model_management//saved_models//statsmodels_ARMA_model.m"
# 模型的加载
model = joblib.load(model_path)
print('模型的数据类型：',type(model))
# <class 'statsmodels.tsa.arima_model.ARMAResultsWrapper'>
# model=pickle.load(open(model_path,"rb"))
y_fit=model.predict()
y_hat=model.forecast(4)[0]
print(y_fit)
print(y_hat)
