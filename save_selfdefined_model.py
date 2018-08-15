#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-08-13 17:22:16
# @Author  : ${guoshengkang} (${kangguosheng1@huokeyi.com})

import os
import pandas as pd
import numpy as np
from numpy import *
from sklearn.externals import joblib
import pickle 
from sklearn.metrics import mean_squared_error,mean_absolute_error
from timeseries_models import *
from feature_engineering import *

# # 测试代码
# if __name__=='__main__':
index=pd.date_range('5/1/2018',periods=20,freq='d')
ts=pd.Series([1.0,2,3,4,3,6,3,7,3,5,1,2,3,4,3,6,3,7,3,5],index=index)
model=ts_WMA(ts,prediction_RMSE)
print(type(model)) # <class '__main__.ts_SMA'>
print(model.best_window)
model.fit()
print(type(model))
print(model.best_window)
y_fit=model.get_fittedvalues()
y_hat=model.predict(4)
print(y_fit)
print(y_hat)

model_root_path="C://Users//Kang//Desktop//model_management//saved_models"

model_name="selfdefined_WMA_model.m"
joblib_path=os.path.join(model_root_path,model_name)
# 将训练好的模型保存到train_model.m中
joblib.dump(model, joblib_path)

model_name="selfdefined_WMA_model.pkl"
pickle_path=os.path.join(model_root_path,model_name)
# 将训练好的模型保存到train_model.pkl中
pickle.dump(model,open(pickle_path,"wb"))

# model2 = joblib.load(pickle_path)
# # model2=pickle.load(open(pickle_path,"rb"))
# print(model2.best_window)
# y_fit=model2.get_fittedvalues()
# y_hat=model2.predict(4)
# print(y_fit)
# print(y_hat)



