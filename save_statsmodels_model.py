#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-08-13 15:46:49
# @Author  : ${guoshengkang} (${kangguosheng1@huokeyi.com})

import os
import pandas as pd
from sklearn.externals import joblib
import pickle 
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA
########################################################################
'''
# 测试指数平滑
index=pd.date_range('5/1/2018',periods=20,freq='d')
ts=pd.Series([1.0,2,3,4,3,6,3,7,3,5,1,2,3,4,3,6,3,7,3,5],index=index)
model = sm.tsa.ExponentialSmoothing(ts).fit(optimized=True)
y_hat=model.forecast(7)
y_fit=model.predict()
print(y_hat)
print(y_fit)

model_root_path="C://Users//Kang//Desktop//model_management//saved_models"

model_name="statsmodels_ES_model.m"
joblib_path=os.path.join(model_root_path,model_name)
# 将训练好的模型保存到train_model.m中
joblib.dump(model, joblib_path)

model_name="statsmodels_ES_model.pkl"
pickle_path=os.path.join(model_root_path,model_name)
# 将训练好的模型保存到train_model.pkl中
pickle.dump(model,open(pickle_path,"wb"))
'''
############################################################################
# 测试ARIMA
index=pd.date_range('5/1/2018',periods=20,freq='d')
ts=pd.Series([1.0,2,3,4,3,6,3,7,3,5,1,2,3,4,3,6,3,7,3,5],index=index)
model = ARMA(ts, order=(2,1)) 
model = model.fit(disp=-1, method='css')
y_fit=model.predict()
y_hat=model.forecast(4)[0] 
print(y_fit)
print(y_hat)


model_root_path="C://Users//Kang//Desktop//model_management//saved_models"

model_name="statsmodels_ARMA_model.m"
joblib_path=os.path.join(model_root_path,model_name)
# 将训练好的模型保存到train_model.m中
joblib.dump(model, joblib_path)

model_name="statsmodels_ARMA_model.pkl"
pickle_path=os.path.join(model_root_path,model_name)
# 将训练好的模型保存到train_model.pkl中
pickle.dump(model,open(pickle_path,"wb"))