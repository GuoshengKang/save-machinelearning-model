#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-08-13 15:46:49
# @Author  : ${guoshengkang} (${kangguosheng1@huokeyi.com})

import os
import pandas as pd
import numpy as np
from numpy import *
from sklearn.externals import joblib
import pickle 
from sklearn.metrics import mean_squared_error,mean_absolute_error
import statsmodels.api as sm
# 
from timeseries_models import *


# if __name__=='__main__':
# 测试指数平滑
index=pd.date_range('5/1/2018',periods=20,freq='d')
ts=pd.Series([1.0,2,3,4,3,6,3,7,3,5,1,2,3,4,3,6,3,7,3,5],index=index)

model_path="C://Users//Kang//Desktop//model_management//saved_models//selfdefined_WMA_model.pkl"
# 模型的加载
# model = joblib.load(model_path)
model=pickle.load(open(model_path,"rb"))
print(model.best_window)
y_fit=model.get_fittedvalues()
y_hat=model.predict(4)
print(y_fit)
print(y_hat)