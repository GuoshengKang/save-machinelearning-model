模型保存的两种方式
方式一
import pickle 
pickle.dump(model,open(model_path,"wb"))
model=pickle.load(open(model_path,"rb"))
方式二
from sklearn.externals import joblib
joblib.dump(model, model_path)
model = joblib.load(model_path)

注意事项
(1) joblib可以读pickle保存的模型,但pickle不能读取joblib保存的模型;
(2) 自定的模型(即类实例),在读取模型文件时需要先导入类的定义;
(3) 在数据库中可使用BLOB类型保存模型数据;
(4) 如果模型读取需要用到其他语言, 则需要用到预测模型标记语言(Predictive Model Markup Language,PMML).