import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from datetime import date

####新闻分析###

data=pd.read_csv('test/Combined_News_DJIA.csv')

data["combined_news"]=data.filter(regex=("Top.*")).apply(lambda x:''.join(str(x.values)),axis=1)
#正则表达式，取出所有top几的新闻 得到新的列叫combined news

#data["Date"]=pd.to_datetime(data["Date"])
#分割测试集和训练集
train=data[data['Date'] <'2015-01-01']
test=data[data['Date'] >'2014-12-31']

feature_extraction=TfidfVectorizer()
#将原始文档转换为TF-IDF特征值
X_train=feature_extraction.fit_transform(train["combined_news"].values)
X_test=feature_extraction.transform(test["combined_news"].values)
#fit从数据集中学习它需要的参数，transform按照学到
#的参数转化数据集，换句话说，只有fit学到的参数不会改变，别的都有可能
#测试集自然不用fit，直接转换

y_train=train["Label"].values
y_test=test["Label"].values
#.values就是变成numpy数组

clf=SVC(probability=True,kernel='rbf')

clf.fit(X_train,y_train)
#前一个参数是特征，后一个参数是对应标签
#学好了以后就可以拿来预测测试集的对应标签
predictions=clf.predict_proba(X_test)

print(predictions)

print('ROC-AUC yields'+str(roc_auc_score(y_test,predictions[:,1])))