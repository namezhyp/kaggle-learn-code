import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from datetime import date

'''import nltk
stopwords=nltk.download('stopwords')'''
from nltk.corpus import stopwords  

import re
from nltk.stem import WordNetLemmatizer
#词形还原库

####新闻分析升级版####

data=pd.read_csv('test/Combined_News_DJIA.csv')

data["combined_news"]=data.filter(regex=("Top.*")).apply(lambda x:''.join(str(x.values)),axis=1)
#正则表达式，取出所有top几的新闻 得到新的列叫combined news

#data["Date"]=pd.to_datetime(data["Date"])
#分割测试集和训练集
train=data[data['Date'] <'2015-01-01']
test=data[data['Date'] >'2014-12-31']

#####优化预处理部分####
X_train=train["combined_news"].str.lower().str.replace('"','').str.replace("'",'').str.split()
X_test=test["combined_news"].str.lower().str.replace('"','').str.replace("'",'').str.split()
#预处理文本，小写化、去除无关符号

stop =stopwords.words('english')
#获取英文停用词列表

def hasNumbers(input):
    return bool(re.search(r'\d',input))
#检查输入是否有数字，有则返回true

WordNet_Lemmatizer=WordNetLemmatizer()
#词语形状还原器  better->good

def check(word):
    if word in stop:
        return False
    elif hasNumbers(word):
        return False
    else:
        return True
##########

X_train=X_train.apply(lambda x:[WordNet_Lemmatizer.lemmatize(item) for item in x if check(item)])
X_test=X_test.apply(lambda x:[WordNet_Lemmatizer.lemmatize(item) for item in x if check(item)])
#lambda表达式 也就是无名函数，后面跟的是参数，用逗号隔开
#对x_test的每个元素放入函数，先检查是否check，如果是true，那就进行词形转化
X_train =X_train.apply(lambda x:' '.join(x))
X_test=X_test.apply(lambda x:' '.join(x))
#所有词用空格连接起来 因为sklearn只支持string输入


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
#返回的是一个n行2列的数组，两列分别对应结果0和结果1的概率，第二个一般被认为是正例
print(predictions)

print('ROC-AUC yields'+str(roc_auc_score(y_test,predictions[:,1])))
#roc_auc_score (y_true,y_preditc) 用于计算性能  只取第二列是因为只需要关注正例
#0.539  反而更差了