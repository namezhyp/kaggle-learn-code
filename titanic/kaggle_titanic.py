import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.impute import SimpleImputer
#用在AgeImputer类

from sklearn.ensemble import RandomForestClassifier #随机森林，不强制可以自己换
from sklearn.model_selection import GridSearchCV  #穷举找出最优超参数并评估各种组合下性能
#将训练数据划分为多个子集，分别做验证集，检查超参数性能


from sklearn.preprocessing import OneHotEncoder
#用在FeatureEncoder类

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


#from sklearn import svm  #这句是为了测试SVM本来不需要

####kaggle入门比赛 泰坦尼克号幸存者预测

class AgeImputer(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        imputer = SimpleImputer(strategy="mean")
        X['Age']= imputer.fit_transform(X[['Age']])
        return X
#处理数据缺失 用均值补充 填好后返回
#如果两个函数都调用，那就是fit_transfrom

class FeatureEncoder(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        encoder=OneHotEncoder()

        matrix=encoder.fit_transform(X[["Embarked"]]).toarray()
        column_names=["C","S","Q","N"]
        for i in range(len(matrix.T)):
            X[column_names[i]]=matrix.T[i]
        
        matrix=encoder.fit_transform(X[["Sex"]]).toarray()
        column_names=["Female","Male"]
        for i in range(len(matrix.T)):
             X[column_names[i]]=matrix.T[i]

        return X
#特征编码

class FeatureDropper(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        return X.drop(["Embarked","Name","Ticket","Cabin","Sex","N"],axis=1,errors="ignore")
#丢弃无用特征

pipeline=Pipeline([("ageimputer",AgeImputer()),
                   ("featureencoder",FeatureEncoder()),
                   ("featuredropper",FeatureDropper())])
#pipeline将多个步骤拼在一起

data=pd.read_csv('train.csv')
#print(data.describe())
#print(data)
#sns.heatmap(data.corr(),cmap="YlGnBu")
#plt.show()
split= StratifiedShuffleSplit(n_splits=1,test_size=0.2)
for train_indices, test_indices in split.split(data,data[["Survived","Pclass","Sex"]]):
    strat_train_set = data.loc[train_indices]
    strat_test_set = data.loc[test_indices]
#print(strat_test_set)
'''plt.subplots(1,2,1)
strat_train_set['Survived'].hist()
strat_train_set['Pclass'].hist()

plt.subplots(1,2,2)
strat_test_set['Survived'].hist()
strat_test_set['Pclass'].hist()

plt.show()'''

'''
strat_train_set=pipeline.fit_transform(strat_train_set)
#print(strat_train_set.info())
scaler=StandardScaler()
X=strat_train_set.drop(["Survived"],axis=1)
y=strat_train_set["Survived"]

X_data=scaler.fit_transform(X)
y_data=y.to_numpy()

#print(X_data)


###clf=RandomForestClassifier()

param_grid=[
    {"n_estimators":[10,100,200,500],
     "max_depth":[None,5,10],
     "min_samples_split":[2,3,4]}
]

grid_search=GridSearchCV(clf,param_grid,cv=3,scoring="accuracy",return_train_score=True)
#print(grid_search)
grid_search.fit(X_data,y_data)
final_clf=grid_search.best_estimator_
#print(final_clf)

####以下为测试集检测性能部分###
strat_test_set=pipeline.fit_transform(strat_test_set)
X_test=strat_test_set.drop(["Survived"],axis=1)
y_test=strat_test_set["Survived"]

scaler=StandardScaler()
X_data_test=scaler.fit_transform(X_test)
y_data_test=y_test.to_numpy()

#print(final_clf.score(X_data_test,y_data_test)) #输出准确率
'''
####用原本的数据训练####
final_data=pipeline.fit_transform(data)  #把切出来的数据用pipeline处理好

X_final=final_data.drop(["Survived"],axis=1)
y_final=final_data["Survived"]

scaler=StandardScaler() #数据标准化，控制尺度统一
X_data_final=scaler.fit_transform(X_final)
y_data_final=y_final.to_numpy()
#将pandas的dataframe对象转化为numpy数组

prod_clf=RandomForestClassifier()
#prod_clf=svm.SVC()
param_grid=[
    {"n_estimators":[10,100,200,500],
     "max_depth":[None,5,10],
     "min_samples_split":[2,3,4]}
]
'''
param_grid=[
    {"C":[1,10,100,1000],
     "kernel":['linear']}
]   #SVM的参数 
'''


grid_search=GridSearchCV(prod_clf,param_grid,cv=3,scoring="accuracy",return_train_score=True)
grid_search.fit(X_data_final,y_data_final) #这一步执行具体计算

#grid_search_svm=GridSearchCV(prod_clf,param_grid,cv=3,scoring="accuracy",return_train_score=True)
#grid_search_svm.fit(X_data_final,y_data_final)

prod_final_clf=grid_search.best_estimator_
print(prod_final_clf)


test_data=pd.read_csv('test.csv')
final_test_data=pipeline.fit_transform(test_data)  #测试集的数据处理

X_final_test=final_test_data
X_final_test=X_final_test.fillna(method="ffill")

scaler=StandardScaler()
X_data_final_test=scaler.fit_transform(X_final_test)  #分两次分别处理
#y不用处理

predictions=prod_final_clf.predict(X_data_final_test)
final_df=pd.DataFrame(test_data["PassengerId"])
final_df["Survived"]=predictions   #kaggle官方的输出只要这两行
#final_df.to_csv("predictions.csv",index=False)
#final_df.to_csv("predictions_SVM.csv",index=False)
print(final_df)

#随机森林 0.78
#SVM  0.76