import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.ensemble import RandomForestRegressor
#classifer用于分类  regressor用于预测连续值 
from sklearn.model_selection import cross_val_score
from nltk.stem.snowball import SnowballStemmer
#bagging  袋装分类器
#原始训练集里多次又放回得到多个训练子集，训练多个分类器
#最终预测结果用表决方法获得


#####自选工具的网页##

df_train=pd.read_csv('text_similar/train.csv',encoding="ISO-8859-1")
df_test=pd.read_csv('text_similar/test.csv',encoding="ISO-8859-1")

df_desc=pd.read_csv("text_similar/product_descriptions.csv")
df_all=pd.concat((df_train,df_test),axis=0,ignore_index=True)

df_all=pd.merge(df_all,df_desc,how='left',on='product_uid')
#print(df_all.head())


#######预处理##
stemmer=SnowballStemmer('english')
#作用是还原词语

def str_stemmer(s):
    return " ".join([stemmer.stem(word) for word in s.lower().split()])
#对句子里每个词遍历进行词干提取
#在搜索匹配里，词形归一是很重要的

def str_common_word(str1,str2): #搜索str和目标str
    return sum(int(str2.find(word)>=0) for word in str1.split())

df_all['search_term']= df_all['search_term'].map(lambda x:str_stemmer(x))
df_all['product_title']= df_all['product_title'].map(lambda x:str_stemmer(x))
df_all['product_description']=df_all['product_description'].map(lambda x:str_stemmer(x))


#####预处理后，自制文本特征####

df_all['len_of_query'] =df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
#map()函数会将方法应用到规定队列的每个元素里
#关键词长度
df_all['commoms_in_title']=df_all.apply(lambda x:str_common_word(x['search_term'],x['product_title']),axis=1)
#标题里有多少次关键词的重合
#apply()是pd带的函数，要规定行或者列，也可以让函数当它的参数，它也会返回一个包含结果的df结构
        #脑洞打开，自己想可以怎么发现什么特征可能有用
df_all['commons_in_desc']=df_all.apply(lambda x:str_common_word(x['search_term'],x['product_description']),axis=1)
#描述里有几次关键词重合
df_all=df_all.drop(['search_term','product_title','product_description'],axis=1)
#模型学不了的特征直接丢掉


####预处理后，恢复数据的样子
df_train=df_all.loc[df_train.index]
df_test=df_all.loc[df_test.index]
#划分训练集和测试集
    #pandas里的数据，二维的是dataframe，一维的是series
    #这里就是用df_train的索引去切出df_all的对应行

test_ids=df_test['id']
#记录测试集的id，没有也行

y_train=df_train['relevance'].values
#取出标签
X_train=df_train.drop(['id','relevance'],axis=1).values
X_test=df_test.drop(['id','relevance'],axis=1).values



params=[1,3,5,6,7,8,9,10]
test_scores=[]
for param in params:
    clf=RandomForestRegressor(n_estimators=30,max_depth=param)
#n_estimators 决策树数量，默认100
    test_score=np.sort(-cross_val_score(clf,X_train,y_train,cv=5,scoring='neg_mean_squared_error'))
    #cross_val_score是交叉验证，会自动调用对应模型的fit()
    test_scores.append(np.mean(test_score))

'''
import matplotlib.pyplot as plt
plt.plot(params,test_scores)
plt.title("Params vs CV error")
'''
rf=RandomForestRegressor(n_estimators=30,max_depth=6)
rf.fit(X_train,y_train)

y_pred=rf.predict(X_test)
pd.DataFrame({'id':test_ids,'relevance':y_pred}).to_csv('text_similar/prediction.csv',index=False)