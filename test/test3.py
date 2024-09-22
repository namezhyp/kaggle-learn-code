import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from datetime import date


from nltk.corpus import stopwords  
import re
from nltk.stem import WordNetLemmatizer

from gensim.models.word2vec import Word2Vec
from nltk.tokenize import word_tokenize


from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

####新闻分析V3####
'''网址：https://www.kaggle.com/datasets/aaron7sun/stocknews'''
data=pd.read_csv('test/Combined_News_DJIA.csv')

data["combined_news"]=data.filter(regex=("Top.*")).apply(lambda x:''.join(str(x.values)),axis=1)
#正则表达式，取出所有top几的新闻 得到新的列叫combined news


#分割测试集和训练集
train=data[data['Date'] <'2015-01-01']
test=data[data['Date'] >'2014-12-31']

#####清洗词语####

X_train=train[train.columns[2:]]
corpus=X_train.values.flatten().astype(str)
#corpus是全部可见的文本资料
#dataframe格式转成numpy二维数组，然后被改成一维数组，里面的非字符串转字符串
#corpus最后是一个一维字符串数组

X_train=X_train.values.astype(str)
X_train=np.array([' '.join(x) for x in X_train])
#将字符串列表的每个元素（还是一堆字符串）拼起来
#如['this','is','an','apple']拼成一个字符串
X_test=test[test.columns[2:]]
X_test=np.array([' '.join(x) for x in X_test])
y_train=train['Label'].values
y_test=test['Label'].values

#x_train这些不能随便flatten

corpus=[word_tokenize(x) for x in corpus]
X_train=[word_tokenize(x) for x in X_train]
X_test=[word_tokenize(x) for x in X_test]
#对字符串进行拆分，拆成词语
##########
#print(X_train[:2])

####优化预处理#####
stop =stopwords.words('english')
#获取英文停用词列表

def hasNumbers(input):
    return bool(re.search(r'\d',input))
#检查输入是否有数字，有则返回true

def isSymbol(input):
    return bool(re.match(r'[^\w]',input))
#检查符号

WordNet_Lemmatizer=WordNetLemmatizer()
#词语形状还原器  better->good

def check(word):
    word=word.lower()
    if word in stop:
        return False
    elif hasNumbers(word):
        return False
    elif isSymbol(word):
        return False
    else:
        return True

def preprocessing(sen):
    res=[]
    for word in sen:
        if check(word):
            word=word.lower().replace("b'",'').replace('b"','').replace('"','').replace("'",'')
            res.append(WordNet_Lemmatizer.lemmatize(word))
    return res

corpus=[preprocessing(x) for x in corpus]
X_train=[preprocessing(x) for x in X_train]
X_test=[preprocessing(x) for x in X_test]

'''print(corpus[553])
print(X_train[523])'''

#corpus是语料，每个元素是句子，按句子切分词语
#X-train则是每天25条新闻被合成拼接再切分小元素
#它们不是一一对应的


####训练NLP模型###

model =Word2Vec(corpus,vector_size=128,window=5,min_count=5,workers=4)
#此时单词就可以像字典一样查出对应向量了
#windows指上下文窗口的大小

#print(model.wv.get_vector('ok')) 查看ok的对应向量

#vocab=model.wv.index_to_key  

#查询任意text的vector
def get_vector(word_list):
    res=np.zeros([128])
    count=0
    for word in word_list:
        if word in model.wv:
            res+=model.wv[word]
            count+=1
    return res/count
#任意word——list的平均vector

#print(get_vector(['hello','from','the','other','side']))

params=[0.1,0.5,1,3,5,7,10,12,16,20,25,30,35,40]
test_scores=[]
for param in params:
    clf=SVR(gamma=param)
    test_score=cross_val_score(clf,X_train,y_train,cv=3,scoring='roc_auc')
    test_scores.append(np.mean(test_score))
#测试各个参数下SVR的性能



'''
import matplotlib.pyplot as plt
plt.plot(params,test_scores)
plt.title("Param cs CV AUCS Score")
'''

'''
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
'''