import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import cross_val_score

df_train=pd.read_csv('home-depot-product-search/train.csv',encoding="ISO-8859-1")
df_test=pd.read_csv('home-depot-product-search/test.csv',encoding="ISO-8859-1")
df_desc=pd.read_csv('home-depot-product-search/product_descriptions.csv',encoding="ISO-8859-1")

df_all=pd.concat((df_train,df_test),axis=0,ignore_index=True)
df_all=pd.merge(df_all,df_desc,how='left',on='product_uid')

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

 ######进阶文本处理
import Levenshtein

#Levenshtein.ratio('hello','hello world')
#测算文本的距离  函数演示  0.625
#文本距离就是从A词到B词要经过几次修改

df_all['dist_in_title']= df_all.apply(lambda x:Levenshtein.ratio(x['search_term'],x['product_title']), axis=1)
df_all['dist_in_desc']= df_all.apply(lambda x:Levenshtein.ratio(x['search_term'],x['product_description']), axis=1)

df_all['all_texts']=df_all['product_title']+'.'+df_all['product_description']+'.'
#合在一起作为语料

#TF表示一个词再文本中出现概率
#IDF表示一个词再整个语料库的普遍程度
#TF-IDF可以算出每个词在文本中的权重

from gensim.utils import tokenize
from gensim.corpora.dictionary import Dictionary
dictionary=Dictionary(list(tokenize(x,errors='ignore')) for x in df_all['all_texts'].values)
#print(dictionary)
#类似一个词典  输出应该是221877个单词的列表
'''Dictionary<221877 unique tokens: ['a', 'against', 'alonehelp', 'also', 'and']...>'''

def myCorpus(object):
    def __iter__(self):
        for x in df_all['all_texts'].values:
            yield dictionary.doc2bow(list(tokenize(x,errors='ignore')))
    '''#扫过所有语料，转化成词袋表示
#词袋表示无视了词语间关系，全部分开表示
#简单地来说，词袋会整理出所有词语
#然后把句子转化成一个向量
#向量每个值分别代表对应词语在句中出现次数'''
corpus=myCorpus()
#这么做是为了内存友好，一整个list不方便读取
#这样就获得了一个标准语料库

from gensim.models.tfidfmodel import TfidfModel
tfidf=TfidfModel(corpus)
#对已经变成BoW向量的数组，做TF-IDF计算
'''
tfidf[dictionary.doc2bow(list(tokenize('hello world,good morning',errors='ingore')))]
举个例子，“hello world，good morning”变成
一个列表，每个元素都包含两个数字，一个是词在字典里的序号
一个是对应tfidf值
'''
from gensim.similarities import MatrixSimilarity
def to_tfidf(text):
    res =tfidf[dictionary.doc2bow(list(tokenize(text,errors='ignore')))]
    return res

def cos_simi(text1,text2):
    tfidf1=to_tfidf(text1)
    tfidf2=to_tfidf(text2)
    index=MatrixSimilarity([tfidf1],num_features=len(dictionary))
    simi=index[tfidf2]
    return float(simi[0])
#向量内积，短的向量补充到长的  结果是np.array，所以用[0]取就行了

df_all['tdidf_cos_sim_in_title']=df_all.apply(lambda x:cos_simi(x['search_term'],x['product_title']),axis=1)
print(df_all['tdidf_cos_sim_in_title'])[:5]

df_all['tdidf_cos_sim_in_desc']=df_all.apply(lambda x:cos_simi(x['search_term'],x['product_description']),axis=1)
#再新增两个自制特征

####使用word2vec
from gensim.models.word2vec import Word2Vec
model=Word2Vec(corpus,size=128,windows=5,min_count=5,workers=4)


def get_vector(text):
    res=np.zeros([128])
    count=0
    for word in word_tokenize(text):
        if word in model.wv:
            res+=model.wv[word]
            count+=1
    return res/count

y_train=df_train['relevance'].values
#取出标签
X_train=df_train.drop(['id','relevance'],axis=1).values
X_test=df_test.drop(['id','relevance'],axis=1).values



params=[10,20,30,50,100,200,400]
test_scores=[]
for param in params:
    clf=RandomForestClassifier(n_estimators=param)
    test_score=cross_val_score(clf,X_train,y_train,cv=3,scoring='accuracy')#分为3折，进行交叉验证,每次输出的test_score都是一个数组
    test_scores.append(np.mean(test_score)) #计算均值

best_estimators=params[np.argmax(test_score)]
rf=RandomForestClassifier(n_estimators=best_estimators)
rf.fit(X_train,y_train)

y_pred=rf.predict(X_test)


