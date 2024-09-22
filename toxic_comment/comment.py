import numpy as np
import pandas as pd

#可视化
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from wordcloud import WordCloud,STOPWORDS
#nlp
import string
import re   #正则表达式
import nltk
from nltk.corpus import stopwords  
#停止词 统一直接删除 that、the等词  导入以后直接过滤
from nltk.stem.wordnet import WordNetLemmatizer

#特征工程
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedGroupKFold

#模型
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
import xgboost as xgb

from sklearn import metrics
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,roc_auc_score
from sklearn.metrics import fbeta_score,log_loss,hamming_loss,consensus_score

import warnings

import os  #调试用
print(os.getcwd())

font={'family':'serif',
      'weight':'normal',
      'size':'14'}
plt.rc('font',**font)

train=pd.read_csv('toxic_comment/train.csv')
test=pd.read_csv('toxic_comment/test.csv')
test_y=pd.read_csv('toxic_comment/test_labels.csv')

#print(train.head())