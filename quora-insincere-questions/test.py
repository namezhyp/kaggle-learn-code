import numpy as np
import pandas as pd
from transformers import BertTokenizer,BertConfig,TFBertModel
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf


MAXLEN=128
BATCH_SIZE=32
NUM_EPOCHS=5
LEARNING_RATE=5e-6

train=pd.read_csv('disaster-tweet/train.csv')
test=pd.read_csv('disaster-tweet/test.csv')

tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
config=BertConfig.from_pretrained('bert-base-uncased')
bert_model=TFBertModel.from_pretrained('bert-base-uncased',config=config)

#https://www.kaggle.com/competitions/quora-insincere-questions-classification/overview
#quora问题分类，数据太多所以暂时放弃做