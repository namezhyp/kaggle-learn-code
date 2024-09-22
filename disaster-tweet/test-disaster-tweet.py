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

#############
text=train['text'][0]
print(text)
input_ids=tokenizer.encode(text,max_length=MAXLEN)
print(input_ids)#tokensize可以将句子每个词转成对应token的数字
print(tokenizer.convert_ids_to_tokens(input_ids)) #转回token可以发现首尾加了标识符
###分词 演示###

train_input_ids=[]
train_attention_mask=[]
train_token_type_ids=[]
for text in train['text']:#每次传入一句话
    input_ids=tokenizer.encode(text,max_length=MAXLEN)#太长的句子被截断，就只返回第一部分，后半会丢失
    padding_length=MAXLEN-len(input_ids)

    train_input_ids.append(input_ids+[0]*padding_length) #加[0]表示添加一个元素0，用作边界划分
    train_attention_mask.append([1]*len(input_ids)+[0]*padding_length)  #填充部分设置为0
    train_token_type_ids.append([0]*MAXLEN)

train_input_ids=np.array(train_input_ids)
train_attention_mask=np.array(train_attention_mask)
train_token_type_ids=np.array(train_token_type_ids)
#token_type_id用于指示输入序列里不同句子或片段 就一个句子，全设成0就行


test_input_ids=[]
test_attention_mask=[]
test_token_type_ids=[]
for text in test['text']:#每次传入一句话
    input_ids=tokenizer.encode(text,max_length=MAXLEN)#太长的句子被截断，就只返回第一部分，后半会丢失
    padding_length=MAXLEN-len(input_ids)

    test_input_ids.append(input_ids+[0]*padding_length) #加[0]表示添加一个元素0，用作边界划分
    test_attention_mask.append([1]*len(input_ids)+[0]*padding_length)
    test_token_type_ids.append([0]*MAXLEN)

test_input_ids=np.array(test_input_ids)
test_attention_mask=np.array(test_attention_mask)
test_token_type_ids=np.array(test_token_type_ids)


y_train=np.array(train['target'])

####训练模型###
input_ids=keras.layers.Input(shape=(MAXLEN,),dtype='int32')
#shape=(,) 第一个参数是每个样本的特征数，后面空着表示输入序列长度不限
attention_mask=keras.layers.Input(shape=(MAXLEN,),dtype='int32')  #形状是一维数组  
#这里的掩码只用于防止模型学习填充部分，模型内的掩码在训练时已经解决
token_type_ids=keras.layers.Input(shape=(MAXLEN,),dtype='int32')
#这三行规定了模型三个输入的格式，里面没有具体数据，但可以作为输入

x=bert_model([input_ids,attention_mask,token_type_ids])[1]
#下划线表示忽略其他层 只关注最后隐藏层的输出 第一个参数是每个时刻的输出
#模型已经预训练好了，可以直接输入数据得到最后的输出

outputs=keras.layers.Dense(1,activation='sigmoid')(x)
#给模型加一个全连接层，仅一个神经元，
# 将bert的输出x调整后用sigmoid函数做一个二分类


'''上面的这些input_ids x这些对象，
都只是表达了一个关系
它们并没有被立刻算出来
要到模型编译并fit()以后才会被正式计算出值'''

model=keras.models.Model(inputs=[input_ids,attention_mask,token_type_ids],outputs=outputs)
model.compile(loss='binary_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),metrics=['accuracy'])
#模型编译  
'''对于深度学习，库里面的模型没有直接绑定损失函数和优化器
所以使用时需要手动参数选好以后现场编译一个模型
对于这个模型，本次的任务是文本分类
bert模型已经预先训练好了，但为了分类加的全连接层没有
所以要重新编译
随机森林等简单模型在一开始就已经设计好了，不用编译'''

#(train_input_ids,valid_input_ids)
(train_input_ids,valid_input_ids,
 train_attention_mask,valid_attention_mask,
 train_token_type_ids,valid_token_type_ids,
 y_train,y_valid)=train_test_split(train_input_ids,train_attention_mask,train_token_type_ids,y_train,test_size=0.1,
     stratify=y_train,random_state=0)
#将输入数据按指定比例切成训练集和验证集
#valid 一般表示验证

early_stopping=keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True)
#回调函数，要求模型性能不改善时停止训练，且结束后要恢复到最佳性能的参数

model.fit([train_input_ids,train_attention_mask,train_token_type_ids],y_train,
          validation_data=([valid_input_ids,valid_attention_mask,valid_token_type_ids],y_valid),
          batch_size=BATCH_SIZE,epochs=NUM_EPOCHS,callbacks=[early_stopping])



y_pred=model.predict([test_input_ids,test_attention_mask,test_token_type_ids],batch_size=BATCH_SIZE,verbose=1).ravel()#多维数组变一维数组
y_pred=(y_pred>=0.5).astype(int)

submisson=pd.read_csv('disaster-tweet/submission.csv')
submisson['target']=y_pred

submisson.to_csv('disaster-tweet/final_submission.csv',index=False)

