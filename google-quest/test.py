
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from transformers import BertTokenizer,BertConfig,TFBertModel
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf


tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
config=BertConfig.from_pretrained('bert-base-uncased')
bert_model=TFBertModel.from_pretrained('bert-base-uncased',config=config)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


train=pd.read_csv('./google-quest/train.csv')
test=pd.read_csv('./google-quest/test.csv')
#sub_label=pd.read_csv('/google-quest-challenge/sample_submission.csv')


MAXLEN=256
BATCH_SIZE=16
NUM_EPOCHS=4
LEARNING_RATE=5e-6

train_data_labels=['question_title','question_body','answer','category']

labels=['question_asker_intent_understanding','question_body_critical','question_conversational','question_expect_short_answer','question_fact_seeking','question_has_commonly_accepted_answer','question_interestingness_others',
'question_interestingness_self','question_multi_intent','question_not_really_a_question','question_opinion_seeking','question_type_choice','question_type_compare','question_type_consequence','question_type_definition','question_type_entity','question_type_instructions',
'question_type_procedure','question_type_reason_explanation','question_type_spelling','question_well_written','answer_helpful','answer_level_of_information','answer_plausible','answer_relevance','answer_satisfaction','answer_type_instructions','answer_type_procedure',
'answer_type_reason_explanation','answer_well_written']


i=0

train_input_ids=[]
train_attention_mask=[]
train_token_type_ids=[]
for qa_id in train['qa_id']:
    if(i%1000==0):   print(i)
    text=train.loc[i,'question_title']+train.loc[i,'question_body']+train.loc[i,'answer']+train.loc[i,'category']
    input_ids=tokenizer.encode(text,max_length=MAXLEN)#太长的句子被截断，就只返回第一部分，后半会丢失
    padding_length=MAXLEN-len(input_ids)

    train_input_ids.append(input_ids+[0]*padding_length) #加[0]表示添加一个元素0，用作边界划分
    train_attention_mask.append([1]*len(input_ids)+[0]*padding_length)  #填充部分设置为0
    train_token_type_ids.append([0]*MAXLEN)

    i=i+1

train_input_ids=np.array(train_input_ids)
train_attention_mask=np.array(train_attention_mask)
train_token_type_ids=np.array(train_token_type_ids)
#token_type_id用于指示输入序列里不同句子或片段 就一个句子，全设成0就行

i=0

test_input_ids=[]
test_attention_mask=[]
test_token_type_ids=[]
for qa_id in test['qa_id']:#每次传入一句话
    text=test.loc[i,'question_title']+test.loc[i,'question_body']+test.loc[i,'answer']+test.loc[i,'category']
    input_ids=tokenizer.encode(text,max_length=MAXLEN)#太长的句子被截断，就只返回第一部分，后半会丢失
    padding_length=MAXLEN-len(input_ids)

    test_input_ids.append(input_ids+[0]*padding_length) #加[0]表示添加一个元素0，用作边界划分
    test_attention_mask.append([1]*len(input_ids)+[0]*padding_length)
    test_token_type_ids.append([0]*MAXLEN)

    i=i+1


test_input_ids=np.array(test_input_ids)
test_attention_mask=np.array(test_attention_mask)
test_token_type_ids=np.array(test_token_type_ids)

y_train=np.array(train[labels])

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

outputs=keras.layers.Dense(30,activation='sigmoid')(x)
#给模型加一个全连接层，30个神经元，一层会不会不太够？



model=keras.models.Model(inputs=[input_ids,attention_mask,token_type_ids],outputs=outputs)
model.compile(loss='binary_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),metrics=['accuracy'])
#模型编译  



(train_input_ids,valid_input_ids,
 train_attention_mask,valid_attention_mask,
 train_token_type_ids,valid_token_type_ids,
 y_train,y_valid)=train_test_split(train_input_ids,train_attention_mask,train_token_type_ids,y_train,test_size=0.1,
     shuffle=True,random_state=0)
#将输入数据按指定比例切成训练集和验证集

#from skmultilearn.model_selection import iterative_stratification

#iter_strat=iterative_stratification(n_splits=2, test_size=0.05, random_state=0)


early_stopping=keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True)
#回调函数，要求模型性能不改善时停止训练，且结束后要恢复到最佳性能的参数

model.fit([train_input_ids,train_attention_mask,train_token_type_ids],y_train,
          validation_data=([valid_input_ids,valid_attention_mask,valid_token_type_ids],y_valid),
          batch_size=BATCH_SIZE,epochs=NUM_EPOCHS,callbacks=[early_stopping])



y_pred=model.predict([test_input_ids,test_attention_mask,test_token_type_ids],batch_size=BATCH_SIZE,verbose=1)#多维数组变一维数组
#这次的数据是连续的，不需要强制划分1 0

submisson=pd.read_csv('./google-quest/sample_submission.csv')
submisson[labels]=y_pred
submisson['qa_id']=test['qa_id']
#赋值时也要注意pd取多列时要用列表包起来
submisson.to_csv('./google-questfinal_submission.csv',index=False)