import numpy as np
import pandas as pd 
from transformers import BertTokenizer,BertConfig,TFBertModel
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

'''原任务是筛选特定文本，但是直接把筛选后文本作为标签，
显然不太合适，所以考虑增加一列用来标记每个词是否需要保留'''

#####初始化部分###
train=pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
test=pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

tokenizer=BertTokenizer.from_pretrained('/kaggle/input/bert-model2')
config=BertConfig.from_pretrained('/kaggle/input/bert-model2')
bert_model=TFBertModel.from_pretrained('/kaggle/input/bert-model2',config=config)

MAXLEN=50
BATCH_SIZE=16
NUM_EPOCHS=2
LEARNING_RATE=5e-6


#####数据预处理
train_input_ids=[]
train_attention_mask=[]
train_token_type_ids=[]
train_labels = []

def generate_labels(text_token, selected_text_token):
    len1=len(text_token)
    labels=[0]*len1
    labels[0]=0  #开头标志不应该被学习到  
    
    j=1
    for i in range(1,len1):
        if(j>=len(selected_text_token)-1):
            break
        if(text_token[i]==selected_text_token[j]): 
            labels[i]=1
            j+=1
        else: 
            labels[i]=0
    labels[i]=0  #结尾标志也不应该学到
    return labels

X_data=train['sentiment']+' '+train['text']  
for i in range(len(X_data)):
    if(i%1000==0):   print(i)
    if(pd.isna(X_data[i])):
        print("出现na,位置是：",i)
        continue
    input_ids=tokenizer.encode(X_data[i],max_length=MAXLEN)
    selected_text_ids=tokenizer.encode(train.loc[i,'selected_text'],max_length=MAXLEN)
    labels=generate_labels(input_ids,selected_text_ids)

    padding_length=MAXLEN-len(input_ids)

    train_input_ids.append(input_ids+[0]*padding_length) #加[0]表示添加一个元素0，用作边界划分
    train_labels.append(labels+[0]*padding_length)  #标签的填充到底用多少好？
    train_attention_mask.append([1]*len(input_ids)+[0]*padding_length)  #填充部分设置为0
    train_token_type_ids.append([0]*MAXLEN)

train_input_ids=np.array(train_input_ids)
train_attention_mask=np.array(train_attention_mask)
train_token_type_ids=np.array(train_token_type_ids)
train_labels = np.array(train_labels)

y_train=train_labels

#######################
test_input_ids=[]
test_attention_mask=[]
test_token_type_ids=[]


X_test=test['sentiment']+' '+test['text']    
for i in range(len(X_test)):
    if(i%1000==0):   print(i)
    input_ids=tokenizer.encode(X_test[i],max_length=MAXLEN)
  
    padding_length=MAXLEN-len(input_ids)

    test_input_ids.append(input_ids+[0]*padding_length) #加[0]表示添加一个元素0，用作边界划分
    test_attention_mask.append([1]*len(input_ids)+[0]*padding_length)  #填充部分设置为0
    test_token_type_ids.append([0]*MAXLEN)


test_input_ids=np.array(test_input_ids)
test_attention_mask=np.array(test_attention_mask)
test_token_type_ids=np.array(test_token_type_ids)



####训练模型###
input_ids=keras.layers.Input(shape=(MAXLEN,),dtype='int32')
#shape=(,) 第一个参数是每个样本的特征数，后面空着表示输入序列长度不限
attention_mask=keras.layers.Input(shape=(MAXLEN,),dtype='int32')  #形状是一维数组  
#这里的掩码只用于防止模型学习填充部分，模型内的掩码在训练时已经解决
token_type_ids=keras.layers.Input(shape=(MAXLEN,),dtype='int32')
#这三行规定了模型三个输入的格式，里面没有具体数据，但可以作为输入

x=bert_model([input_ids,attention_mask,token_type_ids])[1]
#下划线表示忽略其他层 只关注最后隐藏层的输出 第一个参数是每个时刻的输出


outputs=keras.layers.Dense(50,activation='sigmoid')(x)
model=keras.models.Model(inputs=[input_ids,attention_mask,token_type_ids],outputs=outputs)
model.compile(loss='binary_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),metrics=['accuracy'])
#模型编译   sparse_categorical_crossentropy用于多分类

(train_input_ids,valid_input_ids,
 train_attention_mask,valid_attention_mask,
 train_token_type_ids,valid_token_type_ids,
 y_train,y_valid)=train_test_split(train_input_ids,train_attention_mask,train_token_type_ids,y_train,test_size=0.1,
     shuffle=True,random_state=0)


early_stopping=keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True) 
#回调函数，要求模型性能不改善时停止训练，且结束后要恢复到最佳性能的参数

model.fit([train_input_ids,train_attention_mask,train_token_type_ids],y_train,
          validation_data=([valid_input_ids,valid_attention_mask,valid_token_type_ids],y_valid),
          batch_size=BATCH_SIZE,epochs=NUM_EPOCHS,callbacks=[early_stopping])

y_pred=model.predict([test_input_ids,test_attention_mask,test_token_type_ids],batch_size=BATCH_SIZE,verbose=1)#多维数组变一维数组

test_label=y_pred*test_attention_mask  #这一步删去无效部分  这段代码肯定要留
softmax_y_pred=tf.nn.softmax(test_label,axis=-1)#然后重新用softmax函数分配概率
y_pred1=(test_label>0.16).astype(int)

#######输出处理
test_selected_text=[]

def generate_text(text_token,labels):
    #输入源文本token和标签，还原提取出筛选文本
    len1=len(text_token)
    selected_text_token=[0]*len1
    j=0
    for i in range(len1):
        if(labels[i]==1): 
            selected_text_token[j]=text_token[i]
            j+=1
    
    selected_text=tokenizer.decode(selected_text_token)
    return selected_text

for n in range(len(y_pred)):
    res=generate_text(test_input_ids[n],y_pred1[n])  #逐行还原回文本
    res=res.replace('[PAD]','')
    res=res.replace('[SEP]','')
    test_selected_text.append(res)

submisson=pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
submisson['selected_text']=test_selected_text
test=pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
submisson['textID']=test['textID']
submisson.to_csv('/kaggle/working/submission.csv',index=False)
print("finished")