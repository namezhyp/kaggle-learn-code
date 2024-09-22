
import numpy as np
import pandas as pd

from transformers import BertTokenizer,BertConfig,TFBertModel


MAXLEN=512
BATCH_SIZE=16
NUM_EPOCHS=3
LEARNING_RATE=5e-6


train = pd.read_csv('./google-quest/train.csv')

tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
config=BertConfig.from_pretrained('bert-base-uncased')
bert_model=TFBertModel.from_pretrained('bert-base-uncased',config=config)

train_data_labels=['question_title','question_body','answer','category']

labels=['question_asker_intent_understanding','question_body_critical','question_conversational','question_expect_short_answer','question_fact_seeking','question_has_commonly_accepted_answer','question_interestingness_others',
'question_interestingness_self','question_multi_intent','question_not_really_a_question','question_opinion_seeking','question_type_choice','question_type_compare','question_type_consequence','question_type_definition','question_type_entity','question_type_instructions',
'question_type_procedure','question_type_reason_explanation','question_type_spelling','question_well_written','answer_helpful','answer_level_of_information','answer_plausible','answer_relevance','answer_satisfaction','answer_type_instructions','answer_type_procedure',
'answer_type_reason_explanation','answer_well_written']

train_input_ids=[]
train_attention_mask=[]
train_token_type_ids=[]

text=train.loc[0,'question_title']+train.loc[0,'question_body']+train.loc[0,'answer']+train.loc[0,'category']
print(text)
print("文本长度：",len(text))   #1746，不过这是字符串的总长度
input_ids=tokenizer.encode(text,max_length=MAXLEN,truncation=True)#太长的句子被截断，就只返回第一部分，后半会丢失
print(input_ids)
print(len(input_ids))   #用了300多  这是token的数量
final_text=tokenizer.decode(input_ids)
print("翻译回来的结果：",final_text)   #足够了
padding_length=MAXLEN-len(input_ids)
print("padding length:",padding_length)
train_input_ids.append(input_ids+[0]*padding_length) #加[0]表示添加一个元素0，用作边界划分
print(train_input_ids)
print("train_input的长度:",len(train_input_ids))    #1
train_attention_mask.append([1]*len(input_ids)+[0]*padding_length)  #填充部分设置为0
train_token_type_ids.append([0]*MAXLEN)