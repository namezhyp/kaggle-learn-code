import matplotlib.pyplot as plt
from transformers import BertTokenizer,BertConfig,TFBertModel

'''
x=[1,2,3,4,5]
y=[2,4,6,8,10]
plt.plot(x,y)
plt.title('示例')
plt.show()'''

'''
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
a=tf.constant([1.2,2.3,3.6],shape=[3],name='a')
b=tf.constant([1.2,2.3,3.6],shape=[3],name='b')

c=a+b
session=tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
print(session.run(c))'''
#测试gpu是否启用

from transformers import BertModel, BertTokenizer
import torch

# 加载预训练的BERT模型和分词器
model_name = 'bert-base-uncased'
bert_model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# 输入文本
input_text = "Hello, how are you?"
input_ids = tokenizer.encode(input_text, add_special_tokens=True)
input_ids = torch.tensor(input_ids).unsqueeze(0)  # 添加batch维度

# 通过BERT模型获取隐藏状态
with torch.no_grad():
    last_hidden_state, pooled_output = bert_model(input_ids)

# 添加一个全连接层作为解码器
decoder = torch.nn.Linear(bert_model.config.hidden_size, tokenizer.vocab_size)
decoder_output = decoder(last_hidden_state)

# 使用softmax函数获取每个词的概率分布
probs = torch.nn.functional.softmax(decoder_output, dim=-1)

# 生成文本
generated_text = tokenizer.decode(torch.argmax(probs, dim=-1).squeeze())
print(generated_text)

