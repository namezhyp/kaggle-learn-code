import numpy as np
import pandas as pd
from transformers import BertTokenizer,DistilBertConfig,TFBertModel,DistilBertModel

import torch 
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

if torch.cuda.is_available():
    print("CUDA is available")
    device = torch.device("cuda")
else:
    print("CUDA is not available")
    device = torch.device("cpu")
####torch可以直接使用cuda，不用配置

data=pd.read_csv('/kaggle/game-char/data.csv')

###蒸馏学生模型

##gpt没有中文，只能考虑bert
tokenizer=BertTokenizer.from_pretrained('bert-base-chinese')
teacher_model=TFBertModel.from_pretrained('bert-base-chinese')
config=DistilBertConfig(vocab_size=tokenizer.vocab_size)
student_model=DistilBertModel(config)
##学生模型不要导入预训练的模型而是直接利用教师来生成


MAXLEN=128
BATCH_SIZE=8
NUM_EPOCHS=3
LEARNING_RATE=5e-6

###########数据预处理
class CustomDataset(Dataset):  #继承dataset
    def __init__(self, data):
        self.input_ids = data[0]
        self.attention_mask=data[1]
        self.token_type_id=data[2]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, data_type,idx):   #根据索引返回一个数据
        return self.data[data_type][idx]


train_input_ids=[]
train_attention_mask=[]
train_token_type_ids=[]
X_data=data['text']

for i in range(len(X_data)):
    if(i%1000==0):   print(i)
 
    input_ids=tokenizer.encode(X_data[i],max_length=MAXLEN)
    padding_length=MAXLEN-len(input_ids)

    train_input_ids.append(input_ids+[0]*padding_length) #加[0]表示添加一个元素0，用作边界划分
    train_attention_mask.append([1]*len(input_ids)+[0]*padding_length)  #填充部分设置为0
    train_token_type_ids.append([0]*MAXLEN)

train_input_ids=np.array(train_input_ids)
train_attention_mask=np.array(train_attention_mask)
train_token_type_ids=np.array(train_token_type_ids)

data=train_input_ids+train_attention_mask+train_token_type_ids

#####################
# 创建一个CustomDataset实例
dataset = CustomDataset(data)
# 创建一个DataLoader实例
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

criterion = nn.MSELoss()  #损失函数
optimizer = torch.optim.Adam(student_model.parameters(), lr=LEARNING_RATE)  #优化器

# 假设我们有一个数据加载器，它会返回输入序列和目标序列
for epoch in range(NUM_EPOCHS):
    for batch in dataloader:
        input_ids = train_input_ids
        attention_mask = train_attention_mask
        token_type_id=train_token_type_ids
        # 将数据移动到设备上
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_id=token_type_id.to(device)
        # 使用教师模型和学生模型计算输出
        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids, attention_mask=attention_mask,token_type_id=token_type_id)
        #禁止梯度计算，教师模型不会更新  其实就算不写也不会计算梯度，只不过要多占用一些显存空间

        student_outputs = student_model(input_ids, attention_mask=attention_mask,token_type_id=token_type_id)
        
        # 计算损失       
        loss = criterion(student_outputs[0], teacher_outputs[0])
        # 反向传播和优化
        optimizer.zero_grad()  #每次反向传播时要主动清零 梯度  
        loss.backward()      #只有学生模型会更新
        optimizer.step() #更新模型参数