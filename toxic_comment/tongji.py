import numpy as np
import pandas as pd
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

#train=pd.read_csv('toxic_comment/train.csv')
#test=pd.read_csv('toxic_comment/test.csv')


#y_train=np.array(train[['toxic','severe_toxic','obscene','threat','insult','identity_hate']])
