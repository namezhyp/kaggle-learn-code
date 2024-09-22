import numpy as np
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertConfig, TFDistilBertModel
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased', config=config)



######这是对小模型的尝试，效果不好

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

train = pd.read_csv('./google-quest/train.csv')
test = pd.read_csv('./google-quest/test.csv')

MAXLEN = 256
BATCH_SIZE = 16
NUM_EPOCHS = 4
LEARNING_RATE = 5e-6

train_data_labels = ['question_title', 'question_body', 'answer', 'category']
labels = ['question_asker_intent_understanding', 'question_body_critical', 'question_conversational', 'question_expect_short_answer', 'question_fact_seeking', 'question_has_commonly_accepted_answer', 'question_interestingness_others', 'question_interestingness_self', 'question_multi_intent', 'question_not_really_a_question', 'question_opinion_seeking', 'question_type_choice', 'question_type_compare', 'question_type_consequence', 'question_type_definition', 'question_type_entity', 'question_type_instructions', 'question_type_procedure', 'question_type_reason_explanation', 'question_type_spelling', 'question_well_written', 'answer_helpful', 'answer_level_of_information', 'answer_plausible', 'answer_relevance', 'answer_satisfaction', 'answer_type_instructions', 'answer_type_procedure', 'answer_type_reason_explanation', 'answer_well_written']

i = 0
train_input_ids = []
train_attention_mask = []

for qa_id in train['qa_id']:
    if(i % 1000 == 0):
        print(i)
    text = train.loc[i, 'question_title'] + train.loc[i, 'question_body'] + train.loc[i, 'answer'] + train.loc[i, 'category']
    input_ids = tokenizer.encode(text, max_length=MAXLEN)
    padding_length = MAXLEN - len(input_ids)
    train_input_ids.append(input_ids + [0]*padding_length)
    train_attention_mask.append([1]*len(input_ids) + [0]*padding_length)
    i = i + 1

train_input_ids = np.array(train_input_ids)
train_attention_mask = np.array(train_attention_mask)

i=0
test_input_ids = []
test_attention_mask = []

for qa_id in test['qa_id']:
    text = test.loc[i, 'question_title'] + test.loc[i, 'question_body'] + test.loc[i, 'answer'] + test.loc[i, 'category']
    input_ids = tokenizer.encode(text, max_length=MAXLEN)
    padding_length = MAXLEN - len(input_ids)
    test_input_ids.append(input_ids + [0]*padding_length)
    test_attention_mask.append([1]*len(input_ids) + [0]*padding_length)

    i=i+1


test_input_ids = np.array(test_input_ids)
test_attention_mask = np.array(test_attention_mask)

y_train = np.array(train[labels])

input_ids = keras.layers.Input(shape=(MAXLEN,), dtype='int32')
attention_mask = keras.layers.Input(shape=(MAXLEN,), dtype='int32')

x = bert_model([input_ids, attention_mask])[0]
x = keras.layers.GlobalAveragePooling1D()(x)
outputs = keras.layers.Dense(30, activation='sigmoid')(x)

model = keras.models.Model(inputs=[input_ids, attention_mask], outputs=outputs)
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])

(train_input_ids, valid_input_ids, train_attention_mask, valid_attention_mask, y_train, y_valid) = train_test_split(train_input_ids, train_attention_mask, y_train, test_size=0.1, shuffle=True, random_state=0)

early_stopping = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

model.fit([train_input_ids, train_attention_mask], y_train, validation_data=([valid_input_ids, valid_attention_mask], y_valid), batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, callbacks=[early_stopping])

y_pred = model.predict([test_input_ids, test_attention_mask], batch_size=BATCH_SIZE, verbose=1)

submission = pd.read_csv('./google-quest/sample_submission.csv')
submission[labels] = y_pred
submission['qa_id'] = test['qa_id']
submission.to_csv('./google-quest/final_submission.csv', index=False)
