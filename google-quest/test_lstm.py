import numpy as np
import pandas as pd
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf


######改用lstm，效果及其糟糕，下次不用了

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train = pd.read_csv('./google-quest/train.csv')
test = pd.read_csv('./google-quest/test.csv')

MAXLEN = 512
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 5e-6

train_data_labels = ['question_title', 'question_body', 'answer', 'category']
labels = ['question_asker_intent_understanding', 'question_body_critical', 'question_conversational', 'question_expect_short_answer', 'question_fact_seeking', 'question_has_commonly_accepted_answer', 'question_interestingness_others', 'question_interestingness_self', 'question_multi_intent', 'question_not_really_a_question', 'question_opinion_seeking', 'question_type_choice', 'question_type_compare', 'question_type_consequence', 'question_type_definition', 'question_type_entity', 'question_type_instructions', 'question_type_procedure', 'question_type_reason_explanation', 'question_type_spelling', 'question_well_written', 'answer_helpful', 'answer_level_of_information', 'answer_plausible', 'answer_relevance', 'answer_satisfaction', 'answer_type_instructions', 'answer_type_procedure', 'answer_type_reason_explanation', 'answer_well_written']

def preprocess_data(data):
    input_ids = []
    attention_mask = []

    for i, row in data.iterrows():
        text = row['question_title'] + row['question_body'] + row['answer'] + row['category']
        encoded = tokenizer.encode_plus(text, max_length=MAXLEN, padding='max_length', truncation=True)
        input_ids.append(encoded['input_ids'])
        attention_mask.append(encoded['attention_mask'])

    return np.array(input_ids), np.array(attention_mask)

train_input_ids, train_attention_mask = preprocess_data(train)
test_input_ids, test_attention_mask = preprocess_data(test)

y_train = np.array(train[labels])

input_ids = keras.layers.Input(shape=(MAXLEN,), dtype='int32')
attention_mask = keras.layers.Input(shape=(MAXLEN,), dtype='int32')

embedding_layer = keras.layers.Embedding(tokenizer.vocab_size, 128, input_length=MAXLEN)
embedded_input_ids = embedding_layer(input_ids)

masking_layer = keras.layers.Masking(mask_value=0)  # 假设填充值为0
masked_input_ids = masking_layer(embedded_input_ids)

lstm_layer = keras.layers.LSTM(128)
lstm_output = lstm_layer(embedded_input_ids)

outputs = keras.layers.Dense(30, activation='sigmoid')(lstm_output)

model = keras.models.Model(inputs=[input_ids, attention_mask], outputs=outputs)
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])

(train_input_ids, valid_input_ids, train_attention_mask, valid_attention_mask, y_train, y_valid) = train_test_split(train_input_ids, train_attention_mask, y_train, test_size=0.1, shuffle=True, random_state=0)

early_stopping = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

model.fit([train_input_ids, train_attention_mask], y_train, validation_data=([valid_input_ids, valid_attention_mask], y_valid), batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, callbacks=[early_stopping])

y_pred = model.predict([test_input_ids, test_attention_mask], batch_size=BATCH_SIZE, verbose=1)

submission = pd.read_csv('./google-quest/sample_submission.csv')
submission[labels] = y_pred
submission['qa_id'] = test['qa_id']
submission.to_csv('./google-quest/lstm_submission.csv', index=False)