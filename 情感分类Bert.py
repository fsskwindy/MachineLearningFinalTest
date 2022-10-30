# encoding=gbk
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.utils import to_categorical
import os
from transformers import *

# 参考https://www.kaggle.com/code/akensert/quest-bert-base-tf2-0/notebook
# print(tf.__version__)  # 2.10.0

# https://huggingface.co/bert-base-chinese下载预训练参数
TRAIN_PATH = './data/train_dataset/'
TEST_PATH = './data/test_dataset/'
EXAMPLE_PATH = './data/submit_example/'
BERT_PATH = './bert_base_chinese/'
MAX_SEQUENCE_LENGTH = 200

df_train = pd.read_csv(TRAIN_PATH + 'nCoV_100k_train.labled.csv', engine='python')
df_train = df_train[df_train['情感倾向'].isin(['-1', '0', '1'])]
df_test = pd.read_csv(TEST_PATH + 'nCov_10k_test.csv', engine='python')
df_sub = pd.read_csv(EXAMPLE_PATH + 'submit_example.csv')


# print('train shape =', df_train.shape)  # train shape = (99913, 7)
# print('test shape =', df_test.shape)  # test shape = (10000, 6)


# 对微博中文内容进行数据处理
def _convert_to_transformer_inputs(instance, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""

    inputs = tokenizer.encode_plus(instance, add_special_tokens=True, max_length=max_sequence_length,
                                   truncation_strategy='longest_first')
    input_ids = inputs["input_ids"]
    input_masks = [1] * len(input_ids)
    input_segments = inputs["token_type_ids"]
    padding_length = max_sequence_length - len(input_ids)
    padding_id = tokenizer.pad_token_id
    input_ids = input_ids + ([padding_id] * padding_length)
    input_masks = input_masks + ([0] * padding_length)
    input_segments = input_segments + ([0] * padding_length)

    return [input_ids, input_masks, input_segments]


# 将数据信息加入列表
def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
    input_ids, input_masks, input_segments = [], [], []
    for instance in tqdm(df[columns]):
        ids, masks, segments = \
            _convert_to_transformer_inputs(str(instance), tokenizer, max_sequence_length)

        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)

    return [np.asarray(input_ids, dtype=np.int32),
            np.asarray(input_masks, dtype=np.int32),
            np.asarray(input_segments, dtype=np.int32)]


tokenizer = BertTokenizer.from_pretrained(BERT_PATH + 'bert-base-chinese-vocab.txt')
inputs = compute_input_arrays(df_train, '微博中文内容', tokenizer, MAX_SEQUENCE_LENGTH)
test_inputs = compute_input_arrays(df_test, '微博中文内容', tokenizer, MAX_SEQUENCE_LENGTH)


def compute_output_arrays(df, columns):
    return np.asarray(df[columns].astype(int) + 1)


outputs = compute_output_arrays(df_train, '情感倾向')


# 构建模型
def create_model():
    # 输入层
    input_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    config = BertConfig.from_pretrained(BERT_PATH + 'config.json')
    config.output_hidden_states = False
    bert_model = TFBertModel.from_pretrained(BERT_PATH + 'tf_model.h5', config=config)
    embedding = bert_model(input_id, attention_mask=input_mask, token_type_ids=input_atn)[0]

    # 输出层
    x = tf.keras.layers.GlobalAveragePooling1D()(embedding)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(3, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=[input_id, input_mask, input_atn], outputs=x)

    return model


# 交叉验证
gkf = StratifiedKFold(n_splits=5).split(X=df_train['微博中文内容'].fillna('-1'),
                                        y=df_train['情感倾向'].fillna('-1'))

valid_preds = []
test_preds = []
for fold, (train_idx, valid_idx) in enumerate(gkf):
    train_inputs = [inputs[i][train_idx] for i in range(len(inputs))]
    train_outputs = to_categorical(outputs[train_idx])

    valid_inputs = [inputs[i][valid_idx] for i in range(len(inputs))]
    valid_outputs = to_categorical(outputs[valid_idx])

    K.clear_session()
    model = create_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc', 'mae'])
    model.fit(train_inputs, train_outputs,
              validation_data=[valid_inputs, valid_outputs], epochs=1, batch_size=128)
    valid_preds.append(model.predict(valid_inputs))
    test_preds.append(model.predict(test_inputs))

sub = np.average(test_preds, axis=0)
sub = np.argmax(sub, axis=1)  # 取出sub中每个元素最大值对应的索引
df_sub['y'] = sub - 1  # 减去之前多加的1
df_sub['id'] = df_sub['id'].apply(lambda x: str(x) + ' ')  # 将id以字符串形式写入
df_sub.to_csv('submit.csv', index=False, encoding='utf-8')
