from django.db import models
import tensorflow as tf
from konlpy.tag import Okt
from pykospacing import Spacing
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import re

import os

# Create your models here.


os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def get_model():
    lst = ['','갤럭시 노트8', '갤럭시 노트9', '갤럭시 노트10', '갤럭시 노트10플러스', '갤럭시 노트20',
           '갤럭시 노트20울트라', '갤럭시 퀀텀', '갤럭시 퀀텀2', '갤럭시 퀀텀3', '갤럭시 a12', '갤럭시 a21s', '갤럭시 a23',
           '갤럭시 a31', '갤럭시 a32', '갤럭시 a42', '갤럭시 a51', '갤럭시 a52s', '갤럭시 a53',
           '갤럭시 s8', '갤럭시 s8플러스', '갤럭시 s9', '갤럭시 s9플러스', '갤럭시 s10', '갤럭시 s10 5g',
           '갤럭시 s10플러스', '갤럭시 s10e', '갤럭시 s20', '갤럭시 s20울트라', '갤럭시 s20플러스',
           '갤럭시 s21', '갤럭시 s21울트라', '갤럭시 s21플러스', '갤럭시 s22', '갤럭시 s22울트라', '갤럭시 s22플러스',
           '갤럭시 z폴드', '갤럭시 z폴드2', '갤럭시 z폴드3', '갤럭시 z플립', '갤럭시 z플립 5g', '갤럭시 z플립3',
           '아이폰7', '아이폰7플러스', '아이폰8', '아이폰8플러스', '아이폰x',
           '아이폰xr', '아이폰xs', '아이폰xs맥스', '아이폰11', '아이폰11프로',
           '아이폰11프로맥스', '아이폰12', '아이폰12미니', '아이폰12프로', '아이폰12프로맥스',
           '아이폰13', '아이폰13미니', '아이폰13프로', '아이폰13프로맥스', '아이폰se1',
           '아이폰se2', '아이폰se3']
    return lst

def get_code():
    lst = ['4CTJ94N56', 'WH4B3B5TT', 'WYWYWQHYG', 'DQ8FBQ24D', 'GBCWM2W46', '25Z5QDCQ9', 'SQB27PW4S',
           'Y7YYYNDBM', 'DNZX9DXC8', '5W5C2ZG6F', 'S6QPW38FC', 'X5NKRWKR6', '7M2JGBQPR', 'FS5QS2CX4',
           '4TFCKZZH6', 'BX85S688B', 'MSWP3RFJ2', '35N4KT4XX', 'X2XDNT4Q3', 'SBXS58W2N', 'TGK825XD3',
           'DYG5XX9SH', '45PQK4RYW', 'W94KDGSGN', '754ZHYB5Q', '5YZ8R9KY4', '6QY6NHWSD', 'X8CP95KQW',
           '33JBC3CYF', 'YXP8KNCSC', 'QQWFSZBN9', 'W4KTMGZFZ', 'YRC75SG5H', 'XJK4TKD8N', '4HR487NRB',
           'BBWT57RC6', '65ZTY6GSN', 'TS66RQB45', 'B74WYGKY7', 'X92HWC3GH', 'N949HZM32', 'KK9MBWGKH',
           'R35B4YHF3', '9GN79N9BM', '57RX8MSJS', '988TSQ9W9', 'SZKKZQC5M', 'Z6W47NPMF', '9MHPM9C9R',
           'HGGP87GG9', 'YMRJ8FZQR', '3B6ZTCTRG', 'WGBDP2936', 'FT4BJPNDY', '42X492C9M', 'FZPNJZ7ZR',
           'JP6Y486ZH', 'KBFTY56ZJ', 'DZGKXN9NJ', 'Q6QXYH3RM', 'PNDCZHN9K', 'R3CDZ4RWH', 'Z63RJKXB3']
    return lst

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim  # d_model
        self.num_heads = num_heads

        assert embedding_dim % self.num_heads == 0

        self.projection_dim = embedding_dim // num_heads
        self.query_dense = tf.keras.layers.Dense(embedding_dim)
        self.key_dense = tf.keras.layers.Dense(embedding_dim)
        self.value_dense = tf.keras.layers.Dense(embedding_dim)
        self.dense = tf.keras.layers.Dense(embedding_dim)

    def scaled_dot_product_attention(self, query, key, value):
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)
        attention_weights = tf.nn.softmax(logits, axis=-1)
        output = tf.matmul(attention_weights, value)
        return output, attention_weights

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention, _ = self.scaled_dot_product_attention(query, key, value)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.embedding_dim))
        outputs = self.dense(concat_attention)
        return outputs

    def get_config(self):
        config = super(MultiHeadAttention, self)
        return {"embedding_dim": self.embedding_dim,
                "num_heads": self.num_heads}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads, dff, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dff = dff
        self.att = MultiHeadAttention(embedding_dim, num_heads)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(dff, activation="relu"),
             tf.keras.layers.Dense(embedding_dim), ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # Add & Norm
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)  # Add & Norm

    def get_config(self):
        config = super(TransformerBlock, self)
        return {"embedding_dim": self.embedding_dim,
                "num_heads": self.num_heads,
                "dff": self.dff}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len, vocab_size, embedding_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.token_emb = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.pos_emb = tf.keras.layers.Embedding(max_len, embedding_dim)

    def call(self, x):
        max_len = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=max_len, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super(TokenAndPositionEmbedding, self)

        return {"max_len": self.max_len,
                "vocab_size": self.vocab_size,
                "embedding_dim": embedding_dim}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


objects = {"MultiHeadAttention": MultiHeadAttention,
           "TransformerBlock": TransformerBlock,
           "TokenAndPositionEmbedding": TokenAndPositionEmbedding}
model = load_model('./main/model.h5', custom_objects=objects)

with open("./main/stopwords.pickle", "rb") as fr:
    stopwords = pickle.load(fr)
with open("./main/tokenizer.pickle", "rb") as fr:
    tokenizer = pickle.load(fr)

okt = Okt()
spacing = Spacing()

def predict_price(title, content):

    title = re.sub('[^0-9a-zA-Zㄱ-힗]', ' ', title)
    title = re.sub('[+]', '플러스', title)
    title = title.lower()
    content = re.sub('[^ㄱ-힗]', ' ', content)
    txt = title + ' ' + content

    lst = okt.morphs(txt, stem=True)
    print(lst)
    lst = [i for i in lst if i not in stopwords]
    print(lst, len(lst))
    lst = ' '.join(lst)
    print(lst)

    lst_token = tokenizer.texts_to_sequences([lst])
    lst_padded = pad_sequences(lst_token, 150)

    return model.predict(lst_padded)


objects = {"MultiHeadAttention": MultiHeadAttention,
           "TransformerBlock": TransformerBlock,
           "TokenAndPositionEmbedding": TokenAndPositionEmbedding}
model = load_model('./main/model.h5', custom_objects=objects)

with open("./main/stopwords.pickle", "rb") as fr:
    stopwords = pickle.load(fr)
with open("./main/tokenizer.pickle", "rb") as fr:
    tokenizer = pickle.load(fr)

okt = Okt()
spacing = Spacing()

def predict_price(title, content):

    title = re.sub('[^0-9a-zA-Zㄱ-힗]', ' ', title)
    title = re.sub('[+]', '플러스', title)
    title = title.lower()
    content = re.sub('[^ㄱ-힗]', ' ', content)
    content = content.split()
    rm = ['삼성', '플러스', '갤럭시', '노트', '폴드', '플립', '폴더', '애플', '아이폰']
    for txt in rm:
        for j in range(len(content) - 1, -1, -1):
            if txt in content[j]:
                try:
                    del content[j]
                except:
                    pass
    content = ' '.join(content)

    txt = title + ' ' + content

    lst = okt.morphs(txt, stem=True)
    lst = [i for i in lst if i not in stopwords]
    lst = ' '.join(lst)

    lst_token = tokenizer.texts_to_sequences([lst])
    lst_padded = pad_sequences(lst_token, 150)

    return model.predict(lst_padded)
