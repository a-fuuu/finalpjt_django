from django.db import models
import tensorflow as tf
from konlpy.tag import Okt
import pandas as pd
from pykospacing import Spacing
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import re

# Create your models here.
with open("./main/stopwords.pickle", "rb") as fr:
    stopwords = pickle.load(fr)
with open("./main/tokenizer.pickle", "rb") as fr:
    tokenizer = pickle.load(fr)
okt = Okt()
model = load_model('./main/model.h5')
spacing = Spacing()

def predict_price(title, content):

    def fix_remove_char(df):
        # df['content_origin'] = df['content'].copy()
        df.loc[0, 'title'] = re.sub('[^0-9a-zA-Zㄱ-힗]', ' ', df.loc[0, 'title'])
        df.loc[0, 'title'] = re.sub('[+]', '플러스', df.loc[0, 'title'])
        df.loc[0, 'title'] = df.loc[0, 'title'].lower()
        df.loc[0, 'content'] = re.sub('[^ㄱ-힗]', ' ', df.loc[0, 'content'])
        return df

    title = spacing(title)
    content = spacing(content)
    df_input = pd.DataFrame([title,content]).T.rename(columns={0: "title", 1: "content"})
    df_input = fix_remove_char(df_input)
    txt = df_input.loc[0, "title"] + ' ' + df_input.loc[0, "content"]
    lst = okt.morphs(txt, stem=True)
    lst = [i for i in lst if i not in stopwords]
    lst = ' '.join(lst)
    lst_token = tokenizer.texts_to_sequences([lst])
    lst_padded = pad_sequences(lst_token, 150)

    return model.predict(lst_padded)

def get_model():
    lst = ['아이폰7', '아이폰8플러스', '아이폰x', '갤럭시 a31', '아이폰se2', '아이폰8',
       '갤럭시 s8플러스', '아이폰11프로맥스', '아이폰12프로맥스', '갤럭시 a12', '아이폰12', '아이폰xs',
       '아이폰11', '갤럭시 노트20울트라', '갤럭시 s10', '아이폰7플러스', '아이폰11프로', '아이폰12프로',
       '갤럭시 노트10플러스', '아이폰xs맥스', '아이폰xr', '아이폰12미니', '갤럭시 노트10',
       '갤럭시 9플러스', '갤럭시 s10 5g', '갤럭시 z플립', '갤럭시 노트9', '갤럭시 노트20',
       '갤럭시 s20울트라', '아이폰13프로맥스', '아이폰13', '아이폰13미니', '아이폰se1', '아이폰13프로',
       '갤럭시 z플립3', '갤럭시 노트8', '갤럭시 s9', '갤럭시 z플립 5g', '갤럭시 s21', '갤럭시 s8',
       '갤럭시 s20플러스', '갤럭시 s21플러스', '갤럭시 s10e', '갤럭시 a51', '갤럭시 z폴드2',
       '갤럭시 s20', '갤럭시 a42', '갤럭시 s10플러스', '갤럭시 a32', '갤럭시 s21울트라',
       '갤럭시 z폴드3', '갤럭시 a21s', '갤럭시 a52s', '갤럭시 퀀텀2', '갤럭시 z폴드',
       '갤럭시 a53', '갤럭시 s22울트라', '갤럭시 s22플러스', '갤럭시 s22', '아이폰se3',
       '갤럭시 퀀텀', '갤럭시 a23', '갤럭시 퀀텀3']
    return lst





