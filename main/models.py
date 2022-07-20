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
    txt = df_input.loc[0,"title"] + ' ' + df_input.loc[0, "content"]
    lst = okt.morphs(txt,stem=True)
    lst = [i for i in lst if i not in stopwords]
    lst = ' '.join(lst)
    lst_token = tokenizer.texts_to_sequences([lst])
    lst_padded = pad_sequences(lst_token, 150)

    return model.predict(lst_padded)





