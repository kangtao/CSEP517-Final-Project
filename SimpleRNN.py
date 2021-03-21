import numpy as np 
import pandas as pd
import re
import string
import nltk
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

from sklearn.feature_extraction import text
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score   
from sklearn.metrics import confusion_matrix

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,SimpleRNN, LSTM,SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 


def remove_url(text):
     url=re.compile(r"https?://\S+|www\.\S+")
     return url.sub(r" ",text)

def remove_html(text):
  cleanr = re.compile('<.*?>')
  return cleanr.sub(r" ",text)

def remove_num(texts):
   output = re.sub(r'\d+', '', texts)
   return output

def remove_punc(text):
   table=str.maketrans(' ',' ',string.punctuation)
   return text.translate(table)


files = open("data/reviews.ft.txt", "r")
files = files.readlines()

num_train = 40000
num_test = 10000 

train_file = [x for x in files[:num_train]]
test_file = [x for x in files[num_train:num_test + num_train]]
train_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in train_file]
train_sentences = [x.split(' ', 1)[1][:-1].lower() for x in train_file]
test_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in test_file]
test_sentences = [x.split(' ', 1)[1][:-1].lower() for x in test_file]

train = pd.DataFrame({'text':train_sentences,'label':train_labels})
test = pd.DataFrame({'text':test_sentences,'label':test_labels})
train.describe()
test.describe()

train['text']=train.text.map(lambda x:remove_url(x))
train['text']=train.text.map(lambda x:remove_html(x))
train['text']=train.text.map(lambda x:remove_punc(x))
train['text']=train['text'].map(remove_num)
test['text']=test.text.map(lambda x:remove_url(x))
test['text']=test.text.map(lambda x:remove_html(x))
test['text']=test.text.map(lambda x:remove_punc(x))
test['text']=test['text'].map(remove_num)

max_length=100
vocab_size=20000
embedding_dim=64
trunc_type="post"
oov_tok="<OOV>"
padding_type="post"

tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_tok)
tokenizer.fit_on_texts(train['text'])
word_index = tokenizer.word_index
training_sequences = tokenizer.texts_to_sequences(train['text'])
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
testing_sequences = tokenizer.texts_to_sequences(test['text'])
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
model.add(SpatialDropout1D(0.1))
model.add(SimpleRNN(256, dropout=0.1))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()

adam=Adam(lr=0.0001)
model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])
history=model.fit(training_padded,train['label'], epochs=15, batch_size=256,verbose = 1,validation_data=(testing_padded, test['label']))


pred = model.predict(testing_padded)
mat = confusion_matrix(test['label'], np.where(pred > 0.5, 1, 0))
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.savefig("RNN_con.png") 
