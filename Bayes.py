import re
import string
import sys
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from keras.preprocessing import text, sequence
np.set_printoptions(threshold=sys.maxsize)


def sent_list(docs,splitStr='__label__'):
    sent_analysis = []
    for i in range(1,len(docs)):
        text=str(lines[i])
        splitText=text.split(splitStr)
        secHalf=splitText[1]
        sentiment=secHalf[0]
        text=secHalf[2:len(secHalf)-1].lower()
        table=str.maketrans(' ',' ', string.punctuation)
        text.translate(table)
        if 'www.' in text or 'http:' in text or 'https:' in text or '.com' in text:
            text = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", text)
        sent_analysis.append([text,sentiment])
    return sent_analysis

        
f = open("data/reviews.ft.txt", "r")
lines = f.readlines()
sentiment_list=sent_list(lines[:50000],splitStr='__label__')
train_df = pd.DataFrame(sentiment_list, columns=['Text','Sentiment'])
print(train_df.head())
print(train_df.size)

X = train_df["Text"]
Y = train_df["Sentiment"]
x_l, x_test, y_l, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

count_vect = CountVectorizer(ngram_range = (1, 2), max_features = 20000)
count_vect = count_vect.fit(x_l.values)
x_l_wrds = count_vect.transform(x_l.values)
x_l_transformed = x_l_wrds.toarray()
x_test_wrds = count_vect.transform(x_test.values)
x_test_transformed = x_test_wrds.toarray()

clf = MultinomialNB()
clf.fit(x_l_transformed, y_l)

pred = clf.predict(x_test_transformed)
acc = accuracy_score(y_test, pred) 
print("acc is on test data:", acc)


mat = confusion_matrix(y_test, pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.savefig("Bayes_con.png") 

for input, prediction, label in zip(x_test_transformed, pred, y_test):
  if prediction != label:
    print(input, 'has been classified as ', prediction, 'and should be ', label) 
    print("######################")