# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 12:30:41 2020

@author: Qasim
"""
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, LSTM, Bidirectional, GRU
from keras.layers import Dense, Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split
import pandas as pd
import nltk
import string
import re

from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

############## Loading Data from CSV #####################
df = pd.read_csv(r'D:/MSDS/ResearchThesis/py/aaa_all.csv')
df.columns=['Labels','Data']
print(df.head())

########### Converting Data Column into lower case ###################
df['Data']=df['Data'].astype(str).str.lower()
df.head()
print('Converted Tweets into lower Case')
print(df)

########### Removing Digits from Data Column ###################
df['Data']=df['Data'].str.replace('\d+', '')
print(df.head())

########### Removing Puntuations from Data Column ###################
from string import punctuation
df['Data'] =df['Data'].apply(lambda x:''.join([i for i in x if i not in string.punctuation]))
print(df.head())

########### Removing Stop Words from Data Column ###################
def stopword_removal(sentence):
  tokenized_reports_no_stopwords = []
  for words in sentence:
    words=words.split(' ')
    new_term=[]
    for word in words:
      if not word in stopwords.words('english'):
        new_term.append(word)
    tokenized_reports_no_stopwords.append(new_term)
  return tokenized_reports_no_stopwords
df['Data']=stopword_removal(df['Data'])
print(len(df['Data']))
print(df.head())

################# Print the length of Labels and Tweets ##################
print('%d - Length of Labels' % len(df['Labels']))
print('%d - Length of Tweets' % len(df['Data']))

################# Set the max_features and max length of Tweets ##################
max_features = 5000
maxlen = 150
print(df.head())

################# Print the Tweets in original sizes and shape ##################
dataset = df.values
#print(dataset[:,0])
#print(dataset[:,1])

######### Train, Test, Split Data, X and Y, Set maxlen and pad 0 #########
x = dataset[:,1]
y = dataset[:,0]

############### One Hot Representation of String ##########################
from keras.preprocessing.text import Tokenizer

from keras_preprocessing.sequence import pad_sequences
# from keras.preprocessing.sequence import pad_sequences
tk = Tokenizer()
tk.fit_on_texts(x)
index_list = tk.texts_to_sequences((x))

print('One hot vectors of the strings')
print(index_list[50])


x = pad_sequences(index_list, maxlen=maxlen)

print('One hot representation of the strings with padding')
print (x[50])
print('\nTotal number of str len and label length is: %s' % str(len(x)) +'  and\t'+ str(len(y)))


x_train, x_test, y_train, y_test_actual = train_test_split(x, y, test_size=0.20, random_state=40)

print(x_train.shape) 
print(y_train.shape) 

print(x_test.shape) 
print(y_test_actual.shape)

print(df)
#################### Apply Deep Learning Model on the Tweets Data ####################
model = Sequential()
model.add(Embedding(max_features, 300, input_length=maxlen))
print('Word Embeddings representation of the strings with padding')
print (x[10])

model.add(Conv1D(filters=32, kernel_size=3,  padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=64, kernel_size=3,  padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=64, kernel_size=3,  padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=128, kernel_size=3,  padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=1))

model.add(Conv1D(filters=128, kernel_size=3,  padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=1))

model.add(Conv1D(filters=256, kernel_size=2,  padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=1))

model.add(Conv1D(filters=512, kernel_size=2,  padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=1))
model.summary()



model.add(Bidirectional(LSTM(32, return_sequences=True)))
#model.add(Bidirectional(LSTM(32, return_sequences=True)))
#model.add(Bidirectional(LSTM(32, return_sequences=True)))
#model.add(Bidirectional(LSTM(32, return_sequences=True)))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#############Conversion of NumPy array to a Tensor ##############

import numpy as np
x_trainn = np.asarray(x_train).astype(np.float32)
y_trainn = np.asarray(y_train).astype(np.float32)

x_testt = np.asarray(x_test).astype(np.float32)
y_test_actuall = np.asarray(y_test_actual).astype(np.float32)


##################         Compile and fit the model         ########################
model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['acc'])
history = model.fit(x_trainn, y_trainn,
                    epochs=1,
                    batch_size=128,
                    verbose=1)
                    # validation_split=0.2)

########evluate the model
print('accuracy of the model')
scores = model.evaluate(x_testt, y_test_actuall)
print("Eval Loss: %.2f%%" %(scores[0]*100))
print("Eval Accuracy: %.2f%%" %(scores[1]*100))

###############################      Save the Trained Model      #############################
model_json = model.to_json()
open('B_LSTM.json','w').write(model_json)
model.save_weights('B_LSTMweights.h5', overwrite=True)

####################         Compile and fit the model         ########################
from sklearn.metrics import accuracy_score

import numpy as np
pred = model.predict(x_test)
predict=np.round(pred)
pred=predict.astype(int)
pred=pred.reshape(len(pred),)
pred=list(pred)
y_test=list(y_test_actuall)
print("Model Accuracy %.2f%%" %(accuracy_score(y_test,pred)*100))
#
############# Graphical View of Each Epoch       ################ 
import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

plt.plot(history.history['loss'])
plt.title('Model loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
