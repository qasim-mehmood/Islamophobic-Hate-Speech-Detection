# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 12:30:41 2020

@author: Qasim
"""
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Bidirectional
from keras.layers import Dense, Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split
import pandas as pd
import nltk
import string
import re
from nltk.corpus import stopwords
#nltk.download('punkt')
#nltk.download('stopwords')

############## Loading Data from CSV #####################
df = pd.read_csv(r'D:/MSDS/Research/py/aaa_all.csv')
df.columns=['Labels','Data']
#print(df.head())

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
from keras.preprocessing.sequence import pad_sequences
tk = Tokenizer()
tk.fit_on_texts(x)
index_list = tk.texts_to_sequences((x))

print('One hot vectors of the strings')
print(index_list[3])


x = pad_sequences(index_list, maxlen=maxlen)

print('One hot representation of the strings with padding')
print (x[3])
print('\nTotal number of str len and label length is: %s' % str(len(x)) +'  and\t'+ str(len(y)))

########################## train_test Split #################################
x_train, x_test, y_train, y_test_actual = train_test_split(x, y, test_size=0.20, random_state=40)
print('Printed Values of Y: ')
print(y_test_actual)
print('Type of Y:')
print(type(y_test_actual))

print(x_train.shape) 
print(y_train.shape) 

print(x_test.shape) 
print(y_test_actual.shape)

#################### Apply Deep Learning Model on the Tweets Data ####################
#model = Sequential()
#model.add(Embedding(max_features, 300, input_length=maxlen))
#model.add(Bidirectional(SimpleRNN(32, return_sequences=True)))
#model.add(Bidirectional(SimpleRNN(32, return_sequences=True)))
#model.add(Bidirectional(SimpleRNN(32, return_sequences=True)))
#model.add(Bidirectional(SimpleRNN(32)))
#model.add(Dense(1, activation='sigmoid'))
#model.summary()

model = Sequential()
model.add(Embedding(max_features, 300, input_length=maxlen))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(SimpleRNN(32, return_sequences=True)))
model.add(Bidirectional(SimpleRNN(32, return_sequences=True)))
model.add(Bidirectional(SimpleRNN(32, return_sequences=True)))
model.add(Bidirectional(SimpleRNN(32)))
model.add(Dense(1, activation='sigmoid'))
model.summary()


###################         Compile and fit the model         ########################
model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['acc'])
history = model.fit(x_train, y_train,
                    epochs=20,
                    batch_size=128,
                    validation_split=0.2)
print('Printed Values of Y: ')
print(y_test_actual)
print('Type of Y:')
print(type(y_test_actual))

#evluate the model
print('accuracy of the model')
scores = model.evaluate(x_test, y_test_actual)
print("accuracy: %.2f%%" %(scores[1]*100))

##############################      Save the Trained Model      #############################
#model_json = model.to_json()
#open('B_RNNTrg.json','w').write(model_json)
#model.save_weights('B_RNNweights.h5', overwrite=True)

###################         Compile and fit the model         ########################
from sklearn.metrics import accuracy_score

import numpy as np
pred = model.predict(x_test)
predict=np.round(pred)
pred=predict.astype(int)
pred=pred.reshape(len(pred),)
pred=list(pred)
y_test=list(y_test_actual)
print("Model Accuracy %.2f%%" %(accuracy_score(y_test,pred)*100))

############ Graphical View of Each Epoch       ################ 
import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

plt.plot(history.history['loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
#print(y_test)
#y_test_actual=list(y_test_actual)
#print(y_test_actual)