# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, Embedding, Flatten
from keras.layers import SpatialDropout1D
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.models import Sequential
print(os.listdir())

train = pd.read_csv(r'C:\Users\hp\Desktop\Dataset\train.csv')
test = pd.read_csv(r'C:\Users\hp\Desktop\Dataset\test.csv')
sub = pd.read_csv(r'C:\Users\hp\Desktop\Dataset\sample.csv')


X_train = train['Review'].apply(lambda x: x.lower())
X_test = test['Review'].apply(lambda x: x.lower())

maxlen = 256
max_features = 10000

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train))
X_train = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen=maxlen)

y = to_categorical(train['Score'].values)

model = Sequential()

# Input / Embdedding
model.add(Embedding(max_features, 150, input_length=maxlen))

# CNN
model.add(SpatialDropout1D(0.2))

model.add(Conv1D(32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

# Output layer
model.add(Dense(5, activation='sigmoid'))

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size=0.2, random_state=0)

epochs = 5
batch_size = 32

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1)


y_preds = model.predict(X_test, batch_size = 32, verbose = 1)




df_results_csv = pd.DataFrame({'Id':test['Id'],
                               'Score':np.argmax(y_preds,axis=1)})
 
df_results_csv.to_csv("sample.csv", index=False)


