import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,scale
import pandas as pd
from keras.utils import to_categorical
import keras
from keras import activations
from keras.layers import Input, Dense, Dropout, Flatten, Activation
from keras.models import Model, Sequential
from sklearn.metrics import accuracy_score

def LetterParse(letter):
    letter_data = data_end
    letter_data = letter_data[letter_data.letter == letter]
    print(letter_data)
    information=letter_data.iloc[:,1:]
    labels=letter_data.iloc[:,0]

    information = MinMaxScaler().fit_transform(information)
    labels = LabelEncoder().fit_transform(labels)
    labels = to_categorical(labels,classes)
    useless, letter_test, useless_2, letter_labels = train_test_split(information,labels, test_size = 1, random_state = 0)

    return letter_test, letter_labels


path = 'letter-recognition.csv'

data=pd.read_csv(path)

data_start = data[:15000]
data_end = data[15000:]

X_nn=data_start.iloc[:,1:]

Y_nn=data_start.iloc[:,0]

classes = len(np.unique(Y_nn))

X_nn = MinMaxScaler().fit_transform(X_nn)

Y_nn = LabelEncoder().fit_transform(Y_nn)

Y_nn = to_categorical(Y_nn,classes)
# print(X.shape,Y.shape)

# testing {

A_test, A_labels = LetterParse('A')

X_train, X_test, Y_train, Y_test = train_test_split(X_nn,Y_nn, test_size = 0.2, random_state = 0)

# }

# X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 0)

dim = X_nn.shape[1]

model = Sequential()

model.add(Dense(300,activation='relu',input_shape=(dim,)))

model.add(Dropout(0.2))

model.add(Dense(150,name="feature",activation='relu'))

model.add(Dense(classes,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(lr=0.01),metrics=['accuracy'])

model.fit(X_train,Y_train,batch_size=2096, epochs=150,verbose=1,validation_data=(X_test,Y_test))
model.fit(X_train,Y_train,batch_size=2096, epochs=150,verbose=1,validation_data=(A_test,A_labels))