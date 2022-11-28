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


path = 'letter-recognition.csv'

data=pd.read_csv(path)

X=data.iloc[:,1:]

Y=data.iloc[:,0]

classes = len(np.unique(Y))

X = MinMaxScaler().fit_transform(X)

Y = LabelEncoder().fit_transform(Y)

Y = to_categorical(Y,classes)
# print(X.shape,Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 0)

dim = X.shape[1]

model = Sequential()

model.add(Dense(300,activation='relu',input_shape=(dim,)))

model.add(Dropout(0.2))

model.add(Dense(150,name="feature",activation='relu'))

model.add(Dense(classes,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(lr=0.01),metrics=['accuracy'])

model.fit(X_train,Y_train,batch_size=2096, epochs=150,verbose=1,validation_data=(X_test,Y_test))