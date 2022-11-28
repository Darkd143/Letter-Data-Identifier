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

# A_test, A_labels = LetterParse('A')
# B_test, B_labels = LetterParse('B')
# C_test, C_labels = LetterParse('C')
# D_test, D_labels = LetterParse('D')
# E_test, E_labels = LetterParse('E')
# F_test, F_labels = LetterParse('F')
# G_test, G_labels = LetterParse('G')
# H_test, H_labels = LetterParse('H')
# I_test, I_labels = LetterParse('I')
# J_test, J_labels = LetterParse('J')
# K_test, K_labels = LetterParse('K')
# L_test, L_labels = LetterParse('L')
# M_test, M_labels = LetterParse('M')
# N_test, N_labels = LetterParse('N')
# O_test, O_labels = LetterParse('O')
# P_test, P_labels = LetterParse('P')
# Q_test, Q_labels = LetterParse('Q')
# R_test, R_labels = LetterParse('R')
# S_test, S_labels = LetterParse('S')
# T_test, T_labels = LetterParse('T')
# U_test, U_labels = LetterParse('U')
# V_test, V_labels = LetterParse('V')
# W_test, W_labels = LetterParse('W')
# X_test, X_labels = LetterParse('X')
# Y_test, Y_labels = LetterParse('Y')
# Z_test, Z_labels = LetterParse('Z')

NN_train, control_X, NN_train_labels, control_Y = train_test_split(X_nn,Y_nn, test_size = 0.2, random_state = 0)

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

model.fit(NN_train,NN_train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(control_X,control_Y))
# model.fit(NN_train,NN_train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(A_test,A_labels))
# model.fit(NN_train,NN_train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(B_test,B_labels))
# model.fit(NN_train,NN_train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(C_test,C_labels))
# model.fit(NN_train,NN_train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(D_test,D_labels))
# model.fit(NN_train,NN_train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(E_test,E_labels))
# model.fit(NN_train,NN_train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(F_test,F_labels))
# model.fit(NN_train,NN_train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(G_test,G_labels))
# model.fit(NN_train,NN_train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(H_test,H_labels))
# model.fit(NN_train,NN_train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(I_test,I_labels))
# model.fit(NN_train,NN_train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(J_test,J_labels))
# model.fit(NN_train,NN_train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(K_test,K_labels))
# model.fit(NN_train,NN_train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(L_test,L_labels))
# model.fit(NN_train,NN_train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(M_test,M_labels))
# model.fit(NN_train,NN_train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(N_test,N_labels))
# model.fit(NN_train,NN_train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(O_test,O_labels))
# model.fit(NN_train,NN_train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(P_test,P_labels))
# model.fit(NN_train,NN_train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(Q_test,Q_labels))
# model.fit(NN_train,NN_train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(R_test,R_labels))
# model.fit(NN_train,NN_train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(S_test,S_labels))
# model.fit(NN_train,NN_train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(T_test,T_labels))
# model.fit(NN_train,NN_train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(U_test,U_labels))
# model.fit(NN_train,NN_train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(V_test,V_labels))
# model.fit(NN_train,NN_train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(W_test,W_labels))
# model.fit(NN_train,NN_train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(X_test,X_labels))
# model.fit(NN_train,NN_train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(Y_test,Y_labels))
# model.fit(NN_train,NN_train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(Z_test,Z_labels))