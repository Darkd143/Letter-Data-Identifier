import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, scale
from keras.utils import to_categorical
import keras
from keras import activations
from keras.layers import Input, Dense, Dropout, Flatten, Activation
from keras.models import Model, Sequential


# def LetterSplitter(Letter, X, Y, per):
#     LetterArray = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
#                    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
#     x = LetterArray.index(Letter)

#     NewX = []
#     NewY = []
#     LX = []
#     LY = []

#     index = 0
#     i = 1
#     while (index < len(X)):
#         if (Y[index] == x):
#             if (i < per):
#                 LX.append(X[index])
#                 LY.append(Y[index])
#                 i += 1
#             else:
#                 NewX.append(X[index])
#                 NewY.append(Y[index])
#                 i = 1
#         else:
#             NewX.append(X[index])
#             NewY.append(Y[index])
#         index += 1

#     return NewX, LX, NewY, LY


# def ModelDefine(identifier):
#     model = None

#     if (identifier == "xgb"):
#         model = XGBClassifier(silent=False,
#                               scale_pos_weight=1,
#                               learning_rate=0.01,
#                               colsample_bytree=0.4,
#                               subsample=0.8,
#                               #   objective='binary:logistic',
#                               objective='multi:softprob',
#                               n_estimators=1000,
#                               reg_alpha=0.3,
#                               max_depth=4,
#                               gamma=10)
#     elif (identifier == "rf"):
#         model = RandomForestClassifier(n_estimators=100)
#     elif (identifier == "nn"):
#         model = None
#     else:
#         model = None

#     return model


# def GetPreferences():
#     Letter = None
#     Model = None
#     Per = 1

#     Letter = input('Enter Letter: ')[0]
#     Letter = Letter.upper()

#     Model = input('Enter Model: (xgb, rf, nn) ')
#     Model = Model.lower()

#     print('For every x ', Letter, '\'s, keep 1 ',
#           Letter, ' (x must be > 1): ', sep='', end='')
#     Per = int(input())

#     return Letter, Model, Per

def LetterParse(letter):
    letter_data = data_end
    letter_data = letter_data[letter_data.letter == letter]
    print(letter_data)
    information = letter_data.iloc[:, 1:]
    labels = letter_data.iloc[:, 0]

    information = MinMaxScaler().fit_transform(information)
    labels = LabelEncoder().fit_transform(labels)
    labels = to_categorical(labels, classes)
    useless, letter_test, useless_2, letter_labels = train_test_split(
        information, labels, test_size=1, random_state=0)

    return letter_test, letter_labels


# LetterLabel, ModelLabel, Per = GetPreferences()

path = 'letter-recognition.csv'

data = pd.read_csv(path)

data_start = data[:15000]
data_end = data[15000:]

X = data_start.iloc[:, 1:]

Y = data_start.iloc[:, 0]

classes = len(np.unique(Y))

X = MinMaxScaler().fit_transform(X)

Y = LabelEncoder().fit_transform(Y)

Y = to_categorical(Y, classes)
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

train, control_X, train_labels, control_Y = train_test_split(
    X, Y, test_size=0.2, random_state=0)

# }

# X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 0)

# NEURAL NETWORK
dim = X.shape[1]
model = Sequential()
model.add(Dense(300, activation='relu', input_shape=(dim,)))
model.add(Dropout(0.2))
model.add(Dense(150, name="feature", activation='relu'))
model.add(Dense(classes, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])

# XGBoost

# model = XGBClassifier(silent=False,
#                       scale_pos_weight=1,
#                       learning_rate=0.01,
#                       colsample_bytree=0.4,
#                       subsample=0.8,
#                       #   objective='binary:logistic',
#                       objective='multi:softprob',
#                       n_estimators=1000,
#                       reg_alpha=0.3,
#                       max_depth=4,
#                       gamma=10)

model.fit(train, train_labels, batch_size=2096, epochs=150,
          verbose=1, validation_data=(control_X, control_Y))


# model.fit(train,train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(A_test,A_labels))
# model.fit(train,train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(B_test,B_labels))
# model.fit(train,train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(C_test,C_labels))
# model.fit(train,train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(D_test,D_labels))
# model.fit(Ntrain,train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(E_test,E_labels))
# model.fit(train,train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(F_test,F_labels))
# model.fit(train,train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(G_test,G_labels))
# model.fit(train,train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(H_test,H_labels))
# model.fit(train,train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(I_test,I_labels))
# model.fit(train,train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(J_test,J_labels))
# model.fit(train,train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(K_test,K_labels))
# model.fit(train,train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(L_test,L_labels))
# model.fit(train,train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(M_test,M_labels))
# model.fit(train,train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(N_test,N_labels))
# model.fit(train,train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(O_test,O_labels))
# model.fit(train,train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(P_test,P_labels))
# model.fit(train,train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(Q_test,Q_labels))
# model.fit(train,train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(R_test,R_labels))
# model.fit(train,train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(S_test,S_labels))
# model.fit(train,train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(T_test,T_labels))
# model.fit(train,train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(U_test,U_labels))
# model.fit(train,train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(V_test,V_labels))
# model.fit(train,train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(W_test,W_labels))
# model.fit(train,train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(X_test,X_labels))
# model.fit(train,train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(Y_test,Y_labels))
# model.fit(train,train_labels,batch_size=2096, epochs=150,verbose=1,validation_data=(Z_test,Z_labels))

# end

# path = 'letter-recognition.csv'
# data = pd.read_csv(path)


# X = data.iloc[:, 1:]
# Y = data.iloc[:, 0]

# classes = len(np.unique(Y))

# X = MinMaxScaler().fit_transform(X)
# Y = LabelEncoder().fit_transform(Y)

# train_x, test_x, train_y, test_y = LetterSplitter(LetterLabel, X, Y, Per)

# ttrain_x, ttest_x, ttrain_y, ttest_y = train_test_split(X, Y,
#                                                         random_state=42,
#                                                         train_size=0.8,
#                                                         test_size=0.2,
#                                                         shuffle=True)

# # https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b6/sklearn/model_selection/_split.py#L2349

# print(ttrain_x)
# print(np.shape(ttrain_x))

# print("VS")

# print(train_x)
# print(np.shape(train_x))

# # print(np.shape(train_x), ' train X ', np.shape(ttrain_x))
# # print(np.shape(train_y), ' train Y ', np.shape(ttrain_y))
# # print(np.shape(test_x), ' test X ', np.shape(ttest_x))
# # print(np.shape(test_y), ' test Y ', np.shape(ttest_y))

# model = ModelDefine(ModelLabel)

# print(len(train_x))
# print(len(train_y))

# model.fit(train_x, train_y)

# predictions = model.predict(test_x)

# print(metrics.classification_report(test_y, predictions))
# print(accuracy_score(test_y, predictions))
