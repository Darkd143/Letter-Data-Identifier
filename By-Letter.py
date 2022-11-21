import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBClassifier


def LetterSplitter(Letter, X, Y, per):
    LetterArray = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                   'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    x = LetterArray.index(Letter)

    NewX = []
    NewY = []
    LX = []
    LY = []

    index = 0
    i = 0
    while (index < len(X)):
        if (Y[index] == x):
            if (i < per):
                LX.append(X[index])
                LY.append(Y[index])
                i += 1
            else:
                NewX.append(X[index])
                NewY.append(Y[index])
                i = 0
        else:
            NewX.append(X[index])
            NewY.append(Y[index])
        index += 1

    return NewX, NewY, LX, LY


def ModelDefine(identifier):
    model = None

    if (identifier == "xgb"):
        model = XGBClassifier(silent=False,
                              scale_pos_weight=1,
                              learning_rate=0.01,
                              colsample_bytree=0.4,
                              subsample=0.8,
                              #   objective='binary:logistic',
                              objective='multi:softprob',
                              n_estimators=1000,
                              reg_alpha=0.3,
                              max_depth=4,
                              gamma=10)
    elif (identifier == "rf"):
        model = RandomForestClassifier(n_estimators=100)
    elif (identifier == "nn"):
        model = None
    else:
        model = None

    return model


def GetPreferences():
    Letter = None
    Model = None

    Letter = input('Enter Letter: ')[0]
    Letter = Letter.upper()

    Model = input('Enter Model: (xgb, rf, nn) ')[0]
    Model = Letter.lower()

    return Letter, Model


LetterLabel, ModelLabel = GetPreferences()

path = 'letter-recognition.csv'
data = pd.read_csv(path)


X = data.iloc[:, 1:]
Y = data.iloc[:, 0]

classes = len(np.unique(Y))

X = MinMaxScaler().fit_transform(X)
Y = LabelEncoder().fit_transform(Y)

train_x, test_x, train_y, test_y = LetterSplitter(LetterLabel, X, Y, 5)

# print(test_y)

model = ModelDefine(ModelLabel)

model.fit(train_x, train_y)

predictions = model.predict(test_x)

print(metrics.classification_report(test_y, predictions))
print(accuracy_score(test_y, predictions))
