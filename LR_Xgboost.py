import numpy as np
import pandas as pd
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# from keras.utils import to_categorical

######## Load Dataset ##########
path = 'letter-recognition.csv'
# read .txt file using pandas.
data = pd.read_csv(path)
# last 16 colomnn as a inputs
X = data.iloc[:, 1:]
# first column as a labels.
Y = data.iloc[:, 0]
# total number of classes
classes = len(np.unique(Y))
# convert inputs in the range of 0 to 1.
X = MinMaxScaler().fit_transform(X)
# convert letter type labels into numeric type( 0 to 25)
Y = LabelEncoder().fit_transform(Y)
# convert labels into one-hot encode respresntation. if label=2 then [0 0 1....0]
# Y = to_categorical(Y, classes)


train_x, test_x, train_y, test_y = train_test_split(
    X, Y, test_size=0.3, random_state=0)

model = XGBClassifier(silent=False,
                      scale_pos_weight=1,
                      learning_rate=0.01,
                      colsample_bytree=0.4,
                      subsample=0.8,
                      #   objective='binary:logistic',
                      objective='multi:softprob',
                      n_estimators=1000,
                      reg_alpha=0.5,
                      max_depth=10,
                      gamma=3)

# model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#                       colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.1,
#                       max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,
#                       n_estimators=100, n_jobs=1, nthread=None,
#                       objective='multi:softprob', random_state=0, reg_alpha=0,
#                       reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
#                       subsample=1, verbosity=1)

model.fit(train_x, train_y)


predictions = model.predict(test_x)

print(metrics.classification_report(test_y, predictions))
# print(metrics.confusion_matrix(test_y, predictions))

print(accuracy_score(test_y, predictions))
