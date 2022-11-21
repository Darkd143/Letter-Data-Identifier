import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
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
# # convert labels into one-hot encode respresntation. if label=2 then [0 0 1....0]
print(Y)
# Y = to_categorical(Y, classes)


train_x, test_x, train_y, test_y = train_test_split(
    X, Y, test_size=0.3, random_state=0)

clf = RandomForestClassifier(n_estimators=100)

clf.fit(train_x, train_y)

predictions = clf.predict(test_x)

print(accuracy_score(test_y, predictions))
