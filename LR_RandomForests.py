import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

df = pd.read_csv('letter-recognition.csv')

dataframe = pd.DataFrame(df, columns=["letter", "xbox", "ybox", "width", "height", "onpix", "xbar",
                         "ybar", "x2bar", "y2bar", "xybar", "x2ybar", "xy2bar", "xedge", "xedgey", "yedge", "yedgex"])

lab = LabelEncoder()

df['letter'] = lab.fit_transform(df['letter'])

X = df.loc[:, df.columns != 'letter'].values
Y = df["letter"].values

train_x, test_x, train_y, test_y = train_test_split(X, Y,
                                                    random_state=42,
                                                    train_size=0.8,
                                                    test_size=0.2,
                                                    shuffle=True)

clf = RandomForestClassifier(n_estimators=100)

clf.fit(train_x, train_y)

predictions = clf.predict(test_x)

print(accuracy_score(test_y, predictions))
