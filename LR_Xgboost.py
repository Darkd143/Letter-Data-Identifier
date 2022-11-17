# Import Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

df = pd.read_csv('letter-recognition.csv')

df = pd.DataFrame(df, columns=["letter", "xbox", "ybox", "width", "height", "onpix", "xbar",
                         "ybar", "x2bar", "y2bar", "xybar", "x2ybar", "xy2bar", "xedge", "xedgey", "yedge", "yedgex"])

lab = LabelEncoder()

df['letter'] = lab.fit_transform(df['letter'])

X = df.loc[:, df.columns != 'letter'].values
Y = df["letter"].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

model = XGBClassifier(silent=False,
                      scale_pos_weight=1,
                      learning_rate=0.01,
                      colsample_bytree=0.4,
                      subsample=0.8,
                      objective='binary:logistic',
                      n_estimators=1000,
                      reg_alpha=0.3,
                      max_depth=4,
                      gamma=10)

model.fit(X_train, y_train)

expected_y = y_test
predicted_y = model.predict(X_test)

print(metrics.classification_report(expected_y, predicted_y))
print(metrics.confusion_matrix(expected_y, predicted_y))

plt.figure(figsize=(10, 10))
sns.regplot(expected_y, predicted_y, fit_reg=True, scatter_kws={"s": 100})

# X = df.drop('letter',axis=1)
# y = df['letter']
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=66)
