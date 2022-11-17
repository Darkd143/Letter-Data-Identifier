import pandas as pd
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

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
