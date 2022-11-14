#Import Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv('letter-recognition.csv')


X = df.drop('letter',axis=1)
y = df['letter']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=66)
