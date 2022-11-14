# Split the data into training and testing sets
# Instantiate model with 1000 decision trees
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


df = pd.read_csv('letter-recognition.csv')


# One-hot encode the data using pandas get_dummies
# df = pd.get_dummies(df)

lab = LabelEncoder()

df['letter'] = lab.fit_transform(df['letter'])

# Labels are the values we want to predict
labels = np.array(df['letter'])  # Remove the labels from the df
# axis 1 refers to the columns
# Saving feature names for later use
# df = df.drop('letter', axis=1)
# feature_list = list(df.columns)  # Convert to numpy array
# df = np.array(df)
X = df.loc[:, df.columns != 'letter'].values
Y = df["letter"].values
train_x, test_x, train_y, test_y = train_test_split(X, Y,
                                                    random_state=42,
                                                    train_size=0.8,
                                                    test_size=0.2,
                                                    shuffle=True)
# Using Skicit-learn to split data into training and testing sets

# print('Training df Shape:', train_df.shape)
# print('Training Labels Shape:', train_labels.shape)
# print('Testing df Shape:', test_df.shape)
# print('Testing Labels Shape:', test_labels.shape)

# The baseline predictions are the historical averages
# Baseline errors, and display average baseline error
# baseline_preds = test_df[:, feature_list.index('average')]
# baseline_errors = abs(baseline_preds - test_labels)

# print('Average baseline error: ', round(np.mean(baseline_errors), 2))

# Import the model we are using
# Train the model on training data
rf = RandomForestRegressor(n_estimators=1000, random_state=42)
rf.fit(train_x, train_y)

# Use the forest's predict method on the test data
predictions = rf.predict(test_x)  # Calculate the absolute errors
# Print out the mean absolute error (mae)


results_dict = {'Accuracy': 0, 'Precision': 0, 'Recall': 0}
############################STUDENT CODE GOES HERE#########################
# use the decision tree to predict on the test data
# compare the truth labels with the predicted labels for accuracy, precision, and recall
# store the results into the dataframe
results_dict['Accuracy'] = accuracy_score(test_y, predictions)
results_dict['Precision'] = precision_score(test_y, predictions)
results_dict['Recall'] = recall_score(test_y, predictions)
