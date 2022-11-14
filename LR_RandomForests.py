# Split the data into training and testing sets
# Instantiate model with 1000 decision trees
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


features = pd.read_csv('letter-recognition.csv')


# One-hot encode the data using pandas get_dummies
# features = pd.get_dummies(features)

lab = LabelEncoder()

features['letter'] = lab.fit_transform(features['letter'])

# Labels are the values we want to predict
labels = np.array(features['letter'])  # Remove the labels from the features
# axis 1 refers to the columns
# Saving feature names for later use
features = features.drop('letter', axis=1)
feature_list = list(features.columns)  # Convert to numpy array
features = np.array(features)

# Using Skicit-learn to split data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.25, random_state=42)

# print('Training Features Shape:', train_features.shape)
# print('Training Labels Shape:', train_labels.shape)
# print('Testing Features Shape:', test_features.shape)
# print('Testing Labels Shape:', test_labels.shape)

# The baseline predictions are the historical averages
# Baseline errors, and display average baseline error
# baseline_preds = test_features[:, feature_list.index('average')]
# baseline_errors = abs(baseline_preds - test_labels)

# print('Average baseline error: ', round(np.mean(baseline_errors), 2))

# Import the model we are using
# Train the model on training data
rf = RandomForestRegressor(n_estimators=1000, random_state=42)
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)  # Calculate the absolute errors
# Print out the mean absolute error (mae)
errors = abs(predictions - test_labels)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)  # Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
