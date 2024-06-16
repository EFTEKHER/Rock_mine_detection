import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
url = 'https://drive.google.com/uc?export=download&id=1_kjfa1qVrOc3EqA2FqB66GMXcGAH3IN8'
data_set = pd.read_csv(url, header=None)

# Preprocess dataset
label = LabelEncoder()
data_set[60] = label.fit_transform(data_set[60])

# Separate features and labels
X = data_set.drop(columns=60, axis=1)
Y = data_set[60]

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Save the trained model to a file
with open('model/logistic_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved successfully.")
