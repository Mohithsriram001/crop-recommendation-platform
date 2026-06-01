import pandas as pd

# Load the dataset
df = pd.read_csv("data/crop_recommendation.csv")

# Display first 5 rows
print("First 5 rows of dataset:")
print(df.head())

# Display dataset shape
print("\nDataset shape:")
print(df.shape)

# Display column names
print("\nColumn names:")
print(df.columns)

# Check missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Check crop names
print("\nCrop names:")
print(df["label"].unique())

# Count how many records for each crop
print("\nCrop count:")
print(df["label"].value_counts())

# Separate input features and output target
X = df.drop("label", axis=1)
y = df["label"]

print("\nInput features X:")
print(X.head())

print("\nOutput target y:")
print(y.head())

print("\nX shape:")
print(X.shape)

print("\ny shape:")
print(y.shape)

from sklearn.model_selection import train_test_split

# Split dataset into training data and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining input shape:")
print(X_train.shape)

print("\nTesting input shape:")
print(X_test.shape)

print("\nTraining output shape:")
print(y_train.shape)

print("\nTesting output shape:")
print(y_test.shape)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Create KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)

# Train KNN model using training data
knn_model.fit(X_train, y_train)

# Predict using testing data
knn_pred = knn_model.predict(X_test)

# Calculate accuracy
knn_accuracy = accuracy_score(y_test, knn_pred)

print("\nKNN Model Accuracy:")
print(knn_accuracy)

from sklearn.tree import DecisionTreeClassifier

# Create Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)

# Train Decision Tree model
dt_model.fit(X_train, y_train)

# Predict using testing data
dt_pred = dt_model.predict(X_test)

# Calculate accuracy
dt_accuracy = accuracy_score(y_test, dt_pred)

print("\nDecision Tree Model Accuracy:")
print(dt_accuracy)

from sklearn.ensemble import RandomForestClassifier

# Create Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Train Random Forest model
rf_model.fit(X_train, y_train)

# Predict using testing data
rf_pred = rf_model.predict(X_test)

# Calculate accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Model Accuracy:")
print(rf_accuracy)

from sklearn.naive_bayes import GaussianNB

# Create Naive Bayes model
nb_model = GaussianNB()

# Train Naive Bayes model
nb_model.fit(X_train, y_train)

# Predict using testing data
nb_pred = nb_model.predict(X_test)

# Calculate accuracy
nb_accuracy = accuracy_score(y_test, nb_pred)

print("\nNaive Bayes Model Accuracy:")
print(nb_accuracy)

from sklearn.svm import SVC

# Create SVM model
svm_model = SVC()

# Train SVM model
svm_model.fit(X_train, y_train)

# Predict using testing data
svm_pred = svm_model.predict(X_test)

# Calculate accuracy
svm_accuracy = accuracy_score(y_test, svm_pred)

print("\nSVM Model Accuracy:")
print(svm_accuracy)

# Store all model accuracies in a dictionary
accuracies = {
    "KNN": knn_accuracy,
    "Decision Tree": dt_accuracy,
    "Random Forest": rf_accuracy,
    "Naive Bayes": nb_accuracy,
    "SVM": svm_accuracy
}

print("\nModel Accuracy Comparison:")
for model_name, accuracy in accuracies.items():
    print(model_name, ":", accuracy)

# Find best model
best_model_name = max(accuracies, key=accuracies.get)
best_accuracy = accuracies[best_model_name]

print("\nBest Model:")
print(best_model_name)

print("\nBest Model Accuracy:")
print(best_accuracy)

import joblib

# Store trained models in a dictionary
trained_models = {
    "KNN": knn_model,
    "Decision Tree": dt_model,
    "Random Forest": rf_model,
    "Naive Bayes": nb_model,
    "SVM": svm_model
}

# Select the best trained model
best_model = trained_models[best_model_name]

# Save the best model
joblib.dump(best_model, "models/best_crop_model.joblib")

print("\nBest model saved successfully!")
print("Saved model name: models/best_crop_model.joblib")