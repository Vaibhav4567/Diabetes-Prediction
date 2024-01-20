import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

file_path=r"C:\Users\pc\Desktop\Mini Project Diabetes Pridiction Model\diabetes.csv"
diabetes_dataset = pd.read_csv(file_path) 

diabetes_dataset.head()

p = diabetes_dataset.hist(figsize = (20,20))

diabetes_dataset.shape

diabetes_dataset.describe()

diabetes_dataset['Outcome'].value_counts()

diabetes_dataset.groupby('Outcome').mean()

X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

print(X)

print(Y)

scaler = StandardScaler()

scaler.fit(X)

standardized_data = scaler.fit_transform(X)

print(standardized_data)

X = standardized_data
Y = diabetes_dataset['Outcome']

print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

classifier = svm.SVC(kernel='linear')

#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)

input_data = (5,166,72,19,175,25.8,0.587,51)
# Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')

from sklearn.neighbors import KNeighborsClassifier

knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors

# Train the KNN classifier
knn_classifier.fit(X_train, Y_train)

# Predictions on the training set
X_train_prediction = knn_classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score on the training data:', training_data_accuracy)

# Predictions on the testing set
X_test_prediction = knn_classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on the test data:', test_data_accuracy)

# Example input data for prediction
input_data = np.array([5, 166, 72, 19, 175, 25.8, 0.587, 51]).reshape(1, -1)

# Standardize the input data
std_data = scaler.transform(input_data)

# Make predictions using the KNN classifier
prediction = knn_classifier.predict(std_data)
print('Predicted class:', prediction[0])

if prediction[0] == 0:
    print('The person is not diabetic.')
else:
    print('The person is diabetic.')

from sklearn.ensemble import RandomForestClassifier

random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust the number of trees

# Train the Random Forest classifier
random_forest_classifier.fit(X_train, Y_train)

# Predictions on the training set
X_train_prediction = random_forest_classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score on the training data:', training_data_accuracy)

# Predictions on the testing set
X_test_prediction = random_forest_classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on the test data:', test_data_accuracy)

# Example input data for prediction
input_data = np.array([5, 166, 72, 19, 175, 25.8, 0.587, 51]).reshape(1, -1)

# Make predictions using the Random Forest classifier
prediction = random_forest_classifier.predict(input_data)
print('Predicted class:', prediction[0])

if prediction[0] == 0:
    print('The person is not diabetic.')
else:
    print('The person is diabetic.')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Assuming you have already trained SVM, KNN, and Random Forest classifiers
# SVM
svm_predictions = classifier.predict(X_test)
svm_metrics = {
    'Accuracy': accuracy_score(Y_test, svm_predictions),
    'Precision': precision_score(Y_test, svm_predictions),
    'Recall': recall_score(Y_test, svm_predictions),
    'F1 Score': f1_score(Y_test, svm_predictions),
    'ROC AUC': roc_auc_score(Y_test, svm_predictions),
}

# KNN
knn_predictions = knn_classifier.predict(X_test)
knn_metrics = {
    'Accuracy': accuracy_score(Y_test, knn_predictions),
    'Precision': precision_score(Y_test, knn_predictions),
    'Recall': recall_score(Y_test, knn_predictions),
    'F1 Score': f1_score(Y_test, knn_predictions),
    'ROC AUC': roc_auc_score(Y_test, knn_predictions),
}

# Random Forest
rf_predictions = random_forest_classifier.predict(X_test)
rf_metrics = {
    'Accuracy': accuracy_score(Y_test, rf_predictions),
    'Precision': precision_score(Y_test, rf_predictions),
    'Recall': recall_score(Y_test, rf_predictions),
    'F1 Score': f1_score(Y_test, rf_predictions),
    'ROC AUC': roc_auc_score(Y_test, rf_predictions),
}

# Print the metrics for each model
print("SVM Metrics:")
print(svm_metrics)

print("\nKNN Metrics:")
print(knn_metrics)

print("\nRandom Forest Metrics:")
print(rf_metrics)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Function to plot confusion matrix
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(5, 5))
    sns.heatmap(data=cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Confusion Matrix for SVM        
svm_conf_matrix = confusion_matrix(Y_test, svm_predictions)
print("Confusion Matrix for SVM:")
print(svm_conf_matrix)
plot_confusion_matrix(svm_conf_matrix, labels=['Non-Diabetic', 'Diabetic'])

# Confusion Matrix for KNN
knn_conf_matrix = confusion_matrix(Y_test, knn_predictions)
print("\nConfusion Matrix for KNN:")
print(knn_conf_matrix)
plot_confusion_matrix(knn_conf_matrix, labels=['Non-Diabetic', 'Diabetic'])

# Confusion Matrix for Random Forest
rf_conf_matrix = confusion_matrix(Y_test, rf_predictions)
print("\nConfusion Matrix for Random Forest:")
print(rf_conf_matrix)
plot_confusion_matrix(rf_conf_matrix, labels=['Non-Diabetic', 'Diabetic'])




