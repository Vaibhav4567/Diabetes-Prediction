# Diabetes Prediction Model

This project aims to predict the likelihood of an individual having diabetes based on certain health metrics using machine learning algorithms.

## Project Overview

The project includes the following steps:

1. **Data Loading and Preprocessing**: The diabetes dataset is loaded from a CSV file and preprocessed to handle missing values, if any, and scale the features.

2. **Exploratory Data Analysis (EDA)**: Exploratory data analysis is performed to understand the distribution of data, check for correlations, and gain insights into the features.

3. **Model Building**: Several machine learning algorithms including Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and Random Forest are trained on the preprocessed data to predict diabetes.

4. **Model Evaluation**: The trained models are evaluated using various metrics such as accuracy, precision, recall, F1-score, and ROC AUC score.

## Dataset
The dataset used for this project contains the following features:

Pregnancies
Glucose
Blood Pressure
Skin Thickness
Insulin
BMI (Body Mass Index)
Diabetes Pedigree Function
Age
The target variable is "Outcome," indicating whether the individual has diabetes (1) or not (0).

## Models Used
Support Vector Machine (SVM): Linear SVM classifier is used for binary classification.
K-Nearest Neighbors (KNN): KNN classifier is utilized with a chosen number of neighbors.
Random Forest: Random Forest classifier with 100 decision trees is employed.

## Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction
