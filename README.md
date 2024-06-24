# ML-Based-Survival-Analysis-of-mRNA-Gene-Expression-and-Clinical-Outcomes-in-Breast-Cancer

Project Overview
This project investigates the relationship between mRNA gene expression data and clinical outcomes in breast cancer using machine learning techniques. The primary aim is to analyze how alterations in mRNA expression levels contribute to the classification of breast cancer subtypes based on PAM50 and Claudin-Low classifications, and identify key molecular markers characterizing these subtypes.

Problem Statement
Question 3: Classification of Breast Cancer Subtypes
Objective: To determine how changes in the standardized values (z-scores) of mRNA expression levels contribute to the classification of breast cancer subtypes based on PAM50 and Claudin-Low classifications. Identify key molecular markers characterizing these subtypes.
Dataset: METABRIC dataset from Kaggle containing clinical, pathological, and genetic mutation data of breast cancer patients.
Project Structure
1. Data Acquisition
Source: Kaggle METABRIC dataset
Description: The dataset contains information on 1980 primary breast cancer samples, including clinical data, gene expression data, and mutation data.
2. Data Preprocessing
Handling missing values
Standardizing numerical features
Encoding categorical variables
Balancing the dataset using SMOTE for the classification task
3. Feature Selection
Selecting relevant features for the classification task: mRNA expression levels, clinical features, PAM50, and Claudin-Low classifications.
4. Model Training and Evaluation
Algorithms Used:
Random Forest Classifier
Support Vector Machine (SVM)
Decision Tree Classifier
Naive Bayes
XGBoost
Logistic Regression
Evaluation Metrics:
Accuracy
Precision
Recall
F1-score
Confusion Matrix
Scripts Description
data_preprocessing.py
This script handles the preprocessing of the METABRIC dataset, including:

Loading the dataset
Cleaning missing values
Standardizing numerical features
Encoding categorical variables
Balancing the dataset using SMOTE
feature_selection.py
This script performs feature selection for the classification task, focusing on mRNA expression levels and clinical features relevant to PAM50 and Claudin-Low classifications.

model_training.py
This script handles the training and evaluation of various machine learning models, including:

Splitting the dataset into training and testing sets
Training models with hyperparameter tuning
Evaluating model performance using accuracy, precision, recall, F1-score, and confusion matrix
results_analysis.py
This script visualizes the results, including feature importance, confusion matrices, and performance metrics for each model.
