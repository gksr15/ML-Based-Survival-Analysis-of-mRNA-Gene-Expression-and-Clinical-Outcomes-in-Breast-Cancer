# ML-Based-Survival-Analysis-of-mRNA-Gene-Expression-and-Clinical-Outcomes-in-Breast-Cancer

## Project Overview

This project investigates the relationship between mRNA gene expression data and clinical outcomes in breast cancer using machine learning techniques. The primary aim is to analyze how alterations in mRNA expression levels contribute to the classification of breast cancer subtypes based on PAM50 and Claudin-Low classifications, and identify key molecular markers characterizing these subtypes.

## Problem Statement

### Classification of Breast Cancer Subtypes
- **Objective**: To determine how changes in the standardized values (z-scores) of mRNA expression levels contribute to the classification of breast cancer subtypes based on PAM50 and Claudin-Low classifications. Identify key molecular markers characterizing these subtypes.
- **Dataset**: METABRIC dataset from Kaggle containing clinical, pathological, and genetic mutation data of breast cancer patients.

## Project Structure

### 1. Data Acquisition
- **Source**: [Kaggle METABRIC dataset](https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles)
- **Description**: The dataset contains information on 1980 primary breast cancer samples, including clinical data, gene expression data, and mutation data.

### 2. Data Preprocessing
- Handling missing values
- Standardizing numerical features
- Encoding categorical variables
- Balancing the dataset using SMOTE for the classification task

### 3. Feature Selection
- Selecting relevant features for the classification task: mRNA expression levels, clinical features, PAM50, and Claudin-Low classifications.

### 4. Model Training and Evaluation
- **Algorithms Used**:
  - Random Forest Classifier
  - Logistic Regression
  - Random Forest Regressor
  - Lasso Regression
- **Evaluation Metrics**:
  - AUC Score
  - F1-score
  - Confusion Matrix
  - R squared value
  - RMSE 

## Scripts Description

### `Survial_Analysis_Notebook.ipynb`
This script handles the preprocessing of the METABRIC dataset, including:
- Loading the dataset
- Cleaning missing values
- Standardizing numerical features
- Encoding categorical variables

The next part of the script performs feature selection for the classification task, focusing on mRNA expression levels and clinical features relevant to PAM50 and Claudin-Low classifications.

The above notebook also handles the training and evaluation of various machine learning models, including:
- Splitting the dataset into training and testing sets
- Training models with hyperparameter tuning
- Evaluating model performance using accuracy, precision, recall, F1-score, and confusion matrix

Finally visualizes the results, including feature importance, confusion matrices, and performance metrics for each model.

## Results

- **Best Performing Model for Classification**: The Random Forest classifier exhibited the highest accuracy and precision for PAM50 subtype classification.
- **Key Molecular Markers**: Significant genes contributing to subtype classification include GATA3, ESR1, and FOXA1 among others.
- **Model Performance**:
  - Random Forest Classifier: AUC - 0.996, F1-score - 00.996
  - Logistic Regression: AUC- 0.969, F1-score - 0.976
  - Random Forest Regressor - R squared error - 0.492, RMSE - 54.84
  - Lasso Regression -  R squared error - 0.29, RMSE - 64.80

## References

1. Kaggle METABRIC Dataset: [link](https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles)
2. Pereira, B., Chin, SF., Rueda, O. et al. The somatic mutation profiles of 2,433 breast cancers refine their genomic and transcriptomic landscapes. Nat Commun 7, 11479 (2016).
3. Breiman, L. Random Forests. Machine Learning, 45(1), 5-32, 2001.

## Contributors

- Gautham Krishna S R

- Sandhya G


Feel free to raise any issues or contribute to this project by submitting pull requests.
