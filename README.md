# Career Level Classification Model

This project implements a machine learning pipeline to classify job postings into different career levels based on features like title, description, location, function, and industry.

## Project Overview

The goal of this project is to predict career levels from job postings using a Random Forest classifier with sophisticated text processing and feature selection techniques. The model handles imbalanced data through SMOTE oversampling and optimizes feature selection through grid search.

## Features

- Text processing with TF-IDF for title, description, and industry columns
- One-Hot Encoding for categorical features (location and function)
- SMOTE oversampling to handle class imbalance
- Feature selection using Chi-squared test
- GridSearchCV for hyperparameter tuning
- Random Forest classification

## Data Processing

Key data processing steps:
1. Location filtering to extract US state codes
2. Handling missing values
3. Text preprocessing with English stop words removal
4. N-gram processing (unigrams and bigrams) for description field
5. Minimum/maximum document frequency filtering

## Model Pipeline

The classification pipeline consists of:
1. ColumnTransformer for different preprocessing of each feature type
2. Feature selection using SelectPercentile with chi2
3. RandomForestClassifier
4. GridSearchCV for parameter optimization

## Installation

To run this project, you'll need:

```bash
pip install pandas scikit-learn imbalanced-learn odfpy
