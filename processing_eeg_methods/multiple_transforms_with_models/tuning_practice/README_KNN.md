# EEG Feature Extraction and KNN Classification

This code provides a pipeline for extracting features from EEG data and using K-Nearest Neighbors (KNN) to classify brain activity. The process includes  feature selection, and model optimization and training.

## Feature Extraction and KNN Classification Pipeline

### Main Functions

1. **KNN_optimize(data, labels, target_names)**:
    - Performs grid search to find the best hyperparameters (`n_neighbors` and `weights`) for the KNN classifier.

2. **KNN_train(features_array_new, labels, target_names, best_k, best_weights)**:
    - Trains a KNN classifier using the best hyperparameters found in the optimization step.
    - Performs cross-validation and prints a classification report and accuracy.

## Usage

1. **Feature Extraction**:
    - Loads EEG data and labels for a given dataset and subject.
    - Extracts features from the EEG data using the `get_extractions` function.

2. **Feature Selection**:
    - Normalizes the extracted features.
    - Selects the top 10 features using the Chi-squared test.

3. **KNN Optimization and Training**:
    - Optimizes KNN hyperparameters using grid search.
    - Trains the KNN classifier with the optimized parameters and evaluates its performance using cross-validation.
