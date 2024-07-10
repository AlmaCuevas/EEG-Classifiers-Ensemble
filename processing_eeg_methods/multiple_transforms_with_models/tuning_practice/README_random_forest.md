# EEG Feature Extraction and Random Forest Classification

This code contains a pipeline for classifying brain activity using a Random Forest classifier. The process involves model optimization, training, and testing.


### Main Functions

1. **ranf_optimize(data, labels, target_names)**:
    - Performs grid search to find the best hyperparameters for the Random Forest classifier.
    - Returns the best parameters.

2. **ranf_train(data, labels, target_names, best_parameters)**:
    - Trains a Random Forest classifier using the best hyperparameters obtained from the optimization step.
    - Performs cross-validation, prints a classification report, and returns the trained classifier and accuracy.

3. **ranf_test(clf, epoch)**:
    - Tests the classifier on a given epoch of data.
    - Returns the predicted probabilities.

## Usage

1. **Feature Extraction**:
    - Load EEG data and labels for a specified dataset and subject.
    - Extract features from the EEG data using the `extractions_train` function from the custom module.

2. **Feature Normalization and Selection**:
    - Normalize the extracted features.

3. **Random Forest Optimization and Training**:
    - Optimize Random Forest hyperparameters using grid search.
    - Train the Random Forest classifier with the optimized parameters and evaluate its performance using cross-validation.

4. **Testing**:
    - Test the trained classifier on an epoch of data and output the predicted probabilities.
