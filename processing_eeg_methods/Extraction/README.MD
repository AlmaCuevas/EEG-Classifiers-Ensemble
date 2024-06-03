# EEG Feature Extraction and KNN Classification

This code provides a pipeline for extracting features from EEG data and using K-Nearest Neighbors (KNN) to classify brain activity. The process includes  feature selection, and model optimization and training.

## Feature Extraction and KNN Classification Pipeline

### Main Functions

1. **KNN_optimize(data, labels, target_names)**:
    - Performs grid search to find the best hyperparameters (`n_neighbors` and `weights`) for the KNN classifier.

2. **KNN_train(data, labels, target_names, best_k, best_weights)**:
    - Trains a KNN classifier using the best hyperparameters found in the optimization step.
    - Performs cross-validation and prints a classification report and accuracy.
## Dependencies

Ensure you have the following Python packages installed:
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `processing_eeg_methods` (custom module)
- `Extraction` (custom module)

### Usage

1. **Feature Extraction**:
    - Loads EEG data and labels for a given dataset and subject.
    - Extracts features from the EEG data using the `get_extractions` function.

2. **Feature Selection**:
    - Normalizes the extracted features.
    - Selects the top 10 features using the Chi-squared test.

3. **KNN Optimization and Training**:
    - Optimizes KNN hyperparameters using grid search.
    - Trains the KNN classifier with the optimized parameters and evaluates its performance using cross-validation.

### Example:

```python
if __name__ == '__main__':
    subject_id = 7
    dataset_name = 'aguilera_traditional'
    dataset_foldername = dataset_name + '_dataset'
    computer_root_path = ROOT_VOTING_SYSTEM_PATH + "/Datasets/"
    data_path = computer_root_path + dataset_foldername
    dataset_info = datasets_basic_infos[dataset_name]

    _, data, labels = load_data_labels_based_on_dataset(dataset_info, subject_id, data_path, threshold_for_bug=0.00000001)
    features_array = get_extractions(data, dataset_info)
    features_array = normalize(features_array, axis=0) + 1
    target_names = dataset_info['target_names']
    features_array_new = SelectKBest(chi2, k=10).fit_transform(features_array, labels)
    best_k, best_weights = KNN_optimize(features_array_new, labels, target_names)
    KNN_train(data, labels, target_names, best_k, best_weights)
```

### Notes:
- Ensure the paths and module imports are correctly set up before running the script.
- Adjust the `subject_id` and `dataset_name` as needed.
- The `get_extractions` function and other custom modules (`processing_eeg_methods`, `Extraction`) should be correctly implemented and accessible.
