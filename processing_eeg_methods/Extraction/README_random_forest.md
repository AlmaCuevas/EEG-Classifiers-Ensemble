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

## Dependencies

Make sure you have the following Python packages installed:
- `numpy`
- `pandas`
- `scikit-learn`
- `processing_eeg_methods` (custom module)
- `Extraction` (custom module)

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

### Example:

```python

if __name__ == '__main__':
    subject_id = 2
    dataset_name = 'aguilera_traditional'
    dataset_foldername = dataset_name + '_dataset'
    computer_root_path = ROOT_VOTING_SYSTEM_PATH + '/Datasets/'
    data_path = computer_root_path + dataset_foldername
    dataset_info = datasets_basic_infos[dataset_name]

    epochs, data, y = load_data_labels_based_on_dataset(dataset_info, subject_id, data_path)
    target_names = dataset_info['target_names']
    df = extractions_train(data, y, target_names)
    df = df.to_numpy()
    labels = df[:, 3]
    df = normalize(df[:, [0, 1, 2]], axis=0)
    print(df)

    print("******************************** Training ********************************")
    start = time.time()
    best_parameters = ranf_optimize(df[:, [0, 1, 2]], labels, target_names)
    clf, acc = ranf_train(df[:, [0, 1, 2]], labels, target_names, best_parameters)
    end = time.time()
    print("Training time: ", end - start)

    print("******************************** Test ********************************")
    epoch_number = 0
    start = time.time()
    array = ranf_test(clf, df[:, [0, 1, 2]])
    end = time.time()
    print("One epoch, testing time: ", end - start)
    print(target_names)
    print("Probability:", array[epoch_number])
    print("Real:", y[epoch_number])
```

### Notes:
- Ensure the paths and module imports are correctly set up before running the script.
- Adjust the `subject_id` and `dataset_name` as needed.
- The `extractions_train` function and other custom modules (`processing_eeg_methods`, `Extraction`) should be correctly implemented and accessible.
