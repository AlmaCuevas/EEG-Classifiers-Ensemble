# EEG Feature Extraction and Classification

This code contains functions for feature extraction from EEG data and classification of brain activity. 

## Features

### Main Functions
1. **get_entropy(data)**:
    - Calculates the Higuchi Fractal Dimension entropy for each channel in the data.

2. **hjorth(X, D=None)**:
    - Computes Hjorth mobility and complexity of a time series.

3. **get_hjorth(data)**:
    - Computes Hjorth parameters (mobility and complexity) for each channel in the data.

4. **get_ratio(data)**:
    - Calculates the EEG alpha/delta ratio for each channel in the data.

5. **get_lyapunov(data)**:
    - Computes the Lyapunov exponent for the EEG data.

6. **by_frequency_band(data, dataset_info)**:
    - Extracts features from EEG data filtered by different frequency bands (delta, theta, alpha, beta, gamma).

7. **get_extractions(data, dataset_info, frequency_bandwidth_name)**:
    - Combines all extracted features into a DataFrame for a given frequency band.

8. **extractions_train(features, labels)**:
    - Trains a classifier using the extracted features and their corresponding labels.

9. **extractions_test(clf, features_df)**:
    - Uses the trained classifier to predict brain activity based on new features.

## Usage

1. **Dependencies**:
    - Python packages: `scipy`, `mne`, `numpy`, `pandas`, `sklearn`, `antropy`, `EEGExtract`, `data_utils`, `share`, `data_loaders`

2. **Running the Code**:
    - The main script runs the entire process of loading data, extracting features, training, and testing the classifier.

### Example:

```python
if __name__ == "__main__":
    datasets = ['aguilera_gamified']
    for dataset_name in datasets:
        version_name = "features_datatransposed_by_frequency_band"
        dataset_foldername = dataset_name + "_dataset"
        computer_root_path = ROOT_VOTING_SYSTEM_PATH + "/Datasets/"
        data_path = computer_root_path + dataset_foldername

        if dataset_name not in datasets_basic_infos:
            raise Exception(
                f"Not supported dataset named '{dataset_name}', choose from the following: aguilera_traditional, aguilera_gamified, nieto, coretto or torres."
            )
        dataset_info: dict = datasets_basic_infos[dataset_name]
        mean_accuracy_per_subject: list = []
        results_df = pd.DataFrame()

        for subject_id in range(1, dataset_info["subjects"] + 1):
            with open(f"{ROOT_VOTING_SYSTEM_PATH}/Results/{version_name}_{dataset_name}.txt", "a") as f:
                f.write(f"Subject: {subject_id}\n\n")
            epochs, data, labels = load_data_labels_based_on_dataset(dataset_info, subject_id, data_path)

            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            acc_over_cv = []
            testing_time_over_cv = []
            training_time = []
            accuracy = 0
            for train, test in cv.split(epochs, labels):
                start = time.time()
                features_train = by_frequency_band(data[train], dataset_info)
                clf, accuracy = extractions_train(features_train, labels[train])
                training_time.append(time.time() - start)
                with open(f"{ROOT_VOTING_SYSTEM_PATH}/Results/{version_name}_{dataset_name}.txt", "a") as f:
                    f.write(f"Accuracy of training: {accuracy}\n")

                pred_list = []
                testing_time = []
                for epoch_number in test:
                    start = time.time()
                    features_test = by_frequency_band(np.asarray([data[epoch_number]]), dataset_info)
                    array = extractions_test(clf, features_test)
                    end = time.time()
                    testing_time.append(end - start)
                    voting_system_pred = np.argmax(array)
                    pred_list.append(voting_system_pred)

                acc = np.mean(pred_list == labels[test])
                testing_time_over_cv.append(np.mean(testing_time))
                acc_over_cv.append(acc)
                with open(f"{ROOT_VOTING_SYSTEM_PATH}/Results/{version_name}_{dataset_name}.txt", "a") as f:
                    f.write(f"Prediction: {pred_list}\n")
                    f.write(f"Real label:{labels[test]}\n")
                    f.write(f"Mean accuracy in KFold: {acc}\n")

            mean_acc_over_cv = np.mean(acc_over_cv)
            with open(f"{ROOT_VOTING_SYSTEM_PATH}/Results/{version_name}_{dataset_name}.txt", "a") as f:
                f.write(f"Final acc: {mean_acc_over_cv}\n\n\n\n")
            temp = pd.DataFrame({'Subject ID': [subject_id] * len(acc_over_cv),
                                 'Version': [version_name] * len(acc_over_cv), 'Training Accuracy': [accuracy] * len(acc_over_cv), 'Training Time': training_time,
                                 'Testing Accuracy': acc_over_cv, 'Testing Time': testing_time_over_cv})
            results_df = pd.concat([results_df, temp])

        results_df.to_csv(f"{ROOT_VOTING_SYSTEM_PATH}/Results/{version_name}_{dataset_name}.csv")

    print("Congrats! The processing methods are done processing.")
```

### Note:
- Ensure all required dependencies and custom modules (`data_utils`, `share`, `data_loaders`, `EEGExtract`) are properly set up before running the script.
- The script will output results to specified files and directories, so ensure these paths are correct and accessible.
