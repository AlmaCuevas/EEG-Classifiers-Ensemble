import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from processing_eeg_methods.share import GLOBAL_SEED, ROOT_VOTING_SYSTEM_PATH

# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", probability=True),
    RidgeClassifier(),
    # GaussianProcessClassifier(1.0 * RBF(1.0), random_state=GLOBAL_SEED),  It takes a lot of time
    DecisionTreeClassifier(max_depth=5, random_state=GLOBAL_SEED),
    RandomForestClassifier(
        max_depth=5, n_estimators=100, max_features=1, random_state=GLOBAL_SEED
    ),
    MLPClassifier(alpha=1, max_iter=1000, random_state=GLOBAL_SEED),
    AdaBoostClassifier(algorithm="SAMME", random_state=GLOBAL_SEED),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    LinearDiscriminantAnalysis(),
    LogisticRegression(),
]


def write_model_info(
    filepath: str,
    model_name: str,
    channel_config: str,
    dataset_info: dict,
    notes: str = "",
):
    """
    Writes information about a model to a text file.

    Parameters:
    filename (str): The name of the file to write to.
    model_name (str): The name of the model.
    montage_style (str): independent channels or standard montage.
    dataset_info (dict): Information about the dataset.
    notes (str): Any additional notes about the model.

    Returns:
    None
    """

    with open(filepath, "w") as file:
        file.write(f"Model Name: {model_name}\n")
        file.write(f"Channel Configuration: {channel_config}\n")
        file.write("\nDataset Information:\n")
        for key, value in dataset_info.items():
            file.write(f"{key}: {value}\n")
        if notes:
            file.write(f"Notes: {notes}\n")


def class_selection(dataX, dataY, event_dict: dict, selected_classes: list[int]):
    dataX_selected: list = []
    dataY_selected: list = []
    for dataX_idx, dataY_idx in zip(dataX, dataY):
        if dataY_idx in selected_classes:
            dataX_selected.append(dataX_idx)
            dataY_selected.append(dataY_idx)
    dataX_selected_np = np.asarray(dataX_selected)
    dataY_selected_df = pd.Series(dataY_selected)

    label_remap = {
        dataY_original: dataY_remap_idx
        for dataY_remap_idx, dataY_original in enumerate(selected_classes)
    }

    event_dict = {
        key: label_remap[value]
        for key, value in event_dict.items()
        if value in selected_classes
    }

    return (
        dataX_selected_np,
        np.asarray(dataY_selected_df.replace(label_remap)),
        event_dict,
    )


def convert_into_binary(label, chosen_numbered_label):
    label_copy = label.copy()
    for i, label_i in enumerate(label_copy):
        if label_i == chosen_numbered_label:
            label[i] = 1
        else:
            label[i] = 0
    return label


def train_test_val_split(dataX, dataY, valid_flag: bool = False):
    train_ratio = 0.75
    test_ratio = 0.10

    # train is now 75% of the entire data set
    x_train, x_test, y_train, y_test = train_test_split(
        dataX, dataY, test_size=1 - train_ratio
    )

    if valid_flag:
        validation_ratio = 1 - train_ratio - test_ratio
        # test is now 10% of the initial data set
        # validation is now 15% of the initial data set
        x_val, x_test, y_val, y_test = train_test_split(
            x_test, y_test, test_size=test_ratio / (test_ratio + validation_ratio)
        )
    else:
        x_val = None
        y_val = None
    return x_train, x_test, x_val, y_train, y_test, y_val


def standard_saving_path(
    dataset_info: dict,
    processing_name: str,
    version_name: str,
    file_ending: str = "txt",
    subject_id: int = 0,
):
    saving_folder: str = (
        f"{ROOT_VOTING_SYSTEM_PATH}/Results/{dataset_info['dataset_name']}/{processing_name}"
    )
    os.makedirs(saving_folder, exist_ok=True)
    if subject_id:
        return f"{saving_folder}/{version_name}_{subject_id}.{file_ending}"
    else:
        return f"{saving_folder}/{version_name}.{file_ending}"


def get_dataset_basic_info(datasets_basic_infos: dict, dataset_name: str) -> dict:
    if dataset_name not in datasets_basic_infos:
        raise Exception(
            f"Not supported dataset named '{dataset_name}', choose from the following: "
            f"{list(datasets_basic_infos.keys())}"
        )
    else:
        return datasets_basic_infos[dataset_name]


def get_best_classificator_and_test_accuracy(data, labels, estimators):
    param_grid = []
    for classificator in classifiers:
        param_grid.append({"clf__estimator": [classificator]})

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=GLOBAL_SEED)
    gridsearch_clf = GridSearchCV(
        estimator=estimators, param_grid=param_grid, cv=cv
    )  # https://stackoverflow.com/questions/52580023/how-to-get-the-best-estimator-parameters-out-from-pipelined-gridsearch-and-cro

    gridsearch_clf.fit(data, labels)

    best_estimator = gridsearch_clf.best_estimator_
    best_estimator.fit(data, labels)  # Fit the best estimator on the entire dataset

    return best_estimator, gridsearch_clf.best_score_


def convert_into_independent_channels(data, labels):
    data_independent_channels = data.reshape(
        [data.shape[0] * data.shape[1], data.shape[2]], order="C"
    )  # [trials, channels, time] to [trials*channels, time]
    labels_independent_channels = np.repeat(labels, data.shape[1], axis=0)
    return data_independent_channels, labels_independent_channels


def flat_a_list(list_to_flat: list):
    return [x for xs in list_to_flat for x in xs]


class ClfSwitcher(BaseEstimator):
    # https://stackoverflow.com/questions/48507651/multiple-classification-models-in-a-scikit-pipeline-python

    def __init__(
        self,
        estimator=SGDClassifier(),
    ):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """

        self.estimator = estimator

    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        try:
            return self.estimator._predict_proba_lr(X)
        except AttributeError:
            return self.estimator.predict_proba(X)

    def score(self, X, y):
        return self.estimator.score(X, y)

    def coef_(self):
        return self.estimator.coef_


def data_normalization(data):
    min_val = np.min(data)
    max_val = np.max(data)
    scaled_data = (data - min_val) / (max_val - min_val)

    return scaled_data


def get_input_data_path(dataset_name: str) -> str:
    return ROOT_VOTING_SYSTEM_PATH + "/Datasets/" + dataset_name + "_dataset"


def probabilities_to_answer(probs_by_channels: list, voting_by_mode: bool = False):
    if voting_by_mode:
        overall_decision = flat_a_list(probs_by_channels)
        return max(set(overall_decision), key=list(overall_decision).count)
    else:  # voting_by_array_probabilities
        by_channel_decision = np.nanmean(probs_by_channels, axis=0)  # Mean over columns
        return np.argmax(by_channel_decision)


def balance_samples(data, labels, augment=True, super_augmentation: int = 0):
    """
    Process data by either augmenting and balancing or only balancing.

    Parameters:
    data (numpy.ndarray): The input data with shape (num_samples, num_channels, num_timepoints).
    labels (numpy.ndarray): The corresponding labels.
    augment (bool): If True, augment and balance the data;
                    if False, only balance the data.
    super_augmentation (int): The number of times to apply super augmentation.

    Returns:
    processed_data (numpy.ndarray): The processed data.
    processed_labels (numpy.ndarray): The updated labels for the processed data.
    """

    if augment:
        even_samples = data[:, :, ::2]  # Take even index samples
        odd_samples = data[:, :, 1::2]  # Take odd index samples

        # Check if the last dimension lengths match, pad if necessary
        if even_samples.shape[2] != odd_samples.shape[2]:
            min_length = min(even_samples.shape[2], odd_samples.shape[2])
            even_samples = even_samples[:, :, :min_length]
            odd_samples = odd_samples[:, :, :min_length]

        # Concatenate the augmented data
        augmented_data = np.concatenate((even_samples, odd_samples), axis=0)
        augmented_labels = np.concatenate((labels, labels), axis=0)

        # Use augmented data for balancing
        data, labels = augmented_data, augmented_labels

    original_data, original_labels = np.copy(data), np.copy(labels)

    if super_augmentation != 0:
        for i in range(super_augmentation):
            augmented_data = augment_data_with_time_shuffling_by_trial(
                original_data, original_labels, i
            )
            augmented_labels = np.copy(original_labels)

            # Concatenate both augmentations with original data
            data = np.concatenate((data, augmented_data), axis=0)
            labels = np.concatenate((labels, augmented_labels), axis=0)

            # Print progress every 10 iterations
            if (i + 1) % 10 == 0:
                progress = (i + 1) * 100 // super_augmentation
                print(f"Augmentation Progress: {progress}%")

        # Ensuring final progress display if total iterations is not a multiple of 10
        if super_augmentation % 10 != 0:
            print("Augmentation Progress: 100%")

    # Balance the data
    label_counts = Counter(labels)
    min_count = min(label_counts.values())

    balanced_data = []
    balanced_labels = []

    for label in label_counts.keys():
        label_indices = np.where(labels == label)[0]
        np.random.shuffle(label_indices)
        selected_indices = label_indices[:min_count]
        balanced_data.append(data[selected_indices])
        balanced_labels.append(labels[selected_indices])

    balanced_data = np.concatenate(balanced_data, axis=0)
    balanced_labels = np.concatenate(balanced_labels, axis=0)

    return balanced_data, balanced_labels


def augment_data_with_time_shuffling_by_trial(data, labels, random_seed=GLOBAL_SEED):
    """
    Augments data by shuffling time points within each trial, using a specified random seed.

    Parameters:
    data (numpy.ndarray): The input data with shape (num_samples, num_channels, num_timepoints).
    labels (numpy.ndarray): The corresponding labels.
    random_seed (int): The seed for the random number generator (optional).

    Returns:
    augmented_data (numpy.ndarray): The augmented data with shuffled time points.
    """
    augmented_data = np.copy(data)
    unique_labels = np.unique(labels)

    np.random.seed(random_seed)

    for label in unique_labels:
        label_indices = np.where(labels == label)[0]

        for channel in range(data.shape[1]):
            channel_data = data[label_indices, channel, :]
            num_timepoints = channel_data.shape[1]

            for timepoint in range(num_timepoints):
                timepoint_values = channel_data[:, timepoint]
                np.random.shuffle(timepoint_values)

                for i, idx in enumerate(label_indices):
                    augmented_data[idx, channel, timepoint] = timepoint_values[i]

    return augmented_data
