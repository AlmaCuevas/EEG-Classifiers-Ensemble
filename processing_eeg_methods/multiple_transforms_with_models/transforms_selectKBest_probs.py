import time

import mne
import numpy as np
import pandas as pd
from data_loaders import load_data_labels_based_on_dataset
from data_utils import (
    ClfSwitcher,
    get_best_classificator_and_test_accuracy,
    get_dataset_basic_info,
    get_input_data_path,
    standard_saving_path,
)
from mne.decoding import CSP
from pyriemann.estimation import Covariances, ERPCovariances, XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from scipy import signal
from share import ROOT_VOTING_SYSTEM_PATH, datasets_basic_infos
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

# todo: add the test template
# todo: do the deap thing about the FFT: https://github.com/tongdaxu/EEG_Emotion_Classifier_DEAP/blob/master/Preprocess_Deap.ipynb

threshold_for_bug = 0.00000001  # could be any value, ex numpy.min


def transform_data(
    data, dataset_info: dict, labels=None, transform_methods: dict = {}
) -> tuple[pd.DataFrame, dict]:
    features: dict = {
        # Do not use 'Vect' transform, most of the time is nan or 0.25 if anything.
        "ERPcova": Pipeline(
            [("ERPcova", ERPCovariances(estimator="oas")), ("ts", TangentSpace())]
        ),  # Add TangentSpace, otherwise the dimensions are not 2D.
        "XdawnCova": Pipeline(
            [("XdawnCova", XdawnCovariances(estimator="oas")), ("ts", TangentSpace())]
        ),  # Add TangentSpace, otherwise the dimensions are not 2D.
        "CSP": Pipeline(
            [("Vectorizer", CSP(n_components=4, reg=None, log=True, norm_trace=False))]
        ),
        "Cova": Pipeline(
            [("Cova", Covariances()), ("ts", TangentSpace())]
        ),  # Add TangentSpace, otherwise the dimensions are not 2D.
    }
    frequency_ranges: dict = {
        "complete": [0, int(dataset_info["sample_rate"] / 2) - 1],
        "delta": [0, 3],
        "theta": [3, 7],
        "alpha": [7, 13],
        "beta 1": [13, 16],
        "beta 2": [16, 20],
        "beta 3": [20, 35],
        "gamma": [35, int(dataset_info["sample_rate"] / 2) - 1],
    }

    features_df = pd.DataFrame()

    for feature_name, feature_method in features.items():
        if labels is not None:
            transform_methods[feature_name] = Pipeline([(feature_name, feature_method)])
        for frequency_bandwidth_name, frequency_bandwidth in frequency_ranges.items():
            print(frequency_bandwidth)
            iir_params = dict(order=8, ftype="butter")
            filt = mne.filter.create_filter(
                data,
                dataset_info["sample_rate"],
                l_freq=frequency_bandwidth[0],
                h_freq=frequency_bandwidth[1],
                method="iir",
                iir_params=iir_params,
                verbose=True,
            )
            filtered = signal.sosfiltfilt(filt["sos"], data)
            filtered[filtered < threshold_for_bug] = (
                threshold_for_bug  # To avoid the error "SVD did not convergence"
            )
            if labels is not None:
                X_features = transform_methods[feature_name].fit_transform(
                    filtered, labels
                )
            else:
                X_features = transform_methods[feature_name].transform(filtered)
            print("Combined space has", X_features.shape[1], "features")
            column_name = [
                f"{frequency_bandwidth_name}_{feature_name}_{num}"
                for num in range(0, X_features.shape[1])
            ]
            temp_features_df = pd.DataFrame(X_features, columns=column_name)
            features_df = pd.concat([features_df, temp_features_df], axis=1)
    return features_df, transform_methods


def selected_transformers_train(features_df, labels):
    X_SelectKBest = SelectKBest(f_classif, k=100)
    X_new = X_SelectKBest.fit_transform(features_df, labels)
    columns_list = X_SelectKBest.get_feature_names_out()
    features_df = pd.DataFrame(X_new, columns=columns_list)

    classifier, acc = get_best_classificator_and_test_accuracy(
        features_df, labels, Pipeline([("clf", ClfSwitcher())])
    )
    return classifier, acc, columns_list


def selected_transformers_test(clf, features_df):
    """
    This is what the real-time BCI will call.
    Parameters
    ----------
    clf : classifier trained for the specific subject
    features_df: one epoch, the current one that represents the intention of movement of the user.

    Returns Array of classification with 4 floats representing the target classification
    -------

    """

    array = clf.predict_proba(features_df)

    return array


if __name__ == "__main__":
    # Manual Inputs
    datasets = ["braincommand"]
    for dataset_name in datasets:
        version_name = "22_23_independent_channels_one_transforms_table_of_selectKbest"
        processing_name = ""

        data_path: str = get_input_data_path(dataset_name)
        dataset_info: dict = get_dataset_basic_info(datasets_basic_infos, dataset_name)

        saving_txt_path: str = standard_saving_path(
            dataset_info, processing_name, version_name
        )
        mean_accuracy_per_subject: list = []
        results_df = pd.DataFrame()

        for subject_id in range(22, 24):  # Only two things I should be able to change
            print(subject_id)
            with open(
                saving_txt_path,
                "a",
            ) as f:
                f.write(f"Subject: {subject_id}\n\n")
            epochs, data, labels = load_data_labels_based_on_dataset(
                dataset_info,
                subject_id,
                data_path,
                normalize=False,  # If I normalize it doesn't run
                threshold_for_bug=0.00000001,  # could be any value, ex numpy.min
                channels_independent=True,
            )
            # Do cross-validation
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            acc_over_cv = []
            testing_time_over_cv = []
            training_time = []
            accuracy = 0
            for train, test in cv.split(epochs, labels):
                print(
                    "******************************** Training ********************************"
                )
                start = time.time()
                features_train_df, transform_methods = transform_data(
                    data[train], dataset_info=dataset_info, labels=labels[train]
                )

                clf, accuracy, columns_list = selected_transformers_train(
                    features_train_df, labels[train]
                )
                training_time.append(time.time() - start)
                with open(
                    saving_txt_path,
                    "a",
                ) as f:
                    f.write(f"Accuracy of training: {accuracy}\n")
                print(
                    "******************************** Test ********************************"
                )
                pred_list = []
                testing_time = []
                for epoch_number in test:
                    start = time.time()
                    features_test_df, _ = transform_data(
                        np.asarray([data[epoch_number]]),
                        dataset_info=dataset_info,
                        labels=None,
                        transform_methods=transform_methods,
                    )
                    array = selected_transformers_test(
                        clf, features_test_df[columns_list]
                    )
                    end = time.time()
                    testing_time.append(end - start)
                    print(dataset_info["target_names"])
                    print("Probability voting system: ", array)

                    voting_system_pred = np.argmax(array)
                    pred_list.append(voting_system_pred)
                    print("Prediction: ", voting_system_pred)
                    print("Real: ", labels[epoch_number])

                acc = np.mean(pred_list == labels[test])
                testing_time_over_cv.append(np.mean(testing_time))
                acc_over_cv.append(acc)
                with open(
                    saving_txt_path,
                    "a",
                ) as f:
                    f.write(f"Prediction: {pred_list}\n")
                    f.write(f"Real label:{labels[test]}\n")
                    f.write(f"Mean accuracy in KFold: {acc}\n")
                print("Mean accuracy in KFold: ", acc)
                break
            mean_acc_over_cv = np.mean(acc_over_cv)

            with open(
                saving_txt_path,
                "a",
            ) as f:
                f.write(f"Final acc: {mean_acc_over_cv}\n\n\n\n")
            print(f"Final acc: {mean_acc_over_cv}")

            temp = pd.DataFrame(
                {
                    "Subject ID": [subject_id] * len(acc_over_cv),
                    "Version": [version_name] * len(acc_over_cv),
                    "Training Accuracy": [accuracy] * len(acc_over_cv),
                    "Training Time": training_time,
                    "Testing Accuracy": acc_over_cv,
                    "Testing Time": testing_time_over_cv,
                }
            )  # The idea is that the most famous one is the one I use for this dataset
            results_df = pd.concat([results_df, temp])

        results_df.to_csv(
            f"{ROOT_VOTING_SYSTEM_PATH}/Results/{dataset_name}/{version_name}_{dataset_name}.csv"
        )

    print("Congrats! The processing methods are done processing.")
