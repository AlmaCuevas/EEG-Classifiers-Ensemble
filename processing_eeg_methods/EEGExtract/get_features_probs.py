from scipy import signal

import mne
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold


from data_utils import get_best_classificator_and_test_accuracy, ClfSwitcher
from share import datasets_basic_infos, ROOT_VOTING_SYSTEM_PATH
from data_loaders import load_data_labels_based_on_dataset
from sklearn.pipeline import Pipeline
import time
import antropy as ant
import numpy
from sklearn.feature_selection import f_classif
import EEGExtract as eeg
from scipy.stats import kurtosis, skew

def get_entropy(data):
    entropy_values = []
    for data_trial in data:
        entropy_trial = []
        for data_channel in data_trial:
            entropy_trial.append(ant.higuchi_fd(data_channel))
        entropy_values.append(np.mean(entropy_trial))
    return entropy_values

def hjorth(X, D=None):
    """ Compute Hjorth mobility and complexity of a time series from either two
    cases below:
        1. X, the time series of type list (default)
        2. D, a first order differential sequence of X (if D is provided,
           recommended to speed up)

    In case 1, D is computed using Numpy's Difference function.

    Notes
    -----
    To speed up, it is recommended to compute D before calling this function
    because D may also be used by other functions whereas computing it here
    again will slow down.

    Parameters
    ----------

    X
        list

        a time series

    D
        list

        first order differential sequence of a time series

    Returns
    -------

    As indicated in return line

    Hjorth mobility and complexity

    """

    if D is None:
        D = numpy.diff(X)
        D = D.tolist()

    D.insert(0, X[0])  # pad the first difference
    D = numpy.array(D)

    n = len(X)

    M2 = float(sum(D ** 2)) / n
    TP = sum(numpy.array(X) ** 2)
    M4 = 0
    for i in range(1, len(D)):
        M4 += (D[i] - D[i - 1]) ** 2
    M4 = M4 / n

    return numpy.sqrt(M2 / TP), numpy.sqrt(float(M4) * TP / M2 / M2)  # Hjorth Mobility and Complexity

def get_hjorth(data):
    values = []
    Complexity_values = []
    for data_trial in data:
        Mobility_trial = []
        Complexity_trial = []
        for data_channel in data_trial:
            Mobility, Complexity = hjorth(data_channel)
            Mobility_trial.append(Mobility)
            Complexity_trial.append(Complexity)
        values.append(np.mean(Mobility_trial))
        Complexity_values.append(np.mean(Complexity_trial))
    return values, Complexity_values

def get_ratio(data):
    ratio_values = []
    for data_trial in data:
        ratio_trial = []
        for data_channel in data_trial:
            ratio_trial.append(eeg.eegRatio(data_channel,fs=500))
        ratio_values.append(np.mean(ratio_trial))
    return ratio_values

def get_lyapunov(data):
    lyapunov_values = []
    for data_trial in data:
        lyapunov_values.append(eeg.lyapunov(data_trial))
    return lyapunov_values

def by_frequency_band(data, dataset_info: dict):
    """
    This code contains functions for feature extraction from EEG data and classification of brain activity.

    Parameters
    ----------
    data
    Gets converted from [epochs, chans, ms] to [chans x ms x epochs] # Todo: it got like 1, 2, 0: but I think it should be 0, 2, 1
    dataset_info

    Returns
    -------

    """
    frequency_ranges: dict = {
        "delta": [0, 3],
        "theta": [3, 7],
        "alpha": [7, 13],
        "beta 1": [13, 16],
        "beta 2": [16, 20],
        "beta 3": [20, 35],
        "gamma": [35, np.floor(dataset_info['sample_rate']/2)-1],
    }
    features_df = get_extractions(np.transpose(data, (0, 2, 1)), dataset_info, 'complete')
    for frequency_bandwidth_name, frequency_bandwidth in frequency_ranges.items():
        print(frequency_bandwidth)
        iir_params = dict(order=8, ftype="butter")
        filt = mne.filter.create_filter(
            data, dataset_info['sample_rate'], l_freq=frequency_bandwidth[0], h_freq=frequency_bandwidth[1],
            method="iir", iir_params=iir_params, verbose=True
        )
        filtered = signal.sosfiltfilt(filt["sos"], data)
        filtered = filtered.astype('float64')
        features_array_ind = get_extractions(np.transpose(filtered, (0, 2, 1)), dataset_info, frequency_bandwidth_name)
        features_df = pd.concat([features_df, features_array_ind], axis=1)
    return features_df


def get_extractions(data, dataset_info: dict, frequency_bandwidth_name):
    # To use EEGExtract, the data must be [chans x ms x epochs]
    Mobility_values, Complexity_values = get_hjorth(data)
    ratio_values = get_ratio(data) # α/δ Ratio
    lyapunov_values = np.array(get_lyapunov(data),dtype="float64")
    regularity_values=eeg.eegRegularity(data, Fs=dataset_info['sample_rate'])
    std_values=eeg.eegStd(data)
    mean_values = np.mean(data, axis=1)
    kurtosis_values = kurtosis(data, axis=1, bias=True)
    skew_values = skew(data, axis=1, bias=True)
    variance_values = np.var(data, axis=1)
    medianFreq_values=eeg.medianFreq(data, fs=dataset_info['sample_rate'])
    mfcc_values_with_kernel=eeg.mfcc(data, fs=dataset_info['sample_rate'],order=1) # Takes a long time and it doesn't reward much
    mfcc_values = np.squeeze(mfcc_values_with_kernel, axis=2)

    # diffuseSlowing_values=eeg.diffuseSlowing(data, Fs=dataset_info['sample_rate']) # all zeros
    # numBursts_values=eeg.numBursts(data, fs=dataset_info['sample_rate']) # all zeros
    # burstLengthStats_values=eeg.burstLengthStats(data, fs=dataset_info['sample_rate']) # all zeros
    # falseNN = eeg.falseNearestNeighbor(data) # all zeros
    # coherence_res = eeg.coherence(data, dataset_info["sample_rate"]) # all ones
    # entropy = eeg.shannonEntropy(data, bin_min=-200, bin_max=200, binWidth=2) # todo: why entropy is not working?
    # tsalisRes = eeg.tsalisEntropy(data, bin_min=-200, bin_max=200, binWidth=2)

    feature_array = np.array([Complexity_values, Mobility_values, ratio_values],dtype="float64").transpose()
    feature_array = np.concatenate((feature_array, lyapunov_values, regularity_values, std_values, medianFreq_values, kurtosis_values, mean_values, skew_values, variance_values, mfcc_values), axis=1)

    feature_array[np.isfinite(feature_array) == False] = 0

    column_name = (
            [f'{frequency_bandwidth_name}_Complexity_values',
                    f'{frequency_bandwidth_name}_Mobility_values', f'{frequency_bandwidth_name}_ratio_values'] +
                   [f'{frequency_bandwidth_name}_lyapunov_{num}' for num in range(0, lyapunov_values.shape[1])] +
                   [f'{frequency_bandwidth_name}_regularity_values_{num}' for num in range(0, regularity_values.shape[1])] +
                   [f'{frequency_bandwidth_name}_std_values_{num}' for num in range(0, std_values.shape[1])]+
                   [f'{frequency_bandwidth_name}_medianFreq_values_{num}' for num in range(0, medianFreq_values.shape[1])] +
                   [f'{frequency_bandwidth_name}_kurtosis_values_{num}' for num in range(0, kurtosis_values.shape[1])] +
                   [f'{frequency_bandwidth_name}_mean_values_{num}' for num in range(0, mean_values.shape[1])] +
                    [f'{frequency_bandwidth_name}_skew_values_{num}' for num in range(0, skew_values.shape[1])] +
                   [f'{frequency_bandwidth_name}_variance_values_{num}' for num in range(0, variance_values.shape[1])] +
                   [f'{frequency_bandwidth_name}_mfcc_values_{num}' for num in range(0, mfcc_values.shape[1])]
                   )

    feature_df = pd.DataFrame(feature_array, columns=column_name)

    return feature_df

def extractions_train(features_df, labels):
    print('training...')
    X_SelectKBest = SelectKBest(f_classif, k=100)
    X_new = X_SelectKBest.fit_transform(features_df, labels)
    columns_list = X_SelectKBest.get_feature_names_out()
    features_df = pd.DataFrame(X_new, columns=columns_list)

    classifier, acc = get_best_classificator_and_test_accuracy(features_df, labels, Pipeline([('clf', ClfSwitcher())]))
    return classifier, acc, columns_list


def extractions_test(clf, features_df):
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
    datasets = ['braincommand']#, 'aguilera_traditional', 'torres', 'aguilera_gamified'
    for dataset_name in datasets:
        version_name = "features_datatransposed_by_frequency_band" # To keep track what the output processing alteration went through

        # Folders and paths
        dataset_foldername = dataset_name + "_dataset"
        computer_root_path = ROOT_VOTING_SYSTEM_PATH + "/Datasets/"
        data_path = computer_root_path + dataset_foldername
        print(data_path)
        # Initialize
        if dataset_name not in datasets_basic_infos:
            raise Exception(
                f"Not supported dataset named '{dataset_name}', choose from the following: aguilera_traditional, aguilera_gamified, nieto, coretto or torres."
            )
        dataset_info: dict = datasets_basic_infos[dataset_name]

        mean_accuracy_per_subject: list = []
        results_df = pd.DataFrame()

        for subject_id in range(
            29, 30
        ):
            print(subject_id)
            with open(
                f"{ROOT_VOTING_SYSTEM_PATH}/Results/{dataset_name}/{version_name}_{dataset_name}.txt",
                "a",
            ) as f:
                f.write(f"Subject: {subject_id}\n\n")
            epochs, data, _ = load_data_labels_based_on_dataset(dataset_info, subject_id, data_path)
            labels = epochs.events[:, 2].astype(np.int64)
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
                features_train = by_frequency_band(data[train], dataset_info)
                clf, accuracy, columns_list = extractions_train(features_train, labels[train])
                training_time.append(time.time() - start)
                with open(
                    f"{ROOT_VOTING_SYSTEM_PATH}/Results/{dataset_name}/{version_name}_{dataset_name}.txt",
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
                    features_test = by_frequency_band(np.asarray([data[epoch_number]]), dataset_info)
                    array = extractions_test(clf, features_test[columns_list])
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
                    f"{ROOT_VOTING_SYSTEM_PATH}/Results/{dataset_name}/{version_name}_{dataset_name}.txt",
                    "a",
                ) as f:
                    f.write(f"Prediction: {pred_list}\n")
                    f.write(f"Real label:{labels[test]}\n")
                    f.write(f"Mean accuracy in KFold: {acc}\n")
                print("Mean accuracy in KFold: ", acc)
            mean_acc_over_cv = np.mean(acc_over_cv)

            with open(
                f"{ROOT_VOTING_SYSTEM_PATH}/Results/{dataset_name}/{version_name}_{dataset_name}.txt",
                "a",
            ) as f:
                f.write(f"Final acc: {mean_acc_over_cv}\n\n\n\n")
            print(f"Final acc: {mean_acc_over_cv}")

            temp = pd.DataFrame({'Subject ID': [subject_id] * len(acc_over_cv),
                                 'Version': [version_name] * len(acc_over_cv), 'Training Accuracy': [accuracy] * len(acc_over_cv), 'Training Time': training_time,
                                 'Testing Accuracy': acc_over_cv, 'Testing Time': testing_time_over_cv}) # The idea is that the most famous one is the one I use for this dataset
            results_df = pd.concat([results_df, temp])

        results_df.to_csv(
            f"{ROOT_VOTING_SYSTEM_PATH}/Results/{dataset_name}/{version_name}_{dataset_name}.csv")

    print("Congrats! The processing methods are done processing.")
