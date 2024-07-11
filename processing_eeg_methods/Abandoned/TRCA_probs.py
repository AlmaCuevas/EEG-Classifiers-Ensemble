# This was used for SSVEP either way, the inout format is wrong. Maybe not even worth it to find out how to do it

import time

import numpy as np
from data_loaders import load_data_labels_based_on_dataset
from meegkit.trca import TRCA
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from share import datasets_basic_infos
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

filterbank = [
    [(6, 90), (4, 100)],  # passband, stopband freqs [(Wp), (Ws)]
    [(14, 90), (10, 100)],
    [(22, 90), (16, 100)],
    [(30, 90), (24, 100)],
    [(38, 90), (32, 100)],
    [(46, 90), (40, 100)],
    [(54, 90), (48, 100)],
]
is_ensemble = True


def riemman_train(data, labels, dataset_info, target_names):
    clf = TRCA(dataset_info["sample_rate"], filterbank, is_ensemble)

    # Cross validator
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    # Do cross-validation
    preds = np.empty(len(labels))
    for train, test in cv.split(data, labels):
        clf.fit(data[train], labels[train])
        preds[test] = clf.predict(data[test])

    report = classification_report(labels, preds, target_names=target_names)
    print(report)
    acc = np.mean(preds == labels)
    print(acc)
    if acc <= 0.25:
        acc = np.nan
    return clf, acc


def riemman_test(clf, epoch):
    """
    This is what the real-time BCI will call.
    Parameters
    ----------
    clf : classifier trained for the specific subject
    epoch: one epoch, the current one that represents the intention of movement of the user.

    Returns Array of classification with 4 floats representing the target classification
    -------

    """
    # To load the model, just in case
    # loaded_model = pickle.load(open(filename, 'rb'))

    # To see the array of predictions
    array = clf.predict_proba(epoch)
    return array


if __name__ == "__main__":
    # Manual Inputs
    subject_id = 1  # Only two things I should be able to change
    dataset_name = "aguilera_traditional"  # Only two things I should be able to change
    array_format = True

    # Folders and paths
    dataset_foldername = dataset_name + "_dataset"
    # computer_root_path = "/Users/rosit/Documents/MCC/voting_system_platform/Datasets/"  # OMEN
    computer_root_path = "//"  # MAC
    data_path = computer_root_path + dataset_foldername
    dataset_info = datasets_basic_infos[dataset_name]

    data, y = load_data_labels_based_on_dataset(dataset_info, subject_id, data_path)
    data = np.transpose(data, (2, 1, 0))
    target_names = dataset_info["target_names"]

    print("******************************** Training ********************************")
    start = time.time()
    clf, acc = riemman_train(data, y, dataset_info, target_names)
    end = time.time()
    print("Training time: ", end - start)

    print("******************************** Test ********************************")
    epoch_number = 0
    start = time.time()
    array = riemman_test(clf, np.asarray([data[epoch_number]]))
    end = time.time()
    print("One epoch, testing time: ", end - start)
    print(target_names)
    print(
        "Probability: ", array[epoch_number]
    )  # We select the last one, the last epoch which is the current one.
    print("Real: ", y[epoch_number])
