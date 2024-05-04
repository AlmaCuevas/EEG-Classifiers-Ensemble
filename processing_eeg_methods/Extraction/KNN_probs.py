import numpy as np
import pandas as pd
from Extraction.Entropy_Eduardo import extractions_train
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2

from Extraction.get_features_probs import get_extractions
from processing_eeg_methods.share import datasets_basic_infos, ROOT_VOTING_SYSTEM_PATH
from processing_eeg_methods.data_loaders import load_data_labels_based_on_dataset
from sklearn.preprocessing import normalize
import time
from sklearn.pipeline import make_pipeline

# todo: this needs updating to new format for processing.

def KNN_optimize(data,labels,target_names):
    parameters = { "n_neighbors": range(1, 12), "weights": ["uniform", "distance"],}
    gridsearch = GridSearchCV(KNeighborsClassifier(), parameters)
    gridsearch.fit(data, labels)
    best_k = gridsearch.best_params_["n_neighbors"]
    best_weights = gridsearch.best_params_["weights"]
    print("Best N-Neighbor")
    print(best_k)
    print("Best Weight")
    print(best_weights)
    return best_k, best_weights


def KNN_train(data,labels,target_names,best_k,best_weights):
    clf = KNeighborsClassifier(n_neighbors=best_k, weights=best_weights)

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



if __name__ == '__main__':
    # Manual Inputs
    subject_id = 7 # Only two things I should be able to change
    dataset_name = 'aguilera_traditional'  # Only two things I should be able to change

    # Folders and paths
    dataset_foldername = dataset_name + '_dataset'
    computer_root_path = ROOT_VOTING_SYSTEM_PATH + "/Datasets/"
    data_path = computer_root_path + dataset_foldername
    dataset_info = datasets_basic_infos[dataset_name]

    _, data, labels = load_data_labels_based_on_dataset(dataset_info, subject_id, data_path, threshold_for_bug = 0.00000001)

    features_array = get_extractions(data, dataset_info)

    features_array=normalize(features_array, axis=0)+1
    target_names = dataset_info['target_names']
    features_array_new = SelectKBest(chi2, k=10).fit_transform(features_array, labels)
    best_k, best_weights=KNN_optimize(features_array_new,labels,target_names)
    KNN_train(data, labels, target_names, best_k, best_weights)
