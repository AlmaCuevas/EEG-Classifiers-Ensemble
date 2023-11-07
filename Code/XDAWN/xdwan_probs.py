import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from mne.preprocessing import Xdawn
from mne.decoding import Vectorizer
from share import datasets_basic_infos
from data_loaders import load_data_labels_based_on_dataset
import time
import pickle

def xdawn_train(epochs, labels):
    n_filter = 5
    # Create classification pipeline
    clf = make_pipeline(
        Xdawn(n_components=n_filter),
        Vectorizer(),
        MinMaxScaler(),
        LogisticRegression(penalty="l2", solver="lbfgs", multi_class="auto"),
    )

    # Cross validator
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Do cross-validation
    preds = np.empty(len(labels))
    for train, test in cv.split(epochs, labels):
        clf.fit(epochs[train], labels[train])
        preds[test] = clf.predict(epochs[test])

    # save the model to disk
    #filename = f'/Users/almacuevas/work_projects/voting_system_platform/Results/xdawn_model_{dataset_name}_sub{subject_id}.sav'
    #pickle.dump(clf, open(filename, 'wb'))
    target_names = dataset_info['target_names']
    report = classification_report(labels, preds, target_names=target_names)
    print(report)
    return clf

def xdawn_test(clf, epoch):
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
    #loaded_model = pickle.load(open(filename, 'rb'))

    # To see the array of predictions
    array = clf.predict_proba(epoch)
    return array

if __name__ == '__main__':
    # Manual Inputs
    subject_id = 1  # Only two things I should be able to change
    dataset_name = 'aguilera'  # Only two things I should be able to change
    array_format = False

    # Folders and paths
    dataset_foldername = dataset_name + '_dataset'
    # computer_root_path = "/Users/rosit/Documents/MCC/voting_system_platform/Datasets/"  # OMEN
    computer_root_path = "/Users/almacuevas/work_projects/voting_system_platform/Datasets/"  # MAC
    data_path = computer_root_path + dataset_foldername
    dataset_info = datasets_basic_infos[dataset_name]

    epochs, labels = load_data_labels_based_on_dataset(dataset_name, subject_id, data_path, array_format=array_format)
    target_names = dataset_info['target_names']

    print("******************************** Training ********************************")
    start = time.time()
    clf = xdawn_train(epochs, labels)
    end = time.time()
    print("Training time: ", end - start)

    print("******************************** Test ********************************")
    epoch_number = 0
    start = time.time()
    array = xdawn_test(clf, epochs[epoch_number])
    end = time.time()
    print("One epoch, testing time: ", end - start)
    print(target_names)
    print("Probability: " , array)
    print("Real: ", labels[epoch_number])
