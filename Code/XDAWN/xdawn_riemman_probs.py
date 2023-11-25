import numpy as np

from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.pipeline import make_pipeline

from matplotlib import pyplot as plt
from share import datasets_basic_infos
from data_loaders import load_data_labels_based_on_dataset
import time

def xdawn_riemman_train(data, labels, target_names):
    n_components = 2  # pick some components

    # Define a monte-carlo cross-validation generator (reduce variance):
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    clf = make_pipeline(
        XdawnCovariances(n_components),
        TangentSpace(metric="riemann"),
        LogisticRegression(),
    )

    preds = np.zeros(len(labels))

    for train_idx, test_idx in cv.split(data):
        y_train, y_test = labels[train_idx], labels[test_idx]

        clf.fit(data[train_idx], y_train)
        preds[test_idx] = clf.predict(data[test_idx])
    report = classification_report(labels, preds, target_names=target_names)
    print(report)
    return clf

def xdawn_riemman_test(clf, epoch):
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
    subject_id = 3  # Only two things I should be able to change
    dataset_name = 'aguilera_traditional'  # Only two things I should be able to change
    array_format = True

    # Folders and paths
    dataset_foldername = dataset_name + '_dataset'
    # computer_root_path = "/Users/rosit/Documents/MCC/voting_system_platform/Datasets/"  # OMEN
    computer_root_path = "/Users/almacuevas/work_projects/voting_system_platform/Datasets/"  # MAC
    data_path = computer_root_path + dataset_foldername
    dataset_info = datasets_basic_infos[dataset_name]

    data, y = load_data_labels_based_on_dataset(dataset_name, subject_id, data_path, array_format=array_format)
    target_names = dataset_info['target_names']

    print("******************************** Training ********************************")
    start = time.time()
    clf = xdawn_riemman_train(data, y, target_names)
    end = time.time()
    print("Training time: ", end - start)

    print("******************************** Test ********************************")
    epoch_number = 0
    start = time.time()
    array = xdawn_riemman_test(clf, np.asarray([data[epoch_number]]))
    end = time.time()
    print("One epoch, testing time: ", end - start)
    print(target_names)
    print("Probability: " , array[0]) # We select the last one, the last epoch which is the current one.
    print("Real: ", y[0])
