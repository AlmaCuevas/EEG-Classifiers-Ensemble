import numpy as np

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.metrics import classification_report

from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from share import datasets_basic_infos
from data_loaders import load_data_labels_based_on_dataset
import time

# Note: Riemmann can only test with a minimum of 2 epochs, so you have to give the current epoch and the last too.
# TODO: Verify that this works in real time

def riemman_train(data, labels):
    covest = Covariances()
    ts = TangentSpace()
    svc = SVC(kernel='linear', probability=True)
    clf = make_pipeline(covest, ts, svc)

    # Cross validator
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Do cross-validation
    preds = np.empty(len(labels))
    for train, test in cv.split(data, labels):
        clf.fit(data[train], labels[train])
        preds[test] = clf.predict(data[test])

    target_names = dataset_info['target_names']
    report = classification_report(labels, preds, target_names=target_names)
    print(report)
    return clf

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
    #loaded_model = pickle.load(open(filename, 'rb'))

    # To see the array of predictions
    array = clf.predict_proba(epoch)
    return array

if __name__ == '__main__':
    # Manual Inputs
    subject_id = 1  # Only two things I should be able to change
    dataset_name = 'aguilera'  # Only two things I should be able to change
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
    clf = riemman_train(data, y)
    end = time.time()
    print("Training time: ", end - start)

    print("******************************** Test ********************************")
    epoch_number = [0, 1]
    start = time.time()
    array = riemman_test(clf, data[epoch_number])
    end = time.time()
    print("One epoch, testing time: ", end - start)
    print(target_names)
    print("Probability: " , array[1]) # We select the last one, the last epoch which is the current one.
    print("Real: ", y[1])