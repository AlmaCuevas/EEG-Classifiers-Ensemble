import sys
sys.path.append("D:\\Users\\NewUser\\Documents\\GitHub\\voting_system_platform\\")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from share import datasets_basic_infos
from data_loaders import load_data_labels_based_on_dataset
from sklearn.pipeline import Pipeline
import time
import antropy as ant
from pathlib import Path
import numpy

ROOT_VOTING_SYSTEM_PATH: Path = Path(__file__).parent.parent.parent.resolve()

def get_entropy(data):
    entropy_values = []
    for data_trial in data:
        entropy_trial = []
        for data_channel in data_trial:
            entropy_trial.append(ant.higuchi_fd(data_channel))
        entropy_values.append(np.mean(entropy_trial)) # The fastest one
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

# def get_XXXX(data):
    entropy_values = []
    for data_trial in data:
        entropy_trial = []
        for data_channel in data_trial:
            entropy_trial.append(XXXX(data_channel))
        entropy_values.append(np.mean(entropy_trial)) # The fastest one
    return entropy_values

def extractions_train(data, labels, target_names):
    # Create classification pipeline
    entropy = get_entropy(data)
    Mobility_values, Complexity_values = get_hjorth(data)
    #XXXX_values = get_XXXX(data)
    df = pd.DataFrame({"entropy": entropy, "Complexity":Complexity_values, "Mobility":Mobility_values, "labels": labels})
    # df.to_csv("/Users/almacuevas/work_projects/voting_system_platform/Code/df_extractions.csv")
    # df= df.to_numpy()
    print("Characteristics extraction done")
    print(df)
    return df

    ## Use scikit-learn Pipeline with cross_val_score function
    #clf = Pipeline([])
    #
    #preds = np.zeros(len(labels))
    #cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    #for train_idx, test_idx in cv.split(data, labels):
    #    y_train, y_test = labels[train_idx], labels[test_idx]
    #
    #    clf.fit(data[train_idx], y_train) # Does this means that everytime that I fit it, it forget the previous? so when I save the clf, I only save the last one?
    #    preds[test_idx] = clf.predict(data[test_idx])
    #report = classification_report(labels, preds, target_names=target_names)
    #print(report)
    #acc = np.mean(preds == labels)
    #print(acc)
    #if acc <= 0.25:
    #    acc = np.nan
    #return clf, acc


def extractions_test(clf, epoch):
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
    dataset_name = 'aguilera_traditional'  # Only two things I should be able to change
    array_format = True

    # Folders and paths
    dataset_foldername = dataset_name + '_dataset'
    computer_root_path = str(ROOT_VOTING_SYSTEM_PATH) + "/Datasets/"
    data_path = computer_root_path + dataset_foldername
    dataset_info = datasets_basic_infos[dataset_name]

    _, data, labels = load_data_labels_based_on_dataset(dataset_name, subject_id, data_path)
    target_names = dataset_info['target_names']

    print("******************************** Training ********************************")
    start = time.time()
    #clf, acc =
    extractions_train(data, labels, target_names)
    #end = time.time()
    #print("Training time: ", end - start)
    #
    #print("******************************** Test ********************************")
    #epoch_number = 0
    #start = time.time()
    #array = extractions_test(clf, np.asarray([data[epoch_number]]))
    #end = time.time()
    #print("One epoch, testing time: ", end - start)
    #print(target_names)
    #print("Probability: " , array)
    #print("Real: ", labels[epoch_number])