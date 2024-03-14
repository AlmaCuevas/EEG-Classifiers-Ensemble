import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append("D:\\Users\\NewUser\\Documents\\GitHub\\voting_system_platform\\")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from share import datasets_basic_infos
from data_loaders import load_data_labels_based_on_dataset
from Code.Extraction.Entropy import extractions_train
from sklearn.preprocessing import normalize
import time

from sklearn.pipeline import make_pipeline



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



def KNN_test(knn_model,epoch):
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
    subject_id = 13  # Only two things I should be able to change
    dataset_name = 'aguilera_traditional'  # Only two things I should be able to change
    array_format = True

    # Folders and paths
    dataset_foldername = dataset_name + '_dataset'
    # computer_root_path = "/Users/rosit/Documents/MCC/voting_system_platform/Datasets/"  # OMEN
    computer_root_path = "D:\\Users\\NewUser\\Documents\\GitHub\\voting_system_platform\\Datasets\\"  # MAC
    data_path = computer_root_path + dataset_foldername
    dataset_info = datasets_basic_infos[dataset_name]

    epochs,data, y = load_data_labels_based_on_dataset(dataset_name, subject_id, data_path)
    target_names = dataset_info['target_names']
    df= extractions_train(data, y, target_names)
    df= df.to_numpy()
    labels=df[:,3]
    df=normalize(df[:,[0,1,2]], axis=0)
    print(df)
    
    print("******************************** Training ********************************")
    start = time.time()
    best_k, best_weights=KNN_optimize(df[:,[0,1,2]],labels,target_names)
    clf, acc = KNN_train(df[:,[0,1,2]],labels,target_names,best_k,best_weights)
    end = time.time()
    print("Training time: ", end - start)

    print("******************************** Test ********************************")
    epoch_number = 4
    start = time.time()
    array = KNN_test(clf, df[:,[0,1,2]])
    end = time.time()
    print("One epoch, testing time: ", end - start)
    print(target_names)
    print("Probability: " , array[epoch_number]) # We select the last one, the last epoch which is the current one.
    print("Real: ", y[epoch_number])