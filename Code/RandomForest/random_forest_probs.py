import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from share import datasets_basic_infos
from data_loaders import load_data_labels_based_on_dataset
import time

from sklearn.pipeline import make_pipeline


def ranf_optimize(data,labels,target_names):
    parameters = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
    gridsearch = GridSearchCV(RandomForestClassifier(), param_grid = param_grid, cv = 10,scoring='accuracy')
    gridsearch.fit(data, labels)
    best_parameters = gridsearch.best_params_
    
    return     best_parameters


def ranf_train(data, labels, target_names,best_parameters):
    
    clf = RandomForestClassifier(best_parameters)

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

def ranf_test(clf, epoch):
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
    # computer_root_path = "/Users/rosit/Documents/MCC/voting_system_platform/Datasets/"  # OMEN
    computer_root_path = "/Users/almacuevas/work_projects/voting_system_platform/Datasets/"  # MAC
    data_path = computer_root_path + dataset_foldername
    dataset_info = datasets_basic_infos[dataset_name]

    data, y = load_data_labels_based_on_dataset(dataset_name, subject_id, data_path)
    target_names = dataset_info['target_names']

    print("******************************** Training ********************************")
    start = time.time()
    best_parameters=ranf_optimize(data,y,target_names)
    clf, acc = ranf_train(data,y,target_names,best_parameters)
    end = time.time()
    print("Training time: ", end - start)

    print("******************************** Test ********************************")
    epoch_number = 0
    start = time.time()
    array = ranf_test(clf, np.asarray([data[epoch_number]]))
    end = time.time()
    print("One epoch, testing time: ", end - start)
    print(target_names)
    print("Probability: " , array[epoch_number]) # We select the last one, the last epoch which is the current one.
    print("Real: ", y[epoch_number])
