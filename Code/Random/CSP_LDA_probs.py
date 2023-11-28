import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from share import datasets_basic_infos
from data_loaders import load_data_labels_based_on_dataset
import time
import pickle
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP

def CSP_LDA_train(data, labels, target_names):
    # Create classification pipeline
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

    # Use scikit-learn Pipeline with cross_val_score function
    clf = Pipeline([("CSP", csp), ("LDA", lda)])

    preds = np.zeros(len(labels))
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train_idx, test_idx in cv.split(data, labels):
        y_train, y_test = labels[train_idx], labels[test_idx]

        clf.fit(data[train_idx], y_train) # Does this means that everytime that I fit it, it forget the previous? so when I save the clf, I only save the last one?
        preds[test_idx] = clf.predict(data[test_idx])
    report = classification_report(labels, preds, target_names=target_names)
    print(report)
    acc = np.mean(preds == labels)
    print(acc)
    if acc <= 0.25:
        acc = np.nan
    return clf, acc

def CSP_LDA_test(clf, epoch):
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

    data, labels = load_data_labels_based_on_dataset(dataset_name, subject_id, data_path, array_format=array_format)
    target_names = dataset_info['target_names']

    print("******************************** Training ********************************")
    start = time.time()
    clf, acc = CSP_LDA_train(data, labels, target_names)
    end = time.time()
    print("Training time: ", end - start)

    print("******************************** Test ********************************")
    epoch_number = 0
    start = time.time()
    array = CSP_LDA_test(clf, np.asarray([data[epoch_number]]))
    end = time.time()
    print("One epoch, testing time: ", end - start)
    print(target_names)
    print("Probability: " , array)
    print("Real: ", labels[epoch_number])
