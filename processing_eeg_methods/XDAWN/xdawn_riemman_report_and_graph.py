import numpy as np

from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline

from matplotlib import pyplot as plt
from share import datasets_basic_infos
from data_loaders import load_data_labels_based_on_dataset

# This one needs the data to be load in epochs, not in dimensions
#subject_id = 1
dataset_name = 'torres'


dataset_foldername = dataset_name + '_dataset'
computer_root_path = "/Users/almacuevas/work_projects/voting_system_platform/Datasets/" # MAC
#computer_root_path = "/Users/rosit/Documents/MCC/voting_system_platform/Datasets/"  # OMEN

data_path = computer_root_path + dataset_foldername

dataset_info = datasets_basic_infos[dataset_name]
sum_accu = []
for subject_id in range(1, dataset_info['subjects'] + 1):
    epochs, labels = load_data_labels_based_on_dataset(dataset_name, subject_id, data_path)
    n_components = 2  # pick some components

    # Define a monte-carlo cross-validation generator (reduce variance):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    epochs_data = epochs.get_data()

    clf = make_pipeline(
        XdawnCovariances(n_components),
        TangentSpace(metric="riemann"),
        LogisticRegression(),
    )

    preds = np.zeros(len(labels))

    for train_idx, test_idx in cv.split(epochs_data):
        y_train, y_test = labels[train_idx], labels[test_idx]

        clf.fit(epochs_data[train_idx], y_train)
        preds[test_idx] = clf.predict(epochs_data[test_idx])

    # Printing the results
    acc = np.mean(preds == labels)
    sum_accu.append(acc)
    print("Classification accuracy: %f " % (acc))

    names = dataset_info['target_names']
    cm = confusion_matrix(labels, preds)
    ConfusionMatrixDisplay(cm, display_labels=names).plot()
    plt.show()
print("Total accuracy: ", np.average(sum_accu))
