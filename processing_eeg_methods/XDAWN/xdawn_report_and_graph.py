"""
.. _ex-xdawn-decoding:

============================
XDAWN Decoding From EEG data
============================

ERP decoding with Xdawn :footcite:`RivetEtAl2009,RivetEtAl2011`. For each event
type, a set of spatial Xdawn filters are trained and applied on the signal.
Channels are concatenated and rescaled to create features vectors that will be
fed into a logistic regression.
"""
# Authors: Alexandre Barachant <alexandre.barachant@gmail.com>
#
# License: BSD-3-Clause

# %%

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from mne.preprocessing import Xdawn
from mne.decoding import Vectorizer
from share import datasets_basic_infos
from data_loaders import load_data_labels_based_on_dataset
print(__doc__)

# This one needs the data to be load in epochs, not in dimensions
#subject_id = 1
dataset_name = 'aguilera_traditional'


dataset_foldername = dataset_name + '_dataset'
computer_root_path = "/Users/almacuevas/work_projects/voting_system_platform/Datasets/" # MAC
#computer_root_path = "/Users/rosit/Documents/MCC/voting_system_platform/Datasets/"  # OMEN

data_path = computer_root_path + dataset_foldername

dataset_info = datasets_basic_infos[dataset_name]

for subject_id in range(1, dataset_info['subjects'] + 1):
    epochs, y = load_data_labels_based_on_dataset(dataset_info, subject_id, data_path)
    n_filter = 5
    # Create classification pipeline
    clf = make_pipeline(
        Xdawn(n_components=n_filter),
        Vectorizer(),
        MinMaxScaler(),
        LogisticRegression(penalty="l2", solver="lbfgs", multi_class="auto"),
    )

    # Get the labels
    labels = epochs.events[:, -1]

    # Cross validator
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Do cross-validation
    preds = np.empty(len(labels))
    for train, test in cv.split(epochs, labels):
        clf.fit(epochs[train], labels[train])
        preds[test] = clf.predict(epochs[test])

    # To see the array of predictions
    # array = clf.predict_proba(epochs[1])

    # Classification report
    target_names = dataset_info['target_names']
    report = classification_report(labels, preds, target_names=target_names)
    print(report)
    with open(f'/Users/almacuevas/work_projects/voting_system_platform/Code/XDAWN/results_xdawn_{dataset_name}.txt', 'a') as f:
        f.write(f'subject: {subject_id},\n report: {report}\n\n')

    # Normalized confusion matrix
    cm = confusion_matrix(labels, preds)
    cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix
    fig, ax = plt.subplots(1)
    im = ax.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set(title="Normalized Confusion matrix")
    fig.colorbar(im)
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    fig.tight_layout()
    ax.set(ylabel="True label", xlabel="Predicted label")
