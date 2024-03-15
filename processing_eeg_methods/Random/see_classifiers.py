# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from mne.decoding import CSP
from data_loaders import load_data_labels_based_on_dataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pyriemann.classification import MDM

ROOT_VOTING_SYSTEM_PATH: Path = Path(__file__).parent.parent.parent.resolve()

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
    "LDA",
    "MDM",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, random_state=42),
    SVC(gamma=2, C=1, random_state=42),
    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    DecisionTreeClassifier(max_depth=5, random_state=42),
    RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=42
    ),
    MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    AdaBoostClassifier(algorithm="SAMME", random_state=42),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    LinearDiscriminantAnalysis(),
    MDM()
]

datasets = ['aguilera_gamified', 'aguilera_traditional', 'torres']

for ds_cnt, ds in enumerate(datasets):
    # Manual Inputs
    subject_id = 1  # Only two things I should be able to change

    print(ROOT_VOTING_SYSTEM_PATH)
    # Folders and paths
    dataset_foldername = ds + '_dataset'
    computer_root_path = str(ROOT_VOTING_SYSTEM_PATH) + "/Datasets/"
    data_path = computer_root_path + dataset_foldername

    epochs, data, y  = load_data_labels_based_on_dataset(ds, subject_id, data_path)


    threshold_for_bug = 0.00000001  # could be any value, ex numpy.min
    data[data < threshold_for_bug] = threshold_for_bug

    figure = plt.figure(figsize=(27, 9))
    i = 1
    # iterate over datasets

    # preprocess dataset, split into training and test part
    X_train, X_test, y_train, y_test = train_test_split(
        data, y, test_size=0.4, random_state=42
    )
    csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)
    X_train = csp.fit_transform(X_train, y_train)
    X_test = csp.transform(X_test)

    X = csp.transform(data)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # just plot the dataset first
    cm = 'viridis'
    cm_bright = 'viridis'
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    # Plot the testing points
    ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        DecisionBoundaryDisplay.from_estimator(
            clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
        )

        # Plot the training points
        ax.scatter(
            X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
        )
        # Plot the testing points
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6,
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(
            x_max - 0.3,
            y_min + 0.3,
            ("%.2f" % score).lstrip("0"),
            size=15,
            horizontalalignment="right",
        )
        i += 1

plt.tight_layout()
plt.show()


# To figure it out later (maybe its not worth it, but its a nice way to make a graph)

# Compute w_times using the adjusted w_start array
# plt.figure()
# plt.plot(time, accuracy, label="Mean Performance")
# plt.axvline(0, linestyle="--", color="k", label="Onset")
# plt.axhline(0.5, linestyle="-", color="k", label="Chance")
# plt.xlabel("Time (s)")
# plt.ylabel("Classification Accuracy")
# plt.title("Mean Classification Performance over Time for Selected Subjects")
# plt.legend(loc="lower right")
# plt.show()