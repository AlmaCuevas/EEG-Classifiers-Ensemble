from collections import OrderedDict

import numpy as np

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import classification_report

from data_utils import get_best_classificator_and_test_accuracy, classifiers, ClfSwitcher
from share import datasets_basic_infos
from data_loaders import load_data_labels_based_on_dataset
import time
import pickle
from sklearn.pipeline import Pipeline
from pathlib import Path
from mne.decoding import Vectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pyriemann.estimation import ERPCovariances, XdawnCovariances, Covariances
from pyriemann.tangentspace import TangentSpace
from mne.decoding import CSP

ROOT_VOTING_SYSTEM_PATH: Path = Path(__file__).parent.parent.parent.resolve()


# todo: do the deap thing about the FFT: https://github.com/tongdaxu/EEG_Emotion_Classifier_DEAP/blob/master/Preprocess_Deap.ipynb
def customized_train(data, labels): # v1

    estimators = OrderedDict()
    estimators['Vect + StandScaler'] = Pipeline([("Vectorizer", Vectorizer()), ("StandScaler", StandardScaler()), ('clf', ClfSwitcher())])
    estimators['Vect'] = Pipeline([("Vectorizer", Vectorizer()), ('clf', ClfSwitcher())])
    estimators['ERPCov + TS'] = Pipeline([("ERPcova", ERPCovariances(estimator='oas')), ("ts", TangentSpace()), ('clf', ClfSwitcher())])
    estimators['ERPCov'] = Pipeline([("ERPcova", ERPCovariances(estimator='oas')), ('clf', ClfSwitcher())])
    estimators['XdawnCov + TS'] = Pipeline([("XdawnCova", XdawnCovariances(estimator='oas')), ("ts", TangentSpace()), ('clf', ClfSwitcher())])
    estimators['CSP'] = Pipeline([("CSP", CSP(n_components=4, reg=None, log=True, norm_trace=False)), ('clf', ClfSwitcher())])
    estimators['Cova + TS'] = Pipeline([("Cova", Covariances()), ("ts", TangentSpace()), ('clf', ClfSwitcher())])

    parameters = []
    for classificator in classifiers:
        parameters.append({'clf__estimator': [classificator]})

    accuracy_list = []
    classifiers_list=[]
    for name, clf  in estimators.items():
        print(name)
        classifier, acc = get_best_classificator_and_test_accuracy(data, labels, clf, param_grid=parameters)
        accuracy_list.append(acc)
        classifiers_list.append(classifier)

    return classifiers_list[np.argmax(accuracy_list)], accuracy_list[np.argmax(accuracy_list)]

def customized_test(clf, epoch):
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
    # loaded_model = pickle.load(open(filename, 'rb'))

    # To see the array of predictions
    array = clf.predict_proba(epoch)
    return array


if __name__ == "__main__":
    # Manual Inputs
    subject_id = 1  # Only two things I should be able to change
    dataset_name = "aguilera_gamified"  # Only two things I should be able to change
    array_format = True

    # Folders and paths
    dataset_foldername = dataset_name + "_dataset"
    computer_root_path = str(ROOT_VOTING_SYSTEM_PATH) + "/Datasets/"
    data_path = computer_root_path + dataset_foldername
    dataset_info = datasets_basic_infos[dataset_name]

    epochs, data, labels = load_data_labels_based_on_dataset(
        dataset_name, subject_id, data_path
    )
    target_names = dataset_info["target_names"]

    threshold_for_bug = 0.00000001  # could be any value, ex numpy.min
    data[data < threshold_for_bug] = threshold_for_bug

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )

    print("******************************** Training ********************************")
    start = time.time()
    clf, acc = customized_train(X_train, y_train)
    end = time.time()
    print("Training time: ", end - start)

    print("******************************** Test ********************************")
    epoch_number = 0
    start = time.time()
    array = customized_test(clf, np.asarray([X_test[epoch_number]]))
    end = time.time()
    print("One epoch, testing time: ", end - start)
    print(target_names)
    print("Prediction: ", np.argmax(array))
    print("Real: ", y_test[epoch_number])
