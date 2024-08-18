from collections import OrderedDict

import numpy as np
from data_utils import ClfSwitcher, get_best_classificator_and_test_accuracy
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import Pipeline

# todo: add the test template
# todo: do the deap thing about the FFT: https://github.com/tongdaxu/EEG_Emotion_Classifier_DEAP/blob/master/Preprocess_Deap.ipynb


def simplified_spatial_features_train(data, labels):  # v1

    estimators = OrderedDict()
    # Do not use 'Vect' transform, most of the time is nan or 0.25 if anything.
    # estimators['ERPCov + TS'] = Pipeline([("ERPcova", ERPCovariances(estimator='oas')), ("ts", TangentSpace()), ('clf', ClfSwitcher())]) #noqa
    # estimators['XdawnCov + TS'] = Pipeline([("XdawnCova", XdawnCovariances(estimator='oas')), ("ts", TangentSpace()), ('clf', ClfSwitcher())]) #noqa
    # estimators['CSP'] = Pipeline( [ ("CSP", CSP(n_components=4, reg=None, log=True, norm_trace=False)), ('clf', ClfSwitcher())]) # Get into cov.py and do copy='auto' https://stackoverflow.com/questions/76431070/mne-valueerror-data-copying-was-not-requested-by-copy-none-but-it-was-require #noqa
    estimators["Cova + TS"] = Pipeline(
        [("Cova", Covariances()), ("ts", TangentSpace()), ("clf", ClfSwitcher())]
    )  # This is probably the best one, at least for Torres

    accuracy_list = []
    classifiers_list = []
    for name, clf in estimators.items():
        print(name)
        classifier, acc = get_best_classificator_and_test_accuracy(data, labels, clf)
        accuracy_list.append(acc)
        classifiers_list.append(classifier)
    print(estimators.keys())
    print(accuracy_list)
    return (
        classifiers_list[np.argmax(accuracy_list)],
        accuracy_list[np.argmax(accuracy_list)],
        list(estimators.keys())[np.argmax(accuracy_list)],
    )


def simplified_spatial_features_test(clf, trial):
    """
    This is what the real-time BCI will call.
    Parameters
    ----------
    clf : classifier trained for the specific subject
    trial: one epoch, the current one that represents the intention of movement of the user.

    Returns Array of classification with 4 floats representing the target classification
    -------

    """
    array = clf.predict_proba(trial)
    return array
