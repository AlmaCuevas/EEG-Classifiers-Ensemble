from collections import OrderedDict

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from data_utils import get_best_classificator_and_test_accuracy, ClfSwitcher
from share import datasets_basic_infos, ROOT_VOTING_SYSTEM_PATH
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
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from scipy import signal
import mne

# todo: add the test template
# todo: do the deap thing about the FFT: https://github.com/tongdaxu/EEG_Emotion_Classifier_DEAP/blob/master/Preprocess_Deap.ipynb

frequency_ranges: dict = {
    "complete": [0, 140],
    "delta": [0, 3],
    "theta": [3, 7],
    "alpha": [7, 13],
    "beta 1": [13, 16],
    "beta 2": [16, 20],
    "beta 3": [20, 35],
    "gamma": [35, 140],
}

def custom_feature_engineering(X): # ('feature_engineering', FunctionTransformer(custom_feature_engineering)),
    for frequency_bandwidth_name, frequency_bandwidth in frequency_ranges.items():
        print(frequency_bandwidth)
        iir_params = dict(order=8, ftype="butter")
        filt = mne.filter.create_filter(
            X, dataset_info['sample_rate'], l_freq=frequency_bandwidth[0], h_freq=frequency_bandwidth[1],
            method="iir", iir_params=iir_params, verbose=True
        )
        filtered = signal.sosfiltfilt(filt["sos"], X)
    return X # todo: i do not know how to use it

def customized_train(data, labels): # v1

    estimators = OrderedDict()
    # Do not use 'Vect' transform, most of the time is nan or 0.25 if anything.
    #estimators['ERPCov + TS'] = Pipeline([("ERPcova", ERPCovariances(estimator='oas')), ("ts", TangentSpace()), ('clf', ClfSwitcher())])
    #estimators['XdawnCov + TS'] = Pipeline([("XdawnCova", XdawnCovariances(estimator='oas')), ("ts", TangentSpace()), ('clf', ClfSwitcher())])
    #estimators['CSP'] = Pipeline( [ ("CSP", CSP(n_components=4, reg=None, log=True, norm_trace=False)), ('clf', ClfSwitcher())]) # Get into cov.py and do copy='auto' https://stackoverflow.com/questions/76431070/mne-valueerror-data-copying-was-not-requested-by-copy-none-but-it-was-require
    estimators['Cova + TS'] = Pipeline([("Cova", Covariances()), ("ts", TangentSpace()), ('clf', ClfSwitcher())]) # This is probably the best one, at least for Torres

    accuracy_list = []
    classifiers_list=[]
    for name, clf  in estimators.items():
        print(name)
        classifier, acc = get_best_classificator_and_test_accuracy(data, labels, clf)
        accuracy_list.append(acc)
        classifiers_list.append(classifier)
    print(estimators.keys())
    print(accuracy_list)
    return classifiers_list[np.argmax(accuracy_list)], accuracy_list[np.argmax(accuracy_list)], list(estimators.keys())[np.argmax(accuracy_list)]

def customized_test(clf, trial):
    """
    This is what the real-time BCI will call.
    Parameters
    ----------
    clf : classifier trained for the specific subject
    trial: one epoch, the current one that represents the intention of movement of the user.

    Returns Array of classification with 4 floats representing the target classification
    -------

    """
    # To load the model, just in case
    # loaded_model = pickle.load(open(filename, 'rb'))

    # To see the array of predictions
    array = clf.predict_proba(trial)
    return array


if __name__ == "__main__":
    # Manual Inputs
    datasets = ['aguilera_traditional']
    for dataset_name in datasets:
        version_name = "only_customized_two_classes_03_no_preprocess" # To keep track what the output processing alteration went through

        # Folders and paths
        dataset_foldername = dataset_name + "_dataset"
        computer_root_path = ROOT_VOTING_SYSTEM_PATH + "/Datasets/"
        data_path = computer_root_path + dataset_foldername
        print(data_path)
        # Initialize
        processing_name: str = ''
        if dataset_name not in datasets_basic_infos:
            raise Exception(
                f"Not supported dataset named '{dataset_name}', choose from the following: aguilera_traditional, aguilera_gamified, nieto, coretto or torres."
            )
        dataset_info: dict = datasets_basic_infos[dataset_name]

        mean_accuracy_per_subject: list = []
        results_df = pd.DataFrame()

        for subject_id in range(
            1, dataset_info["subjects"] + 1
        ):  # Only two things I should be able to change
            print(subject_id)
            with open(
                f"{ROOT_VOTING_SYSTEM_PATH}/Results/{version_name}_{dataset_name}.txt",
                "a",
            ) as f:
                f.write(f"Subject: {subject_id}\n\n")
            epochs, data, labels = load_data_labels_based_on_dataset(dataset_info, subject_id, data_path, selected_classes=[0, 3], threshold_for_bug = 0.00000001)  # could be any value, ex numpy.min
            # Do cross-validation
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            acc_over_cv = []
            testing_time_over_cv = []
            training_time = []
            accuracy = 0
            for train, test in cv.split(epochs, labels):
                print(
                    "******************************** Training ********************************"
                )
                start = time.time()
                clf, accuracy, processing_name = customized_train(data[train], labels[train])
                training_time.append(time.time() - start)
                with open(
                    f"{ROOT_VOTING_SYSTEM_PATH}/Results/{version_name}_{dataset_name}.txt",
                    "a",
                ) as f:
                    f.write(f"{processing_name}\n")
                    f.write(f"Accuracy of training: {accuracy}\n")
                print(
                    "******************************** Test ********************************"
                )
                pred_list = []
                testing_time = []
                for epoch_number in test:
                    start = time.time()
                    array = customized_test(clf, np.asarray([data[epoch_number]]))
                    end = time.time()
                    testing_time.append(end - start)
                    print(dataset_info["target_names"])
                    print("Probability voting system: ", array)

                    voting_system_pred = np.argmax(array)
                    pred_list.append(voting_system_pred)
                    print("Prediction: ", voting_system_pred)
                    print("Real: ", labels[epoch_number])

                acc = np.mean(pred_list == labels[test])
                testing_time_over_cv.append(np.mean(testing_time))
                acc_over_cv.append(acc)
                with open(
                    f"{ROOT_VOTING_SYSTEM_PATH}/Results/{version_name}_{dataset_name}.txt",
                    "a",
                ) as f:
                    f.write(f"Prediction: {pred_list}\n")
                    f.write(f"Real label:{labels[test]}\n")
                    f.write(f"Mean accuracy in KFold: {acc}\n")
                print("Mean accuracy in KFold: ", acc)
            mean_acc_over_cv = np.mean(acc_over_cv)

            with open(
                f"{ROOT_VOTING_SYSTEM_PATH}/Results/{version_name}_{dataset_name}.txt",
                "a",
            ) as f:
                f.write(f"Final acc: {mean_acc_over_cv}\n\n\n\n")
            print(f"Final acc: {mean_acc_over_cv}")

            temp = pd.DataFrame({'Methods': [processing_name] * len(acc_over_cv), 'Subject ID': [subject_id] * len(acc_over_cv),
                                 'Version': [version_name] * len(acc_over_cv), 'Training Accuracy': [accuracy] * len(acc_over_cv), 'Training Time': training_time,
                                 'Testing Accuracy': acc_over_cv, 'Testing Time': testing_time_over_cv}) # The idea is that the most famous one is the one I use for this dataset
            results_df = pd.concat([results_df, temp])

        results_df.to_csv(
            f"{ROOT_VOTING_SYSTEM_PATH}/Results/{version_name}_{dataset_name}.csv")

    print("Congrats! The processing methods are done processing.")
