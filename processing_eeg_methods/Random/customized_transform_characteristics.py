from collections import OrderedDict

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import classification_report

from data_utils import get_best_classificator_and_test_accuracy, classifiers, ClfSwitcher
from share import datasets_basic_infos
from data_loaders import load_data_labels_based_on_dataset
import time
import pickle
from sklearn.pipeline import Pipeline, FeatureUnion
from pathlib import Path
from mne.decoding import Vectorizer
from sklearn.preprocessing import StandardScaler
from pyriemann.estimation import ERPCovariances, XdawnCovariances, Covariances
from pyriemann.tangentspace import TangentSpace
from mne.decoding import CSP
from sklearn.decomposition import PCA

ROOT_VOTING_SYSTEM_PATH: Path = Path(__file__).parent.parent.parent.resolve()

# todo: the arrays doesn't match when I try to do this. I need a way to get the features, save them in a table.
#  and do it multiple times for delta, alpha, beta, gamma, all. For the features mentiones below
#  and for the feature extraction like entropy and stuff.

# todo: Only after getting the table I'll be able to run the feature selection
#  and then finally select the characteristics. Trying to run it here hasn't work and I should move on.

threshold_for_bug = 0.00000001  # could be any value, ex numpy.min

def customized_train(data, labels): # v1
    combined_features = FeatureUnion([
        ("Vectorizer", Vectorizer()),
        ("ERPcova", ERPCovariances(estimator='oas')),
        ("XdawnCova", XdawnCovariances(estimator='oas')),
        ("CSP", CSP(n_components=4, reg=None, log=True, norm_trace=False)),
        ("Cova", Covariances()),
                          ])

    # Use combined features to transform dataset:
    X_features = combined_features.fit(data, labels).transform(data)
    print("Combined space has", X_features.shape[1], "features")


    estimators = OrderedDict()
    estimators['Vect + StandScaler'] = Pipeline([("Union", combined_features), ("StandScaler", StandardScaler()), ('clf', ClfSwitcher())])
    estimators['ERPCov + TS'] = Pipeline([("Union", combined_features), ("ts", TangentSpace()), ('clf', ClfSwitcher())])
    estimators['CSP'] = Pipeline([("Union", combined_features), ('clf', ClfSwitcher())])
    #estimators['Cova + TS'] = Pipeline([("Cova", Covariances()), ("ts", TangentSpace()), ('clf', ClfSwitcher())]) # probably the best one, at least for Torres, is this combination

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
    #dataset_name = "torres"  # Only two things I should be able to change
    datasets = ['aguilera_traditional', 'aguilera_gamified', 'torres']
    for dataset_name in datasets:
        version_name = "customized_only" # To keep track what the output processing alteration went through

        # Folders and paths
        dataset_foldername = dataset_name + "_dataset"
        computer_root_path = str(ROOT_VOTING_SYSTEM_PATH) + "/Datasets/"
        data_path = computer_root_path + dataset_foldername
        print(data_path)
        # Initialize
        processing_name: str = ''
        if dataset_name not in [
            "aguilera_traditional",
            "aguilera_gamified",
            "nieto",
            "coretto",
            "torres",
        ]:
            raise Exception(
                f"Not supported dataset named '{dataset_name}', choose from the following: aguilera_traditional, aguilera_gamified, nieto, coretto or torres."
            )
        dataset_info: dict = datasets_basic_infos[dataset_name]

        mean_accuracy_per_subject: list = []
        results_df = pd.DataFrame()

        for subject_id in range(
            1, dataset_info["subjects"] + 1 #todo: run not normalized again. I think normalized is better though
        ):  # Only two things I should be able to change
            print(subject_id)
            with open(
                f"{str(ROOT_VOTING_SYSTEM_PATH)}/Results/{version_name}_{dataset_name}.txt",
                "a",
            ) as f:
                f.write(f"Subject: {subject_id}\n\n")
            epochs, data, labels = load_data_labels_based_on_dataset(dataset_name, subject_id, data_path)
            data[data < threshold_for_bug] = threshold_for_bug # To avoid the error "SVD did not convergence"
            # Do cross-validation
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            acc_over_cv = []
            testing_time_over_cv = []
            accuracy = 0
            for train, test in cv.split(epochs, labels):
                print(
                    "******************************** Training ********************************"
                )
                clf, accuracy, processing_name = customized_train(data[train], labels[train])
                with open(
                    f"{str(ROOT_VOTING_SYSTEM_PATH)}/Results/{version_name}_{dataset_name}.txt",
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
                    f"{str(ROOT_VOTING_SYSTEM_PATH)}/Results/{version_name}_{dataset_name}.txt",
                    "a",
                ) as f:
                    f.write(f"Prediction: {pred_list}\n")
                    f.write(f"Real label:{labels[test]}\n")
                    f.write(f"Mean accuracy in KFold: {acc}\n")
                print("Mean accuracy in KFold: ", acc)
            mean_acc_over_cv = np.mean(acc_over_cv)

            with open(
                f"{str(ROOT_VOTING_SYSTEM_PATH)}/Results/{version_name}_{dataset_name}.txt",
                "a",
            ) as f:
                f.write(f"Final acc: {mean_acc_over_cv}\n\n\n\n")
            print(f"Final acc: {mean_acc_over_cv}")

            temp = pd.DataFrame({'Methods': [processing_name] * len(acc_over_cv), 'Subject ID': [subject_id] * len(acc_over_cv),
                                 'Version': [version_name] * len(acc_over_cv), 'Training Accuracy': [accuracy] * len(acc_over_cv),
                                 'Testing Accuracy': acc_over_cv, 'Testing Time': testing_time_over_cv}) # The idea is that the most famous one is the one I use for this dataset
            results_df = pd.concat([results_df, temp])

        results_df.to_csv(
            f"{ROOT_VOTING_SYSTEM_PATH}/Results/{version_name}_{dataset_name}.csv")

    print("Congrats! The processing methods are done processing.")
