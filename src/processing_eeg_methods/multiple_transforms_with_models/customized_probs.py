import time
from collections import OrderedDict

import numpy as np
import pandas as pd
from data_loaders import load_data_labels_based_on_dataset
from data_utils import (
    ClfSwitcher,
    get_best_classificator_and_test_accuracy,
    get_dataset_basic_info,
    get_input_data_path,
    standard_saving_path,
)
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from share import ROOT_VOTING_SYSTEM_PATH, datasets_basic_infos
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

# todo: add the test template
# todo: do the deap thing about the FFT: https://github.com/tongdaxu/EEG_Emotion_Classifier_DEAP/blob/master/Preprocess_Deap.ipynb


def customized_train(data, labels):  # v1

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
    array = clf.predict_proba(trial)
    return array


if __name__ == "__main__":
    # Manual Inputs
    datasets = ["ic_bci_2020"]
    for dataset_name in datasets:
        version_name = "only_customized_two_classes_12_no_preprocess"
        processing_name: str = ""

        data_path: str = get_input_data_path(dataset_name)
        dataset_info: dict = get_dataset_basic_info(datasets_basic_infos, dataset_name)

        saving_txt_path: str = standard_saving_path(
            dataset_info, processing_name, version_name
        )

        mean_accuracy_per_subject: list = []
        results_df = pd.DataFrame()

        for subject_id in range(
            1, dataset_info["subjects"] + 1
        ):  # Only two things I should be able to change
            print(subject_id)
            with open(
                saving_txt_path,
                "a",
            ) as f:
                f.write(f"Subject: {subject_id}\n\n")
            epochs, data, labels = load_data_labels_based_on_dataset(
                dataset_info,
                subject_id,
                data_path,
                selected_classes=[1, 2],
                threshold_for_bug=0.00000001,
            )  # could be any value, ex numpy.min
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
                clf, accuracy, processing_name = customized_train(
                    data[train], labels[train]
                )
                training_time.append(time.time() - start)
                with open(
                    saving_txt_path,
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
                    saving_txt_path,
                    "a",
                ) as f:
                    f.write(f"Prediction: {pred_list}\n")
                    f.write(f"Real label:{labels[test]}\n")
                    f.write(f"Mean accuracy in KFold: {acc}\n")
                print("Mean accuracy in KFold: ", acc)
            mean_acc_over_cv = np.mean(acc_over_cv)

            with open(
                saving_txt_path,
                "a",
            ) as f:
                f.write(f"Final acc: {mean_acc_over_cv}\n\n\n\n")
            print(f"Final acc: {mean_acc_over_cv}")

            temp = pd.DataFrame(
                {
                    "Methods": [processing_name] * len(acc_over_cv),
                    "Subject ID": [subject_id] * len(acc_over_cv),
                    "Version": [version_name] * len(acc_over_cv),
                    "Training Accuracy": [accuracy] * len(acc_over_cv),
                    "Training Time": training_time,
                    "Testing Accuracy": acc_over_cv,
                    "Testing Time": testing_time_over_cv,
                }
            )  # The idea is that the most famous one is the one I use for this dataset
            results_df = pd.concat([results_df, temp])

        results_df.to_csv(
            f"{ROOT_VOTING_SYSTEM_PATH}/Results/{dataset_name}/{version_name}_{processing_name}_{dataset_name}.csv"
        )

    print("Congrats! The processing methods are done processing.")
