import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold


from data_utils import get_best_classificator_and_test_accuracy, ClfSwitcher
from share import datasets_basic_infos, ROOT_VOTING_SYSTEM_PATH
from data_loaders import load_data_labels_based_on_dataset
from sklearn.pipeline import Pipeline
import time
import antropy as ant
import numpy
from sklearn.feature_selection import f_classif


def get_entropy(data):
    entropy_values = []
    for data_trial in data:
        entropy_trial = []
        for data_channel in data_trial:
            entropy_trial.append(ant.higuchi_fd(data_channel))
        entropy_values.append(np.mean(entropy_trial)) # The fastest one
    return entropy_values

def hjorth(X, D=None):
    """ Compute Hjorth mobility and complexity of a time series from either two
    cases below:
        1. X, the time series of type list (default)
        2. D, a first order differential sequence of X (if D is provided,
           recommended to speed up)

    In case 1, D is computed using Numpy's Difference function.

    Notes
    -----
    To speed up, it is recommended to compute D before calling this function
    because D may also be used by other functions whereas computing it here
    again will slow down.

    Parameters
    ----------

    X
        list

        a time series

    D
        list

        first order differential sequence of a time series

    Returns
    -------

    As indicated in return line

    Hjorth mobility and complexity

    """

    if D is None:
        D = numpy.diff(X)
        D = D.tolist()

    D.insert(0, X[0])  # pad the first difference
    D = numpy.array(D)

    n = len(X)

    M2 = float(sum(D ** 2)) / n
    TP = sum(numpy.array(X) ** 2)
    M4 = 0
    for i in range(1, len(D)):
        M4 += (D[i] - D[i - 1]) ** 2
    M4 = M4 / n

    return numpy.sqrt(M2 / TP), numpy.sqrt(float(M4) * TP / M2 / M2)  # Hjorth Mobility and Complexity

def get_hjorth(data):
    values = []
    Complexity_values = []
    for data_trial in data:
        Mobility_trial = []
        Complexity_trial = []
        for data_channel in data_trial:
            Mobility, Complexity = hjorth(data_channel)
            Mobility_trial.append(Mobility)
            Complexity_trial.append(Complexity)
        values.append(np.mean(Mobility_trial))
        Complexity_values.append(np.mean(Complexity_trial))
    return values, Complexity_values

def get_XXXX(data):
    entropy_values = []
    for data_trial in data:
        entropy_trial = []
        for data_channel in data_trial:
            entropy_trial.append(getXXXX(data_channel))
        entropy_values.append(np.mean(entropy_trial)) # The fastest one
    return entropy_values

def get_extractions(data) -> pd.DataFrame:
    # Create classification pipeline
    entropy = get_entropy(data)
    Mobility_values, Complexity_values = get_hjorth(data)
    XXXX_values = get_XXXX(data)
    df = pd.DataFrame({"entropy": entropy, "Complexity": Complexity_values, "Mobility": Mobility_values})
    return df

def extractions_train(features_df, labels): # v1
    X_SelectKBest = SelectKBest(f_classif, k=100)
    X_new = X_SelectKBest.fit_transform(features_df, labels)
    columns_list = X_SelectKBest.get_feature_names_out()
    X_new_df = pd.DataFrame(X_new, columns=columns_list)

    classifier, acc = get_best_classificator_and_test_accuracy(X_new_df, labels, Pipeline([('clf', ClfSwitcher())]))
    return classifier, acc, columns_list


def extractions_test(clf, features_df):
    """
    This is what the real-time BCI will call.
    Parameters
    ----------
    clf : classifier trained for the specific subject
    features_df: one epoch, the current one that represents the intention of movement of the user.

    Returns Array of classification with 4 floats representing the target classification
    -------

    """

    array = clf.predict_proba(features_df)

    return array


if __name__ == "__main__":
    # Manual Inputs
    datasets = ['aguilera_gamified', 'aguilera_traditional', 'torres']
    for dataset_name in datasets:
        version_name = "one_transforms_table_of_selectKbest" # To keep track what the output processing alteration went through

        # Folders and paths
        dataset_foldername = dataset_name + "_dataset"
        computer_root_path = str(ROOT_VOTING_SYSTEM_PATH) + "/Datasets/"
        data_path = computer_root_path + dataset_foldername
        print(data_path)
        # Initialize
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
                features_train_df, methods = get_extractions(data[train])
                clf, accuracy, columns_list = extractions_train(features_train_df, labels[train])
                training_time.append(time.time() - start)
                with open(
                    f"{str(ROOT_VOTING_SYSTEM_PATH)}/Results/{version_name}_{dataset_name}.txt",
                    "a",
                ) as f:
                    f.write(f"Accuracy of training: {accuracy}\n")
                print(
                    "******************************** Test ********************************"
                )
                pred_list = []
                testing_time = []
                for epoch_number in test:
                    start = time.time()
                    features_test_df, _ = get_extractions(np.asarray([data[epoch_number]]))
                    array = extractions_test(clf, features_test_df[columns_list])
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

            temp = pd.DataFrame({'Subject ID': [subject_id] * len(acc_over_cv),
                                 'Version': [version_name] * len(acc_over_cv), 'Training Accuracy': [accuracy] * len(acc_over_cv), 'Training Time': training_time,
                                 'Testing Accuracy': acc_over_cv, 'Testing Time': testing_time_over_cv}) # The idea is that the most famous one is the one I use for this dataset
            results_df = pd.concat([results_df, temp])

        results_df.to_csv(
            f"{ROOT_VOTING_SYSTEM_PATH}/Results/{version_name}_{dataset_name}.csv")

    print("Congrats! The processing methods are done processing.")
