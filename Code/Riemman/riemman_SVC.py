from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from Code.data_loaders import load_data_labels_based_on_dataset
from share import datasets_basic_infos

import numpy as np
if __name__ == '__main__':
    # Manual Inputs

    #subject_id = 1  # Only two things I should be able to change
    dataset_name = 'torres'  # Only two things I should be able to change

    # Folders and paths
    dataset_foldername = dataset_name + '_dataset'
    data_path = "/Users/almacuevas/work_projects/voting_system_platform/Datasets/" + dataset_foldername
    dataset_info = datasets_basic_infos[dataset_name]

    sum_acc = []
    for subject_id in range(1, dataset_info['subjects']+1):
        # load your data
        X, y = load_data_labels_based_on_dataset(dataset_name, subject_id, data_path, array_format=True)

        # build your pipeline
        covest = Covariances()
        ts = TangentSpace()
        svc = SVC(kernel='linear')
        clf = make_pipeline(covest, ts, svc)

        # cross validation
        accuracy = cross_val_score(clf, X, y)
        sum_acc.append(accuracy.mean())
        # The output is not the array of commands that I need, is literally the accuracies of the CV
        print(f"Subject {subject_id} , accuracies: {accuracy}, mean accuracy: {accuracy.mean()}")

    print("Final accuracy: ", np.average(sum_acc))
