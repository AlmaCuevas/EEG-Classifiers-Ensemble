import pyriemann
from sklearn.model_selection import cross_val_score

from Code.data_loaders import load_data_labels_based_on_dataset
from share import datasets_basic_infos

if __name__ == '__main__':
    # Manual Inputs
    #subject_id = 1  # Only two things I should be able to change
    dataset_name = 'torres'  # Only two things I should be able to change

    # Folders and paths
    dataset_foldername = dataset_name + '_dataset'
    data_path = "/Users/almacuevas/work_projects/voting_system_platform/Datasets/" + dataset_foldername
    dataset_info = datasets_basic_infos[dataset_name]

    for subject_id in range(1, dataset_info['subjects']+1):
        # load your data
        X, y = load_data_labels_based_on_dataset(dataset_name, subject_id, data_path, array_format=True)

        # estimate covariance matrices
        cov = pyriemann.estimation.Covariances().fit_transform(X)

        # cross validation
        mdm = pyriemann.classification.MDM()

        accuracy = cross_val_score(mdm, cov, y)

        print(f"Subject {subject_id} , accuracies: {accuracy}, mean accuracy: {accuracy.mean()}")