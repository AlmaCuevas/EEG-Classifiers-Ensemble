from share import datasets_basic_infos
from data_loaders import load_data_labels_based_on_dataset
import time

#TODO: Load the CCSPNET file and divide it in train and test, find a way to provide the array from pairs.
# project_combine_cnn_lstm_run_3.ipynb does something similar, it didn't gave good result but you could try.

def ccspnet_train(data, y):

    return clf

def ccspnet_test(clf, epoch):
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
    #loaded_model = pickle.load(open(filename, 'rb'))

    # To see the array of predictions
    array = clf.predict_proba(epoch)
    return array

if __name__ == '__main__':
    # Manual Inputs
    subject_id = 1  # Only two things I should be able to change
    dataset_name = 'aguilera'  # Only two things I should be able to change
    array_format = True

    # Folders and paths
    dataset_foldername = dataset_name + '_dataset'
    # computer_root_path = "/Users/rosit/Documents/MCC/voting_system_platform/Datasets/"  # OMEN
    computer_root_path = "/Users/almacuevas/work_projects/voting_system_platform/Datasets/"  # MAC
    data_path = computer_root_path + dataset_foldername
    dataset_info = datasets_basic_infos[dataset_name]

    data, y = load_data_labels_based_on_dataset(dataset_name, subject_id, data_path, array_format=array_format)
    target_names = dataset_info['target_names']

    print("******************************** Training ********************************")
    start = time.time()
    clf = ccspnet_train(data, y)
    end = time.time()
    print("Training time: ", end - start)

    print("******************************** Test ********************************")
    epoch_number = 0
    start = time.time()
    xdawn_array = ccspnet_test(clf, data[epoch_number])
    end = time.time()
    print("One epoch, testing time: ", end - start)
    print(target_names)
    print("Probability: " , xdawn_array)
    print("Real: ", y[epoch_number])