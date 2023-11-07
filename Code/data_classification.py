from Code.data_loaders import load_data_labels_based_on_dataset
from share import datasets_basic_infos

def group_methods() -> list[float]:

    # Results are always arrays of probability like [0.2, 0.3, 0.1, 0.4], but they are not normalized.
    # TODO: We have to find a way to decide who has the highest weight. Maybe based on the report.
    probs = [0.2, 0.3, 0.1, 0.4]
    return probs


if __name__ == '__main__':
    # Manual Inputs
    subject_id = 1  # Only two things I should be able to change
    dataset_name = 'aguilera'  # Only two things I should be able to change
    array_format = True

    # Folders and paths
    dataset_foldername = dataset_name + '_dataset'
    #computer_root_path = "/Users/rosit/Documents/MCC/voting_system_platform/Datasets/"  # OMEN
    computer_root_path = "/Users/almacuevas/work_projects/voting_system_platform/Datasets/" # MAC
    data_path = computer_root_path + dataset_foldername

    data, label = load_data_labels_based_on_dataset(dataset_name, subject_id, data_path, array_format=array_format)

    print("Congrats! The first draft of the voting system is now working.")
