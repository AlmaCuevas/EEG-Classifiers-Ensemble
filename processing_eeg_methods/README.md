# EEG Processing Methods

## Table of Contents

1. [share.py](https://github.com/AlmaCuevas/voting_system_platform/tree/main/processing_eeg_methods#sharepy)<br />
    1.2. [Datasets](https://github.com/AlmaCuevas/voting_system_platform/tree/main/processing_eeg_methods#datasets)<br />
    1.3. [Content of Datasets](https://github.com/AlmaCuevas/voting_system_platform/tree/main/processing_eeg_methods#description-of-datasets)<br />
2. [data_loaders.py](https://github.com/AlmaCuevas/voting_system_platform/tree/main/processing_eeg_methods#data_loaderspy)<br />
    2.1. [Functions](https://github.com/AlmaCuevas/voting_system_platform/tree/main/processing_eeg_methods#functions)<br />
    2.2. [Main Function](https://github.com/AlmaCuevas/voting_system_platform/tree/main/processing_eeg_methods#main-function)<br />
    2.3. [How to Use](https://github.com/AlmaCuevas/voting_system_platform/tree/main/processing_eeg_methods#how-to-use)<br />
   
## share.py

The code implies the possibility of running experiments with all datasets simultaneously. It offers a well-organized framework for managing multiple datasets cohesively, simplifying data analysis and experimentation.

The code provides detailed information about different EEG datasets, including metadata such as class names, number of channels, sampling rate, and more. It organizes this information into dictionaries for easy access and reference during data analysis or processing tasks.

### Datasets:

The code begins by defining dictionaries for each dataset:
* aguilera_traditional_info
* aguilera_gamified_info
* nieto_info, coretto_info
* torres_info
* ic_bci_2020

After defining individual dataset information dictionaries, the code creates a master dictionary named `datasets_basic_infos`.
This master dictionary contains all the dataset information dictionaries, indexed by their names.

This organization allows easy access to dataset information using the dataset names as keys.

### Content of Datasets:
Each dataset is described with specific details:
* **Class information**: Number of classes and their respective names.
* **Channel information**: Number of EEG channels and their names.
* **Sampling information**: Number of samples, sample rate, and duration.
* **Subject and trial information**: Number of subjects and total number of trials.

## data_loaders.py

This Python script is designed to load and preprocess EEG (electroencephalography) data from multiple datasets. It includes functions for loading data from different EEG datasets, each with its preprocessing steps. The script's main purpose is to facilitate the loading and preprocessing of EEG data for further analysis or machine learning tasks.

### Functions:

* **aguilera_dataset_loader**: This function loads data from the Aguilera dataset and performs preprocessing steps such as renaming channels, filtering, and artifact removal using ICA.
* **nieto_dataset_loader**: Loads data from the Nieto dataset and performs preprocessing steps such as selecting a time window, transforming data for classification, etc.
* **torres_dataset_loader**: Loads data from the Torres dataset and preprocesses it.
* **coretto_dataset_loader**: Loads data from the Coretto dataset and preprocesses it.

### Main Function:

`load_data_labels_based_on_dataset`: The main function responsible for loading data based on the specified dataset name, subject ID, and data path. It calls the appropriate dataset loader function based on the dataset name provided.

### How to Use:

`epochs, data, labels = load_data_labels_based_on_dataset(dataset_name, subject_id, data_path)`

* Specify the dataset name and subject ID.
* Define the dataset path based on the computer's root path and the dataset name.

The datasets should be loaded in a folder. It should look like this:<br />
<img width="267" alt="image" src="https://github.com/AlmaCuevas/voting_system_platform/assets/46833474/86715cdd-ee61-4137-96d2-348519b46c0d">


#### Run the Script:

Execute the script to load and preprocess the EEG data.
The loaded data and labels will be printed to verify successful loading.

#### Inputs:
subject_id: Integer representing the subject's ID from which EEG data will be loaded.
dataset_name: String representing the name of the dataset from which EEG data will be loaded.

#### Outputs:
epochs: EEG data loaded as MNE Epochs objects.
data: Processed EEG data in numpy array format.
labels: Labels associated with the loaded EEG data.

#### Main Function: 
The `load_data_labels_based_on_dataset` function is the main function that loads data based on the specified dataset name, subject ID, and data path. It calls the appropriate dataset loader function based on the dataset name provided.

#### Defining Functions
* **aguilera_dataset_loader(data_path: str, gamified: bool)**: This function loads and preprocesses data for the Aguilera dataset based on whether it's gamified or not.

* **nieto_dataset_loader(root_dir: str, N_S: int)**: This function loads and preprocesses data for the Nieto dataset.

* **torres_dataset_loader(filepath: str, subject_id: int)**: This function loads and preprocesses data for the Torres dataset.

* **coretto_dataset_loader(filepath: str)**: This function loads and preprocesses data for the Coretto dataset.

* **load_data_labels_based_on_dataset(dataset_name: str, subject_id: int, data_path: str, transpose: bool = False)**: This function loads data and labels based on the specified dataset and subject ID.


#### Main Block
In the `__main__` block, the script sets manual inputs for the subject ID and dataset name. Then, it constructs the data path based on the dataset name and loads the data using the `load_data_labels_based_on_dataset` function.
    if __name__ == '__main__':
    # Manual Inputs
    subject_id = 2  # Only two things I should be able to change
    dataset_name = 'aguilera_traditional'  # Only two things I should be able to change

    # Folders and paths
    dataset_foldername = dataset_name + '_dataset'
    computer_root_path = str(ROOT_VOTING_SYSTEM_PATH) + "/Datasets/"
    data_path = computer_root_path + dataset_foldername

    epochs, data, labels = load_data_labels_based_on_dataset(dataset_name, subject_id, data_path)

    print(data.shape)
    print(labels.shape)
    print("Congrats! You were able to load data. You can now use this in a processing method.")

This block is the main part of the script. It sets manual inputs for `subject_id` and `dataset_name`, then constructs the data path based on these inputs. Finally, it calls `load_data_labels_based_on_dataset()` function to load data and labels based on the provided dataset and subject ID, prints out the shapes of data and labels, and provides a congratulatory message.
