# EEG Processing Methods

## Table of Contents

1. [share.py](https://github.com/AlmaCuevas/voting_system_platform/edit/main/processing_eeg_methods/README.md#sharepy)<br />
    1.1. [Purpose](https://github.com/AlmaCuevas/voting_system_platform/edit/main/processing_eeg_methods/README.md#purpose)<br />
    1.2. [Use](https://github.com/AlmaCuevas/voting_system_platform/edit/main/processing_eeg_methods/README.md#use)<br />
    1.3. [Dataset Information Definition](https://github.com/AlmaCuevas/voting_system_platform/edit/main/processing_eeg_methods/README.md#dataset-information-definition)<br />
    1.4. [Dataset Basic Infos](https://github.com/AlmaCuevas/voting_system_platform/edit/main/processing_eeg_methods/README.md#dataset-basic-infos)<br />
    1.5. [Description of Datasets](https://github.com/AlmaCuevas/voting_system_platform/edit/main/processing_eeg_methods/README.md#description-of-datasets)<br />
    1.6. [Documentation and Organization](https://github.com/AlmaCuevas/voting_system_platform/edit/main/processing_eeg_methods/README.md#documentation-and-organization)<br />
2. [data_loaders.py](https://github.com/AlmaCuevas/voting_system_platform/edit/main/processing_eeg_methods/README.md#data_loaderspy)<br />
    2.1. [Description](https://github.com/AlmaCuevas/voting_system_platform/edit/main/processing_eeg_methods/README.md#description)<br />
    2.2. [Functions](https://github.com/AlmaCuevas/voting_system_platform/edit/main/processing_eeg_methods/README.md#functions)<br />
    2.3. [Main Function](https://github.com/AlmaCuevas/voting_system_platform/edit/main/processing_eeg_methods/README.md#main-function)<br />
    2.4. [How to Use](https://github.com/AlmaCuevas/voting_system_platform/edit/main/processing_eeg_methods/README.md#how-to-use)<br />
   
## share.py

### Purpose:

The code provide detailed information about different EEG datasets, including metadata such as class names, number of channels, sampling rate, and more. It organizes this information into dictionaries for easy access and reference during data analysis or processing tasks.

### Use:
The code hints at potential future use cases, such as running experiments with all datasets at once.
It provides a structured framework for handling multiple datasets in a unified manner, facilitating data analysis and experimentation.


### Dataset Information Definition:

The code begins by defining dictionaries for each dataset:
* aguilera_traditional_info
* aguilera_gamified_info
* nieto_info, coretto_info
* torres_info
* ic_bci_2020

Each dictionary contains specific information about a particular EEG dataset, such as the number of classes, class names, number of EEG channels, sampling rate, channel names, number of subjects, and total number of trials.

#### Dataset Basic Infos:

After defining individual dataset information dictionaries, the code creates a master dictionary named datasets_basic_infos.
This master dictionary contains all the dataset information dictionaries, indexed by their names (aguilera_traditional, aguilera_gamified, nieto, coretto, torres).
This organization allows easy access to dataset information using the dataset names as keys.

#### Description of Datasets:
Each dataset is described with specific details:
* **Class information**: Number of classes and their respective names.
* **Channel information**: Number of EEG channels and their names.
* **Sampling information**: Number of samples, sample rate, and duration.
* **Subject and trial information**: Number of subjects and total number of trials.

#### Documentation and Organization:
Comments are provided throughout the code to document and explain the purpose of each variable and dictionary key.
The code is organized into sections, with each dataset's information clearly defined and separated.

## data_loaders.py

### Description:

This Python script is designed to load and preprocess EEG (electroencephalography) data from multiple datasets. It includes functions for loading data from different EEG datasets, each with its own preprocessing steps. The main purpose of the script is to facilitate the loading and preprocessing of EEG data for further analysis or machine learning tasks.

### Functions:
Dataset Loader Functions:

* **aguilera_dataset_loader**: Loads data from the Aguilera dataset, performs preprocessing steps such as renaming channels, filtering, and artifact removal using ICA.
* **nieto_dataset_loader**: Loads data from the Nieto dataset, performs preprocessing steps such as selecting a time window, transforming data for classification, etc.
* **torres_dataset_loader**: Loads data from the Torres dataset and preprocesses it.
* **coretto_dataset_loader**: Loads data from the Coretto dataset and preprocesses it.

### Main Function:

`load_data_labels_based_on_dataset`: The main function responsible for loading data based on the specified dataset name, subject ID, and data path. It calls the appropriate dataset loader function based on the dataset name provided.

### How to Use:
#### Set Parameters:

Specify the subject ID and dataset name in the __main__ block.
Define the dataset path based on the computer's root path and the dataset name.

#### Run the Script:

Execute the script to load and preprocess the EEG data.
The loaded data and labels will be printed to verify successful loading.

#### Inputs:
subject_id: Integer representing the ID of the subject from whom EEG data will be loaded.
dataset_name: String representing the name of the dataset from which EEG data will be loaded.

#### Outputs:
epochs: EEG data loaded as MNE Epochs objects.
data: Processed EEG data in numpy array format.
labels: Labels associated with the loaded EEG data.

#### Purpose:
The purpose of this script is to automate the process of loading and preprocessing EEG data from multiple datasets. By providing functions tailored to each dataset, it allows researchers and practitioners to easily access and prepare EEG data for analysis, modeling, or other applications in neuroscience research or clinical settings.

#### Importing Libraries

    import numpy as np
    import mne 
    from scipy.io import loadmat
    import os
    from share import datasets_basic_infos
    from Inner_Speech_Dataset.Python_Processing.Data_extractions import Extract_data_from_subject
    from Inner_Speech_Dataset.Python_Processing.Data_processing import Select_time_window, Transform_for_classificator
    from mne import io, Epochs, events_from_annotations, EpochsArray
    from mne.preprocessing import ICA, create_eog_epochs
    from autoreject import AutoReject
    from pathlib import Path

This block imports necessary libraries and modules required for the code.

#### Main Function: 
The `load_data_labels_based_on_dataset` function is the main function that loads data based on the specified dataset name, subject ID, and data path. It calls the appropriate dataset loader function based on the dataset name provided.

#### Defining Functions
* **aguilera_dataset_loader(data_path: str, gamified: bool)**: This function loads and preprocesses data for the Aguilera dataset based on whether it's gamified or not.

* **nieto_dataset_loader(root_dir: str, N_S: int)**: This function loads and preprocesses data for the Nieto dataset.

* **torres_dataset_loader(filepath: str, subject_id: int)**: This function loads and preprocesses data for the Torres dataset.

* **coretto_dataset_loader(filepath: str)**: This function loads and preprocesses data for the Coretto dataset.

* **load_data_labels_based_on_dataset(dataset_name: str, subject_id: int, data_path: str, transpose: bool = False)**: This function loads data and labels based on the specified dataset and subject ID.


#### Main Block
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

This block is the main part of the script. It sets manual inputs for `subject_id` and dataset_name, then constructs the data path based on these inputs. Finally, it calls `load_data_labels_based_on_dataset()` function to load data and labels based on the provided dataset and subject ID, prints out the shapes of data and labels, and provides a congratulatory message.

#### Load Data: 

In the `__main__` block, the script sets manual inputs for the subject ID and dataset name. Then it constructs the data path based on the dataset name and loads the data using the `load_data_labels_based_on_dataset` function.

#### Print Data Shape:
Finally, the script prints the shape of the loaded data and labels to verify that the data loading was successful.



