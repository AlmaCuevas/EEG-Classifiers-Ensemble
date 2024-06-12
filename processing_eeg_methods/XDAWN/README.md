EEG Signal Classification with Xdawn and Logistic Regression
This repository contains code for training and testing a machine learning model to classify EEG signals using the Xdawn algorithm and Logistic Regression. The goal is to provide a robust model for identifying user intentions from EEG data, which can be used in various biomedical applications, including brain-computer interfaces (BCIs).

Overview
The provided code performs the following tasks:

Data Loading: Loads EEG data and labels based on a specified dataset and subject ID.
Model Training: Utilizes Xdawn for signal enhancement and Logistic Regression for classification.
Cross-Validation: Implements Stratified K-Fold cross-validation to evaluate model performance.
Model Testing: Tests the trained classifier on a given epoch and provides classification probabilities.
Getting Started
Prerequisites
Ensure you have the following Python packages installed:

NumPy
scikit-learn
mne
pathlib
You will also need to have access to the EEG datasets and the data_loaders module, as well as the datasets_basic_infos configuration.

Code Structure
Imports and Configuration:
The code begins by importing necessary libraries and appending the system path to include the voting system platform directory.

Training Function:
xdawn_train function performs cross-validation using Xdawn and Logistic Regression, and outputs a classification report along with the accuracy.

Testing Function:
xdawn_test function takes a trained classifier and an epoch, and returns the classification probabilities for that epoch.

Main Execution:
The main block of the code sets manual inputs for the subject ID and dataset name, loads the data, trains the model, and tests it on a sample epoch.

Usage
Manual Inputs:
Modify the subject_id and dataset_name variables to specify which subject and dataset to use.

Folders and Paths:
Update the computer_root_path to the location of your datasets.

Run the Code:
Execute the script to train the model and test its performance. The script will output the training accuracy, classification report, and test probabilities.

python
Copiar código
if __name__ == '__main__':
    # Manual Inputs
    subject_id = 1  # Only two things I should be able to change
    dataset_name = 'aguilera_traditional'  # Only two things I should be able to change
    array_format = False

    # Folders and paths
    dataset_foldername = dataset_name + '_dataset'
    computer_root_path = "/Users/almacuevas/work_projects/voting_system_platform/Datasets/"  # Update to your path
    data_path = computer_root_path + dataset_foldername
    dataset_info = datasets_basic_infos[dataset_name]

    epochs, labels = load_data_labels_based_on_dataset(dataset_info, subject_id, data_path)
    target_names = dataset_info['target_names']

    print("******************************** Training ********************************")
    start = time.time()
    clf, acc = xdawn_train(epochs, labels, target_names)
    end = time.time()
    print("Training time: ", end - start)

    print("******************************** Test ********************************")
    epoch_number = 0
    start = time.time()
    array = xdawn_test(clf, epochs[epoch_number])
    end = time.time()
    print("One epoch, testing time: ", end - start)
    print(target_names)
    print("Probability: ", array)
    print("Real: ", labels[epoch_number])
Notes
Ensure the data paths and module imports are correctly set up according to your project structure.
The code uses hardcoded paths; update them as necessary to fit your environment.
The model is saved using pickle (commented out in the provided script); uncomment and update the path to save the trained model.
