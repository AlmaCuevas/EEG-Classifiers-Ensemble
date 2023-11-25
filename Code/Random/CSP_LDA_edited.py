# At the end i didn't really use this. It's fishy that they train with a section of the data and they give back the data
# for testing. I did took the CSP and LDA idea to create the _probs version.
# BTW this uses array format, not epochs
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP

from data_loaders import load_data_labels_based_on_dataset
from share import datasets_basic_infos

print(__doc__)

# Set parameters
tmin, tmax = -1.0, 4.0
# event_id = dict(hands=2, feet=3)
event_id = dict(left_hand=2, right_hand=3)  # T1 corresponds to left hand and T2 corresponds to right hand
# runs = [6, 10, 14]  # motor imagery: hands vs feet
# runs = [5, 9, 13]  # actual movement: both fists or both feet
runs = [4, 8, 12]  # motor imagery: left vs right hand
# runs = [3, 7, 11] # actual movement: left vs right hand
# Define lists to store performance metrics across subjects
mean_performance_over_time_selected = []
selected_subjects = []
classification_accuracies_selected = []  # This will store classification accuracies
classification_accuracies_all = []  # Stores accuracies for all subjects

# Loop over subjects
for subject_id in range(1,17):
    #subject_id = 3  # Only two things I should be able to change
    dataset_name = 'aguilera_traditional'  # Only two things I should be able to change
    array_format = False

    # Folders and paths
    dataset_foldername = dataset_name + '_dataset'
    # computer_root_path = "/Users/rosit/Documents/MCC/voting_system_platform/Datasets/"  # OMEN
    computer_root_path = "/Users/almacuevas/work_projects/voting_system_platform/Datasets/"  # MAC
    data_path = computer_root_path + dataset_foldername
    dataset_info = datasets_basic_infos[dataset_name]

    epochs, labels = load_data_labels_based_on_dataset(dataset_name, subject_id, data_path, array_format=array_format)
    target_names = dataset_info['target_names']
    epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0)
    # Classification with linear discriminant analysis

    # Define a monte-carlo cross-validation generator (reduce variance)
    scores = []
    epochs_data = epochs.get_data()
    epochs_data_train = epochs_train.get_data()
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_data_train)

    # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

    # Use scikit-learn Pipeline with cross_val_score function
    clf = Pipeline([("CSP", csp), ("LDA", lda)])
    scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=None)

    # store classification accuracies for all subjects
    classification_accuracies_all.append(np.mean(scores))

    # Printing the results
    class_balance = np.mean(labels == labels[0])
    with open('/Users/almacuevas/work_projects/voting_system_platform/Code/Random/classification_info.txt', 'a') as f:
        f.write(f"Classification accuracy for Subject {subject_id}: {np.mean(scores)} / Chance level: {class_balance}\n")

    # Look at performance over time

    sfreq = dataset_info['sample_rate']
    w_length = int(sfreq * 0.5)  # running classifier: window length
    w_step = int(sfreq * 0.1)  # running classifier: window step size
    w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)

    scores_windows = []

    for train_idx, test_idx in cv_split:
        y_train, y_test = labels[train_idx], labels[test_idx]

        X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
        X_test = csp.transform(epochs_data_train[test_idx])

        # fit classifier
        lda.fit(X_train, y_train)

        # running classifier: test classifier on sliding window
        score_this_window = []
        for n in w_start:
            X_test = csp.transform(epochs_data[test_idx][:, :, n: (n + w_length)])
            score_this_window.append(lda.score(X_test, y_test))
        scores_windows.append(score_this_window)

    # Classification accuracy for the current subject
    classification_accuracy = np.mean(scores)
    mean_performance = np.mean(scores_windows, axis=0)

    # plot CSP patterns estimated on full data for visualization
    csp.fit_transform(epochs_data, labels)

    # Save to different folders depending on whether subject is selected or not
    # if np.mean(mean_performance[1:]) > 0.70:
    #    fig = csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
    #    fig.savefig(f'CSP_Patterns_Selected_Imaginary/subject_id_{subject_id}_CSP_patterns.png')
    #    plt.close(fig)
    # else:
    #    fig = csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
    #    fig.savefig(f'CSP_Patterns_Unselected_Imaginary/subject_id_{subject_id}_CSP_patterns.png')
    #    plt.close(fig)

    # Check if subject meets the mean performance criteria
    if np.mean(mean_performance[1:]) > 0.25:
        # Append metrics to the lists
        classification_accuracies_selected.append(classification_accuracy)
        mean_performance_over_time_selected.append(mean_performance)
        selected_subjects.append(subject_id)

# Plot the mean performance over time for selected subjects
if selected_subjects:
    # Check the minimum length of elements in mean_performance_over_time_selected
    min_len = min([len(arr) for arr in mean_performance_over_time_selected])

    # Trim each element in mean_performance_over_time_selected to the minimum length
    mean_performance_over_time_selected = [arr[:min_len] for arr in mean_performance_over_time_selected]

    # Calculate statistics and plot for selected subjects only
    overall_mean_performance = np.mean(mean_performance_over_time_selected, axis=0)
    average_classification_accuracy = np.mean(classification_accuracies_selected)

    # Compute w_times using the adjusted w_start array
    w_times = (w_start[:min_len] + w_length / 2.0) / sfreq + epochs.tmin

    plt.figure()
    plt.plot(w_times, overall_mean_performance, label="Mean Performance")
    plt.axvline(0, linestyle="--", color="k", label="Onset")
    plt.axhline(0.5, linestyle="-", color="k", label="Chance")
    plt.xlabel("Time (s)")
    plt.ylabel("Classification Accuracy")
    plt.title("Mean Classification Performance over Time for Selected Subjects")
    plt.legend(loc="lower right")
    plt.show()

    # Print selected subjects and average classification accuracy
    print("Selected Subjects:", selected_subjects)
    print("Average Classification Accuracy:", average_classification_accuracy)
else:
    print("No subjects met the mean performance criteria.")

with open('classification_info.txt', 'a') as f:
    f.write(f"Selected Subjects: {selected_subjects}\n")
    f.write(f"Average Classification Accuracy for Selected Subjects: {average_classification_accuracy}\n")
    f.write(f"Average Classification Accuracy for All Subjects: {np.mean(classification_accuracies_all)}\n")
    f.write(f"Average Classification Accuracy for All Subjects: {np.mean(classification_accuracies_all)}\n")