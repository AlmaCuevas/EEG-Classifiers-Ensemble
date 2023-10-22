import numpy as np
import mne
from scipy.io import loadmat
import os
from Code.data_preprocess import mne_apply, bandpass_cnt
from share import datasets_basic_infos
from Code.Inner_Speech_Dataset.Python_Processing.Data_extractions import Extract_data_from_subject
from Code.Inner_Speech_Dataset.Python_Processing.Data_processing import Select_time_window, Transform_for_classificator, Split_trial_in_time
from mne import io, Epochs, events_from_annotations, EpochsArray


# TODO: Create a DataLoader for Torres

def aguilera_dataset_loader(data_path: str):
    raw = io.read_raw_edf(data_path, preload=True, verbose=40, exclude=['Channel 21', 'Channel 22', 'Gyro 1', 'Gyro 2', 'Gyro 3'])
    events, event_id = events_from_annotations(raw)

    event_id.pop('OVTK_StimulationId_Label_05') # This is not a command
    events = events[3:] # From the one that is not a command

    # Read epochs
    epochs = Epochs(raw, events, event_id, preload=True, tmin=0, tmax=1.4, baseline=None)
    label = epochs.events[:, -1]
    return epochs, label

def nieto_dataset_loader(root_dir: str, N_S: int):
    ### Hyperparameters
    # N_S: Subject number

    # Data Type
    datatype = "EEG"

    # Sampling rate
    fs = 256

    # Select the useful par of each trial. Time in seconds
    t_start = 1.5
    t_end = 3.5

    # Load all trials for a single subject
    X, Y = Extract_data_from_subject(root_dir, N_S, datatype) # This uses the derivatives folder

    # Cut useful time. i.e action interval
    X = Select_time_window(X=X, t_start=t_start, t_end=t_end, fs=fs)

    # Conditions to compared
    Conditions = [["Inner"], ["Inner"],["Inner"], ["Inner"]]
    # The class for the above condition
    Classes = [["Up"], ["Down"], ["Right"], ["Left"]]

    # Transform data and keep only the trials of interest
    X, Y = Transform_for_classificator(X, Y, Classes, Conditions)
    Y = Y.astype(int)
    event_dict = {'Arriba': 0, 'Abajo': 1, 'Derecha': 2,'Izquierda': 3}
    return X, Y, event_dict

def coretto_dataset_loader(filepath: str):
    """
    Load data from all .mat files, combine them, eliminate EOG signals, shuffle and seperate
    training data, validation data and testing data. Also do mean subtraction on x.

    F3 -- Muestra 1:4096
    F4 -- Muestra 4097:8192
    C3 -- Muestra 8193:12288
    C4 -- Muestra 12289:16384
    P3 -- Muestra 16385:20480
    P4 -- Muestra 20481:24576
    Etiquetas :  Modalidad: 1 - Imaginada
	     		            2 - Pronunciada

                 Estímulo:  1 - A
                            2 - E
                            3 - I
                            4 - O
                            5 - U
                            6 - Arriba
                            7 - Abajo
                            8 - Adelante
                            9 - Atrás
                            10 - Derecha
                            11 - Izquierda
                 Artefactos: 1 - No presenta
                             2 - Presencia de parpadeo(blink)
    """

    x = []
    y = []

    #for i in np.arange(1, 10): # import all data in 9 .mat files # TODO: We are still not loading everyone. Just one subject at a time.
    EEG = loadmat(filepath) # Channels and labels are concat

    #modality = EEG['EEG'][:,24576]
    #stimulus = EEG['EEG'][:, 24577]

    direction_labels = [6, 7, 10, 11]
    EEG_filtered_by_labels = EEG['EEG'][(EEG['EEG'][:,24576] == 1) & (np.in1d(EEG['EEG'][:,24577], direction_labels))]
    x_channels_concat = EEG_filtered_by_labels[:,:-3] # Remove labels
    x_divided_in_channels = np.asarray(np.split(x_channels_concat,6,axis=1))
    # There are 3 words trials, but the samples didn't match so a series of conversions had to be done
    x_divided_in_channels_and_thirds = np.asarray(np.split(x_divided_in_channels[:,:,1:],3,axis=2))
    x_transposed = np.transpose(x_divided_in_channels_and_thirds, (1, 3, 0, 2))
    x_transposed_reshaped = x_transposed.reshape(x_transposed.shape[:-2] + (-1,))

    y = EEG_filtered_by_labels[:,-2] # Direction labels array

    # reshape x in 3d data(Trials, Channels, Samples) and y in 1d data(Trials)
    x = np.transpose(x_transposed_reshaped, (2, 0, 1))
    y = np.asarray(y, dtype=np.int32)
    y = np.repeat(y, 3, axis=0)
    # N, C, H = x.shape # You can use something like this for unit test later.
    event_dict = {"Arriba": 6, "Abajo": 7, "Derecha": 10, "Izquierda": 11}
    return x, y, event_dict

def load_data_labels_based_on_dataset(dataset_name: str, subject_id: int, data_path: str, transpose: bool = False, array_format: bool = True):
    dataset_info = datasets_basic_infos[dataset_name]
    if dataset_name == 'aguilera':
        filename = F"S{subject_id}.edf"
        filepath = os.path.join(data_path, filename)
        data, label = aguilera_dataset_loader(filepath) # The output is epochs
        if array_format:
            data = data.get_data()
    elif dataset_name == 'nieto': # Checar sample
        data, label, event_dict = nieto_dataset_loader(data_path, subject_id)
    elif dataset_name == 'coretto':
        foldername = "S{:02d}".format(subject_id)
        filename = foldername + "_EEG.mat"
        path = [data_path, foldername, filename]
        filepath = os.path.join(*path)
        data, label, event_dict = coretto_dataset_loader(filepath)
    elif dataset_name == 'torres':
        raise Exception("Torres is not ready yet.")
    else:
        raise Exception("Not supported dataset, choose from the following: aguilera, nieto, coretto or torres.")
    if transpose:
        data = np.transpose(data, (0, 2, 1))
    if not array_format and dataset_name != 'aguilera':
        #raise Exception("For now epoch is only available for Aguilera")
        events = np.column_stack((
            np.arange(0, dataset_info['sample_rate'] * dataset_info['total_trials'], dataset_info['sample_rate']),
            np.zeros(len(label), dtype=int),
            np.array(label),
        ))
        data = EpochsArray(data, info=mne.create_info(dataset_info['#_channels'],
                                                        sfreq=dataset_info['sample_rate'], ch_types='eeg'), events=events,
                             event_id=event_dict)
    return data, label

if __name__ == '__main__':
    # Manual Inputs
    subject_id = 1  # Only two things I should be able to change
    dataset_name = 'coretto'  # Only two things I should be able to change
    array_format = False

    # Folders and paths
    dataset_foldername = dataset_name + '_dataset'
    data_path = "/Users/almacuevas/work_projects/voting_system_platform/Datasets/" + dataset_foldername

    data, label = load_data_labels_based_on_dataset(dataset_name, subject_id, data_path, array_format=array_format)

    if array_format:
        print(data.shape)
        print(label.shape)
    else:
        print(data)
    print("Congrats! You were able to load data. You can now use this in a processing method.")



