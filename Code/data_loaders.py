import numpy as np
import mne
from scipy.io import loadmat
import os
from voting_system_platform.Code.Inner_Speech_Dataset.Python_Processing.Data_extractions import Extract_data_from_subject
from voting_system_platform.Code.Inner_Speech_Dataset.Python_Processing.Data_processing import Select_time_window, Transform_for_classificator, Split_trial_in_time

# TODO: Create a DataLoader for Nieto and Torres
class aguilera_dataset_loader:
    # Aguilera dataset are .edf
    def __init__(self, filename):
        self.filename = filename

    def load(self):
        cnt = self.extract_data()
        events, artifact_trial_mask = self.extract_events(cnt)
        cnt.info["temp"]["events"] = events
        cnt.info["temp"]["artifact_trial_mask"] = artifact_trial_mask
        return cnt

    def extract_data(self):
        raw_edf = mne.io.read_raw_edf(self.filename, stim_channel="auto", verbose='ERROR',
                                      exclude=(['Gyro 1',
                           'Gyro 2',
                           'Gyro 3',
                           'Channel 20',
                           'Channel 21']))
        #raw_edf.rename_channels(
        #    {'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz', 'EEG-3': 'FC2', 'EEG-4': 'FC4',
        #     'EEG-5': 'C5', 'EEG-C3': 'C3', 'EEG-6': 'C1', 'EEG-Cz': 'Cz', 'EEG-7': 'C2', 'EEG-C4': 'C4', 'EEG-8': 'C6',
        #     'EEG-9': 'CP3', 'EEG-10': 'CP1', 'EEG-11': 'CPz', 'EEG-12': 'CP2', 'EEG-13': 'CP4',
        #     'EEG-14': 'P1', 'EEG-15': 'Pz', 'EEG-16': 'P2', 'EEG-Pz': 'POz'})
        raw_edf.load_data()
        # correct nan values
        data = raw_edf.get_data()

        for i_chan in range(data.shape[0]):
            # first set to nan, than replace nans by nanmean.
            this_chan = data[i_chan]
            data[i_chan] = np.where(
                this_chan == np.min(this_chan), np.nan, this_chan
            )
            mask = np.isnan(data[i_chan])
            chan_mean = np.nanmean(data[i_chan])
            data[i_chan, mask] = chan_mean

        edf_events = mne.events_from_annotations(raw_edf)
        raw_edf = mne.io.RawArray(data, raw_edf.info, verbose="ERROR")
        # remember gdf events
        raw_edf.info["temp"] = dict()
        raw_edf.info["temp"]["events"] = edf_events
        return raw_edf

    def extract_events(self, raw_edf):
        events, name_to_code = raw_edf.info["temp"]["events"]

        class_names = ['OVTK_StimulationId_Label_01', 'OVTK_StimulationId_Label_02', 'OVTK_StimulationId_Label_03', 'OVTK_StimulationId_Label_04', 'OVTK_StimulationId_Label_05']

        trial_codes = [value_tag for class_name, value_tag in name_to_code.items() if any(class_name in x for x in class_names)]


        trial_mask = [ev_code in trial_codes for ev_code in events[:, 2]]
        trial_events = events[trial_mask]

        # TODO: We don't have rejected trials yet
        #trial_start_events_tag = [value_tag for class_name, value_tag in name_to_code.items() if class_name == 'OVTK_GDF_Start_Of_Trial']

        #trial_start_events = events[events[:, 2] == trial_start_events_tag[0]]

        #assert len(trial_start_events) == len(trial_events)
        artifact_trial_mask = np.zeros(len(trial_events), dtype=np.uint8)
        #artifact_events = events[events[:, 2] == 1]

        #for artifact_time in artifact_events[:, 0]:
        #    try: # I think this is poorly done. It should be "delete the trial that contains the boundary, that is the
        #        # the one between the before and after the timestamp of the boundary"
        #        i_trial = trial_start_events[:, 0].tolist().index(artifact_time)
        #        artifact_trial_mask[i_trial] = 1
        #    except:
        #        pass

        return trial_events, artifact_trial_mask


def extract_segment_trial(raw, baseline=(0, 0), duration=1.4): # I think this could have been done with Andrea's approach, but it works so I won't move it yet.
    '''
    get segmented data and corresponding labels from raw_gdb.
    :param raw: raw data
    :param baseline: unit: second. baseline for the segment data. The first value is time before cue.
                     The second value is the time after the Mi duration. Positive values represent the time delays,
                     negative values represent the time lead.
    :param duration: unit: seconds. mi duration time
    :return: array data: trial data, labels
    '''
    events = raw.info["temp"]["events"]
    raw_data = raw.get_data()
    freqs = raw.info['sfreq']
    epoch_duration = int(freqs * duration)
    duration_before_epoch = int(freqs * baseline[0])
    duration_after_epoch = int(freqs * baseline[1])

    labels = np.array(events[:, 2])

    trial_data = []
    for i_event in events:  # i_event [time, 0, class]
        segmented_data = raw_data[:,
                         int(i_event[0]) + duration_before_epoch:int(i_event[0]) + epoch_duration + duration_after_epoch]
        assert segmented_data.shape[-1] == epoch_duration - duration_before_epoch + duration_after_epoch
        trial_data.append(segmented_data)
    trial_data = np.stack(trial_data, 0)

    return trial_data, labels


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

    # Transform data and keep only the trials of interes
    X, Y = Transform_for_classificator(X, Y, Classes, Conditions)

    return X, Y

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
    x = [np.split(x_channel,6) for x_channel in x_channels_concat]
    y = EEG_filtered_by_labels[:,-2] # Direction labels array

    # reshape x in 3d data(Trials, Channels, Samples) and y in 1d data(Trials)
    x = np.asarray(x)
    y = np.asarray(y, dtype=np.int32)

    # N, C, H = x.shape # You can use something like this for unit test later.
    return x, y

if __name__ == '__main__':
    # Manual Inputs
    subject_id = 1  # Only two things I should be able to change
    dataset_name = 'nieto'  # Only two things I should be able to change

    # Folders and paths
    dataset_foldername = dataset_name + '_dataset'
    data_path = "/Users/almacuevas/work_projects/voting_system_platform/Datasets/" + dataset_foldername



    if dataset_name == 'aguilera':
        filename = F"S{subject_id}.edf"
        filepath = os.path.join(data_path, filename)
        loader = aguilera_dataset_loader(filepath)
        cnt = loader.load()
        print(cnt.get_data().shape)
    elif dataset_name == 'nieto':
        X, Y = nieto_dataset_loader(data_path, subject_id)
        print(X.shape)
        print(Y.shape)



