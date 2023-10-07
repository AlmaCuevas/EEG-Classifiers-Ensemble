import numpy as np
import mne
from scipy.io import loadmat
import os

from sklearn.model_selection import train_test_split


# TODO: Create a DataLoader for each dataset
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


def extract_segment_trial(raw, baseline=(-0.5, 0), duration=4): # I think this could have been done with Andrea's approach, but it works so I won't move it yet.
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

def train_test_val_split(dataX, dataY, valid_flag: bool = False):
    train_ratio = 0.75
    test_ratio = 0.10

    # train is now 75% of the entire data set
    x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=1 - train_ratio)

    if valid_flag:
        validation_ratio = 0.15
        # test is now 10% of the initial data set
        # validation is now 15% of the initial data set
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                        test_size=test_ratio / (test_ratio + validation_ratio))
    else:
        x_val = None
        y_val = None
    return x_train, x_test, x_val, y_train, y_test, y_val

if __name__ == '__main__':
    subject_id = 1
    data_path = "/Users/almacuevas/work_projects/voting_system_platform/Datasets/aguilera_dataset"

    train_filename = F"S{subject_id}.edf"
    train_filepath = os.path.join(data_path, train_filename)

    train_loader = aguilera_dataset_loader(train_filepath)
    train_cnt = train_loader.load()

    print(train_cnt.get_data().shape)
