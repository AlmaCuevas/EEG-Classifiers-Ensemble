import numpy as np
import mne
from scipy.io import loadmat
import os


class BCICompetition4Set2A:

    def __init__(self, filename, labels_filename=None):
        self.filename = filename
        self.labels_filename = labels_filename

    def load(self):
        cnt = self.extract_data()
        events, artifact_trial_mask = self.extract_events(cnt)
        cnt.info["temp"]["events"] = events
        cnt.info["temp"]["artifact_trial_mask"] = artifact_trial_mask
        return cnt

    def extract_data(self):
        raw_gdf = mne.io.read_raw_edf(self.filename, stim_channel="auto", verbose='ERROR',
                                      exclude=(["EOG-left", "EOG-central", "EOG-right"]))
        raw_gdf.rename_channels(
            {'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz', 'EEG-3': 'FC2', 'EEG-4': 'FC4',
             'EEG-5': 'C5', 'EEG-C3': 'C3', 'EEG-6': 'C1', 'EEG-Cz': 'Cz', 'EEG-7': 'C2', 'EEG-C4': 'C4', 'EEG-8': 'C6',
             'EEG-9': 'CP3', 'EEG-10': 'CP1', 'EEG-11': 'CPz', 'EEG-12': 'CP2', 'EEG-13': 'CP4',
             'EEG-14': 'P1', 'EEG-15': 'Pz', 'EEG-16': 'P2', 'EEG-Pz': 'POz'})
        raw_gdf.load_data()
        # correct nan values
        data = raw_gdf.get_data()

        for i_chan in range(data.shape[0]):
            # first set to nan, than replace nans by nanmean.
            this_chan = data[i_chan]
            data[i_chan] = np.where(
                this_chan == np.min(this_chan), np.nan, this_chan
            )
            mask = np.isnan(data[i_chan])
            chan_mean = np.nanmean(data[i_chan])
            data[i_chan, mask] = chan_mean

        gdf_events = mne.events_from_annotations(raw_gdf)
        raw_gdf = mne.io.RawArray(data, raw_gdf.info, verbose="ERROR")
        # remember gdf events
        raw_gdf.info["temp"] = dict()
        raw_gdf.info["temp"]["events"] = gdf_events
        return raw_gdf

    def extract_events(self, raw_gdf):
        # all events
        events, name_to_code = raw_gdf.info["temp"]["events"]
        if "T" in self.filename:
            train_set = True
        else:
            train_set = False
            # Commented because it didn't work, plus I'm not even gonna really use the BCI competitions dataset
            # assert (
            #    # "cue unknown/undefined (used for BCI competition) "
            #    "783" in name_to_code
            # )
        class_names = ['OVTK_GDF_Foot', 'OVTK_GDF_Left', 'OVTK_GDF_Right', 'OVTK_GDF_Tongue']
        if train_set:
            trial_codes = [value_tag for class_name, value_tag in name_to_code.items() if any(class_name in x for x in class_names)]
        else:
            trial_codes = [7]  # "unknown" class

        trial_mask = [ev_code in trial_codes for ev_code in events[:, 2]]
        trial_events = events[trial_mask]

        # assert len(trial_events) == 288, "Got {:d} markers".format(
        #    len(trial_events)
        # )
        # print('self.filename[-8:-5]: ', self.filename[-8:-5])

        # Maybe this section could be done with a map(). Is to change the whatever class number to 1, 2, 3 and 4.
        unique_trial_events = set(trial_events[:, 2])
        count = 0
        for unique_event in unique_trial_events:
            count += 1
            for row_event, trial_class in enumerate(trial_events[:, 2]):
                if trial_class == unique_event:
                    trial_events[row_event, 2] = count


        # possibly overwrite with markers from labels file
        # if self.labels_filename is not None:
        #    classes = loadmat(self.labels_filename)["classlabel"].squeeze()
        #    if train_set:  # 确保另外给的train_label和train data中的label一样
        #        np.testing.assert_array_equal(trial_events[:, 2], classes)
        #    trial_events[:, 2] = classes
        # unique_classes = np.unique(trial_events[:, 2])
        # assert np.array_equal(
        #    [1, 2, 3, 4], unique_classes
        # ), "Expect 1,2,3,4 as class labels, got {:s}".format(
        #    str(unique_classes)
        # )

        # now also create 0-1 vector for rejected trials
        trial_start_events_tag = [value_tag for class_name, value_tag in name_to_code.items() if class_name == 'OVTK_GDF_Start_Of_Trial']

        trial_start_events = events[events[:, 2] == trial_start_events_tag[0]]

        assert len(trial_start_events) == len(trial_events)
        artifact_trial_mask = np.zeros(len(trial_events), dtype=np.uint8)
        artifact_events = events[events[:, 2] == 1]

        for artifact_time in artifact_events[:, 0]:
            try: # I think this is poorly done. It should be "delete the trial that contains the boundary, that is the
                # the one between the before and after the timestamp of the boundary"
                i_trial = trial_start_events[:, 0].tolist().index(artifact_time)
                artifact_trial_mask[i_trial] = 1
            except:
                pass

        return trial_events, artifact_trial_mask


def extract_segment_trial(raw_gdb, baseline=(-0.5, 0), duration=4):
    '''
    get segmented data and corresponding labels from raw_gdb.
    :param raw_gdb: raw data
    :param baseline: unit: second. baseline for the segment data. The first value is time before cue.
                     The second value is the time after the Mi duration. Positive values represent the time delays,
                     negative values represent the time lead.
    :param duration: unit: seconds. mi duration time
    :return: array data: trial data, labels
    '''
    events = raw_gdb.info["temp"]["events"]
    raw_data = raw_gdb.get_data()
    freqs = raw_gdb.info['sfreq']
    mi_duration = int(freqs * duration)
    duration_before_mi = int(freqs * baseline[0])
    duration_after_mi = int(freqs * baseline[1])

    labels = np.array(events[:, 2])

    trial_data = []
    for i_event in events:  # i_event [time, 0, class]
        segmented_data = raw_data[:,
                         int(i_event[0]) + duration_before_mi:int(i_event[0]) + mi_duration + duration_after_mi]
        assert segmented_data.shape[-1] == mi_duration - duration_before_mi + duration_after_mi
        trial_data.append(segmented_data)
    trial_data = np.stack(trial_data, 0)

    return trial_data, labels


if __name__ == '__main__':
    subject_id = 3
    data_path = "/Users/rosit/Documents/MCC/LMDA-Code/BCICIV_2a_edf/"

    train_filename = "A{:02d}T.edf".format(subject_id)
    test_filename = "A{:02d}E.edf".format(subject_id)
    train_filepath = os.path.join(data_path, train_filename)
    test_filepath = os.path.join(data_path, test_filename)
    train_label_filepath = train_filepath.replace(".edf", ".mat")
    test_label_filepath = test_filepath.replace(".edf", ".mat")

    train_loader = BCICompetition4Set2A(
        train_filepath, labels_filename=train_label_filepath
    )
    test_loader = BCICompetition4Set2A(
        test_filepath, labels_filename=test_label_filepath
    )
    train_cnt = train_loader.load()
    test_cnt = test_loader.load()

    train_cnt = train_cnt.drop_channels(
        ["EOG-left", "EOG-central", "EOG-right"]
    )
    test_cnt = test_cnt.drop_channels(
        ["EOG-left", "EOG-central", "EOG-right"]
    )
    print(train_cnt.get_data().shape)
    print(test_cnt.get_data().shape)
