import numpy as np
from data_dataclass import ProcessingMethods, complete_experiment, probability_input
from data_loaders import load_data_labels_based_on_dataset
from data_utils import (
    convert_into_independent_channels,
    get_dataset_basic_info,
    get_input_data_path,
    standard_saving_path,
)
from share import datasets_basic_infos
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    # Manual Inputs
    dataset_name = "braincommand"
    selected_classes = [0, 1, 2, 3]

    ce = complete_experiment()

    pm = ProcessingMethods()

    dataset_info = get_dataset_basic_info(datasets_basic_infos, dataset_name)
    dataset_info["#_class"] = len(selected_classes)

    pm.activate_methods(
        selected_transformers=True,  # Training is over-fitted. Training accuracy >90
        customized=True,  # Simpler than selected_transformers, only one transformer and no frequency bands. No need to activate both at the same time
        ShallowFBCSPNet=False,
        LSTM=False,  # Training is over-fitted. Training accuracy >90
        GRU=False,  # Training is over-fitted. Training accuracy >90
        diffE=False,  # It doesn't work if you only use one channel in the data
        feature_extraction=False,
        number_of_classes=dataset_info["#_class"],
    )
    activated_methods: list[str] = pm.get_activated_methods()

    version_name = "test_delete_later"  # To keep track what the output processing alteration went through
    processing_name: str = "test_delete_later"

    data_path = get_input_data_path(dataset_name)

    save_original_channels = dataset_info["#_channels"]
    save_original_trials = dataset_info["total_trials"]

    for subject_id in range(29, 30):
        print(subject_id)

        dataset_info["#_channels"] = save_original_channels
        dataset_info["total_trials"] = save_original_trials

        epochs, data, labels = load_data_labels_based_on_dataset(
            dataset_info,
            subject_id,
            data_path,
            selected_classes=selected_classes,
            threshold_for_bug=0.00000001,
        )  # could be any value, ex numpy.min

        # Only if using independent channels:
        dataset_info["total_trials"] = save_original_trials * save_original_channels
        dataset_info["#_channels"] = 1

        cv = StratifiedKFold(
            n_splits=10, shuffle=True, random_state=42
        )  # Do cross-validation

        count_Kfolds: int = 0
        for train, test in cv.split(epochs, labels):
            print(
                "******************************** Training ********************************"
            )
            count_Kfolds += 1
            # Convert independent channels to pseudo-trials
            data_train, labels_train = convert_into_independent_channels(
                data[train], labels[train]
            )
            data_train = np.transpose(np.array([data_train]), (1, 0, 2))

            pm.train(
                subject_id=subject_id,
                data=data_train,
                labels=labels_train,
                dataset_info=dataset_info,
            )

            print(
                "******************************** Test ********************************"
            )

            for epoch_number in test:
                # Convert independent channels to pseudo-trials
                data_test, labels_test = convert_into_independent_channels(
                    np.asarray([data[epoch_number]]), labels[epoch_number]
                )
                data_test = np.transpose(np.array([data_test]), (1, 0, 2))

                for pseudo_trial in range(len(data_test)):
                    pm.test(
                        subject_id=subject_id,
                        data=np.asarray([data_test[pseudo_trial]]),
                        dataset_info=dataset_info,
                    )

                    for method_name in vars(pm):
                        method = getattr(pm, method_name)
                        ce.data_point.append(
                            probability_input(
                                dataset_name=dataset_name,
                                methods=method_name,
                                probabilities=method.testing.probabilities,
                                subject_id=subject_id,
                                channel=pseudo_trial,
                                kfold=count_Kfolds,
                                label=labels[epoch_number],
                                training_accuracy=method.training.accuracy,
                                training_timing=method.training.timing,
                                testing_timing=method.testing.timing,
                            )
                        )

    ce.to_df().to_csv(
        standard_saving_path(
            dataset_info,
            "_".join(activated_methods),
            version_name + "_all_probabilities",
            file_ending="csv",
        )
    )

    print("Congrats! The processing methods are done processing.")
