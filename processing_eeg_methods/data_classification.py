# When I preprocess, the first and honestly only useful algorithms doesn't work. SVD did not converge. Why?
# Maybe if you can solve that you can use them and get better performance.

import copy
import time

import numpy as np
import pandas as pd
from BigProject.GRU_probs import GRU_test, GRU_train
from BigProject.LSTM_probs import LSTM_test, LSTM_train
from data_loaders import load_data_labels_based_on_dataset
from data_utils import (
    convert_into_binary,
    convert_into_independent_channels,
    flat_a_list,
    get_dataset_basic_info,
    get_input_data_path,
    standard_saving_path,
)
from DiffE.diffE_probs import diffE_test
from DiffE.diffE_training import diffE_train
from features_extraction.get_features_probs import (
    by_frequency_band,
    extractions_test,
    extractions_train,
)
from multiple_transforms_with_models.customized_probs import (
    customized_test,
    customized_train,
)
from multiple_transforms_with_models.transforms_selectKBest_probs import (
    selected_transformers_test,
    selected_transformers_train,
    transform_data,
)
from NeuroTechX_dl_eeg.ShallowFBCSPNet_probs import (
    ShallowFBCSPNet_test,
    ShallowFBCSPNet_train,
)
from share import datasets_basic_infos
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize


def group_methods_train(
    subject_id: int,
    methods: dict,
    models_outputs: dict,
    data,
    labels,
    dataset_info,
):
    processing_name: str = ""
    columns_list: list = []
    transform_methods: dict = {}
    # Standard methods:
    if methods["selected_transformers"]:
        print("Selected Transformers")
        start_time = time.time()
        features_train_df, transform_methods = transform_data(
            data, dataset_info=dataset_info, labels=labels
        )
        (
            models_outputs["selected_transformers_clf"],
            models_outputs["selected_transformers_accuracy"],
            columns_list,
        ) = selected_transformers_train(features_train_df, labels)
        models_outputs["selected_transformers_train_timer"] = time.time() - start_time
    if methods["customized"]:
        print("customized")
        start_time = time.time()
        (
            models_outputs["customized_clf"],
            models_outputs["customized_accuracy"],
            processing_name,
        ) = customized_train(copy.deepcopy(data), labels)
        models_outputs["customized_train_timer"] = time.time() - start_time

    if methods["ShallowFBCSPNet"]:
        print("ShallowFBCSPNet")
        start_time = time.time()
        temp_data = (data * 1e6).astype(np.float32)
        model_ShallowFBCSPNet_accuracies = []
        for chosen_numbered_label in range(0, dataset_info["#_class"] + 1):
            temp_labels = convert_into_binary(
                labels.copy(), chosen_numbered_label=chosen_numbered_label
            )
            model_ShallowFBCSPNet_accuracies.append(
                ShallowFBCSPNet_train(
                    temp_data,
                    temp_labels,
                    chosen_numbered_label=chosen_numbered_label,
                    dataset_info=dataset_info,
                    subject_id=subject_id,
                )
            )
        models_outputs["ShallowFBCSPNet_accuracy"] = np.mean(
            model_ShallowFBCSPNet_accuracies
        )
        models_outputs["ShallowFBCSPNet_train_timer"] = time.time() - start_time
    if methods["LSTM"]:
        print("LSTM")
        start_time = time.time()
        models_outputs["LSTM_clf"], models_outputs["LSTM_accuracy"] = LSTM_train(
            dataset_info, data, labels, subject_id
        )
        models_outputs["LSTM_train_timer"] = time.time() - start_time
    if methods["GRU"]:
        print("GRU")
        start_time = time.time()
        models_outputs["GRU_clf"], models_outputs["GRU_accuracy"] = GRU_train(
            dataset_info["dataset_name"], data, labels, dataset_info["#_class"]
        )
        models_outputs["GRU_train_timer"] = time.time() - start_time
    if methods["diffE"]:
        print("diffE")
        start_time = time.time()
        models_outputs["diffE_accuracy"] = diffE_train(
            subject_id=subject_id, X=data, Y=labels, dataset_info=dataset_info
        )  # The trained clf is saved in a file
        models_outputs["diffE_train_timer"] = time.time() - start_time

    if methods["feature_extraction"]:
        print("feature_extraction")
        start_time = time.time()
        data_simplified, labels_simplified = convert_into_independent_channels(
            data, labels
        )
        features_df = by_frequency_band(data_simplified, dataset_info)
        (
            models_outputs["feature_extraction_clf"],
            models_outputs["feature_extraction_accuracy"],
        ) = extractions_train(features_df, labels_simplified)
        models_outputs["feature_extraction_train_timer"] = time.time() - start_time

    return models_outputs, processing_name, columns_list, transform_methods


def group_methods_test(
    methods: dict,
    columns_list: list,
    transform_methods: dict,
    models_outputs: dict,
    data_array,
):
    if methods["selected_transformers"] and models_outputs["selected_transformers_clf"]:
        print("selected_transformers")
        start_time = time.time()
        transforms_test_df, _ = transform_data(
            data_array,
            dataset_info=dataset_info,
            labels=None,
            transform_methods=transform_methods,
        )
        models_outputs["selected_transformers_probabilities"] = normalize(
            selected_transformers_test(
                models_outputs["selected_transformers_clf"],
                transforms_test_df[columns_list],
            )
        )
        models_outputs["selected_transformers_test_timer"] = time.time() - start_time
    if methods["customized"] and models_outputs["customized_clf"]:
        print("customized")
        start_time = time.time()
        models_outputs["customized_probabilities"] = normalize(
            customized_test(models_outputs["customized_clf"], data_array)
        )
        models_outputs["customized_test_timer"] = time.time() - start_time

    if methods["ShallowFBCSPNet"]:
        print("ShallowFBCSPNet")
        start_time = time.time()
        temp_data_array = (data_array * 1e6).astype(np.float32)
        ShallowFBCSPNet_arrays = []
        for chosen_numbered_label in range(0, dataset_info["#_class"]):
            ShallowFBCSPNet_arrays.append(
                ShallowFBCSPNet_test(
                    subject_id,
                    temp_data_array,
                    dataset_info,
                    chosen_numbered_label=chosen_numbered_label,
                )[0]
            )
        models_outputs["ShallowFBCSPNet_array"] = normalize(
            np.array([prob_array[1] for prob_array in ShallowFBCSPNet_arrays]).reshape(
                1, -1
            )
        )
        models_outputs["ShallowFBCSPNet_test_timer"] = time.time() - start_time
    if methods["LSTM"]:
        print("LSTM")
        start_time = time.time()
        models_outputs["LSTM_probabilities"] = normalize(
            LSTM_test(models_outputs["LSTM_clf"], data_array)
        )
        models_outputs["LSTM_test_timer"] = time.time() - start_time
    if methods["GRU"]:
        print("GRU")
        start_time = time.time()
        models_outputs["GRU_probabilities"] = normalize(
            GRU_test(models_outputs["GRU_clf"], data_array)
        )
        models_outputs["GRU_test_timer"] = time.time() - start_time
    if methods["diffE"]:
        print("diffE")
        start_time = time.time()
        models_outputs["diffE_probabilities"] = normalize(
            diffE_test(subject_id=subject_id, X=data_array, dataset_info=dataset_info)
        )
        models_outputs["diffE_test_timer"] = time.time() - start_time
    if methods["feature_extraction"]:
        print("feature_extraction")
        start_time = time.time()
        data_array_simplified, _ = convert_into_independent_channels(data, [1])
        features_df = by_frequency_band(data_array_simplified, dataset_info)
        models_outputs["feature_extraction_probabilities"] = normalize(
            extractions_test(models_outputs["feature_extraction_clf"], features_df)
        )
        models_outputs["feature_extraction_test_timer"] = time.time() - start_time

    return methods, models_outputs


def voting_decision(
    methods: dict,
    models_outputs: dict,
    voting_by_mode: bool = False,
    weighted_accuracy: bool = True,
):
    if voting_by_mode:
        probs_list = [
            np.argmax(models_outputs[f"{method}_probabilities"])
            for method in methods
            if models_outputs[f"{method}_accuracy"] is not np.nan
        ]
        return probs_list
    else:  # voting_by_array_probabilities
        if weighted_accuracy:
            probs_list = [
                np.multiply(
                    models_outputs[f"{method}_probabilities"],
                    models_outputs[f"{method}_accuracy"],
                )
                for method in methods
            ]
        else:
            probs_list = [
                models_outputs[f"{method}_probabilities"] for method in methods
            ]

        # You need to select at least two for this to work
        probs = np.nanmean(probs_list, axis=0)  # Mean over columns

        return probs


def probabilities_to_answer(probs_by_channels: list, voting_by_mode: bool = False):
    if voting_by_mode:
        overall_decision = flat_a_list(probs_by_channels)
        return max(set(overall_decision), key=list(overall_decision).count)
    else:  # voting_by_array_probabilities
        by_channel_decision = np.nanmean(probs_by_channels, axis=0)  # Mean over columns
        return np.argmax(by_channel_decision)


if __name__ == "__main__":
    # Manual Inputs
    datasets = ["braincommand"]
    voting_by_mode = False
    weighted_accuracy = False

    for dataset_name in datasets:
        selected_classes = [0, 1, 2, 3]
        version_name = "channel_independent_unweighted_accuracy"  # To keep track what the output processing alteration went through
        processing_name: str = ""

        data_path = get_input_data_path(dataset_name)

        # Initialize
        methods = {
            "selected_transformers": False,  # Training is over-fitted. Training accuracy >90
            "customized": False,  # Simpler than selected_transformers, only one transformer and no frequency bands. No need to activate both at the same time
            "ShallowFBCSPNet": True,
            "LSTM": False,  # Training is over-fitted. Training accuracy >90
            "GRU": False,  # Training is over-fitted. Training accuracy >90
            "diffE": False,  # It doesn't work if you only use one channel in the data
            "feature_extraction": False,
        }
        keys = list(methods.keys())
        # todo: It would be more convenient to get all the outputs, save all probabilities, and then do the combinations
        # todo:     instead of running all the code everytime I want to do a different combination.
        # todo:     With that idea, a notebook where all the combination results are calculated would be able plus
        # todo:     graphs and a "random" version, to know what the baseline is.

        dataset_info = get_dataset_basic_info(datasets_basic_infos, dataset_name)
        dataset_info["#_class"] = len(selected_classes)

        models_outputs = dict.fromkeys([key + "_accuracy" for key in keys], np.nan)
        models_outputs.update(
            dict.fromkeys(
                [key + "_probabilities" for key in keys],
                np.asarray([[np.nan] * dataset_info["#_class"]]),
            )
        )

        mean_accuracy_per_subject: list = []
        results_df = pd.DataFrame()
        activated_methods: list = [k for k, v in methods.items() if v]

        saving_txt_path: str = standard_saving_path(
            dataset_info, "_".join(activated_methods), version_name
        )

        save_original_channels = dataset_info["#_channels"]
        save_original_trials = dataset_info["total_trials"]

        for subject_id in range(22, 24):
            print(subject_id)
            with open(
                saving_txt_path,
                "a",
            ) as f:
                f.write(f"Subject: {subject_id}\n\n")

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
            acc_over_cv = []
            test_accuracy_dict_lists = dict((f"{k}_acc", []) for k in activated_methods)
            test_timer_dict_lists = dict(
                (f"{k}_test_timer", []) for k in activated_methods
            )

            train_accuracy_dict_lists = dict(
                (f"{k}_acc", []) for k in activated_methods
            )
            train_timer_dict_lists = dict(
                (f"{k}_train_timer", []) for k in activated_methods
            )
            for train, test in cv.split(epochs, labels):
                print(
                    "******************************** Training ********************************"
                )
                # Convert independent channels to pseudo-trials
                data_train, labels_train = convert_into_independent_channels(
                    data[train], labels[train]
                )
                data_train = np.transpose(np.array([data_train]), (1, 0, 2))

                models_outputs, processing_name, columns_list, transform_methods = (
                    group_methods_train(
                        subject_id,
                        methods,
                        models_outputs,
                        data_train,
                        labels_train,
                        dataset_info,
                    )
                )

                activated_outputs = dict(
                    (k, v)
                    for k, v in models_outputs.items()
                    if k[:-9] in activated_methods
                )
                with open(
                    saving_txt_path,
                    "a",
                ) as f:
                    f.write(f"{activated_outputs}\n")

                for k, v in models_outputs.items():
                    if "accuracy" in k and k[:-9] in activated_methods:
                        train_accuracy_dict_lists[f"{k[:-9]}_acc"].append(v)
                    if "train_timer" in k and k[:-12] in activated_methods:
                        train_timer_dict_lists[f"{k[:-12]}_train_timer"].append(v)

                print(
                    "******************************** Test ********************************"
                )
                pred_list = []
                test_pred_methods = dict((f"{k}_acc", []) for k in activated_methods)
                test_timer_methods = dict(
                    (f"{k}_test_timer", []) for k in activated_methods
                )
                all_methods_pred_list = []
                for epoch_number in test:
                    # Convert independent channels to pseudo-trials
                    data_test, labels_test = convert_into_independent_channels(
                        np.asarray([data[epoch_number]]), labels[epoch_number]
                    )
                    data_test = np.transpose(np.array([data_test]), (1, 0, 2))
                    probs_by_channel = []
                    for pseudo_trial in range(len(data_test)):
                        methods, models_outputs = group_methods_test(
                            methods,
                            columns_list,
                            transform_methods,
                            models_outputs,
                            np.asarray([data_test[pseudo_trial]]),
                        )
                        probs_by_channel.append(
                            voting_decision(
                                methods,
                                models_outputs,
                                voting_by_mode,
                                weighted_accuracy,
                            )
                        )

                    voting_system_pred = probabilities_to_answer(
                        probs_by_channel, voting_by_mode
                    )
                    print(dataset_info["target_names"])

                    for k, v in models_outputs.items():
                        if "probabilities" in k and k[:-14] in activated_methods:
                            test_pred_methods[f"{k[:-14]}_acc"].append(np.argmax(v))
                        if "test_timer" in k and k[:-11] in activated_methods:
                            test_timer_methods[f"{k[:-11]}_test_timer"].append(v)

                    pred_list.append(voting_system_pred)
                    print("Prediction: ", voting_system_pred)
                    print("Real: ", labels[epoch_number])

                for pred_method, value in test_pred_methods.items():
                    test_accuracy_dict_lists[pred_method].append(
                        np.mean(value == labels[test])
                    )  # Accuracy per epoch

                for timer_method, value in test_timer_methods.items():
                    test_timer_dict_lists[timer_method].append(
                        np.mean(value)
                    )  # Accuracy per epoch

                acc = np.mean(pred_list == labels[test])
                acc_over_cv.append(acc)
                with open(
                    saving_txt_path,
                    "a",
                ) as f:
                    f.write(f"Prediction: {pred_list}\n")
                    f.write(f"Real label:{labels[test]}\n")
                    f.write(f"Mean accuracy in KFold: {acc}\n")
                print("Mean accuracy in KFold: ", acc)
            mean_acc_over_cv = np.mean(acc_over_cv)
            test_final_accuracy = dict(
                (k, np.mean(v)) for k, v in test_accuracy_dict_lists.items()
            )
            test_final_timer = dict(
                (k, np.mean(v)) for k, v in test_timer_dict_lists.items()
            )

            train_final_accuracy = dict(
                (k, np.nanmean(v)) for k, v in train_accuracy_dict_lists.items()
            )
            train_final_timer = dict(
                (k, np.mean(v)) for k, v in train_timer_dict_lists.items()
            )

            with open(
                saving_txt_path,
                "a",
            ) as f:
                f.write(f"Final acc: {mean_acc_over_cv}\n\n\n\n")
            print(f"Final acc: {mean_acc_over_cv}")

            train_timer_outputs = dict(
                (k, v) for k, v in models_outputs.items() if "train_timer" in k
            )
            accuracy_outputs = dict(
                (k, v)
                for k, v in models_outputs.items()
                if "accuracy" in k and k[:-9] in activated_methods
            )

            # SAVE DATAFRAME
            activated_methods_list = [
                w.replace("customized", processing_name) for w in activated_methods
            ]
            temp = pd.DataFrame(
                {
                    "Methods": activated_methods_list,
                    "Subject ID": [subject_id] * len(activated_methods),
                    "Version": [version_name] * len(activated_methods),
                    "Train Accuracy": train_final_accuracy.values(),
                    "Train Timer": train_final_timer.values(),
                    "Test Accuracy": test_final_accuracy.values(),
                    "Test Timer": test_final_timer.values(),
                }
            )  # mean_accuracy_per_subject
            results_df = pd.concat([results_df, temp])

        results_df.to_csv(
            standard_saving_path(
                dataset_info,
                "_".join(activated_methods),
                version_name,
                file_ending="csv",
            )
        )

    print("Congrats! The processing methods are done processing.")
