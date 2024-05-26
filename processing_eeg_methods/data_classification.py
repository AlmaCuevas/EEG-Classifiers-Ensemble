# When I preprocess, the first and honestly only useful algorithms doesn't work. SVD did not converge. Why?
# Maybe if you can solve that you can use them and get better performance.

import copy

from sklearn.model_selection import StratifiedKFold

from BigProject.CNN_LSTM_probs import CNN_LSTM_test, CNN_LSTM_train
from multiple_transforms_with_models.customized_probs import customized_train, customized_test
from multiple_transforms_with_models.transforms_selectKBest_probs import selected_transformers_train, transform_data, selected_transformers_test
from data_loaders import load_data_labels_based_on_dataset
from arl_eegmodels.examples.ERP_probs import ERP_train, ERP_test
from BigProject.GRU_probs import GRU_train, GRU_test
from BigProject.LSTM_probs import LSTM_train, LSTM_test
from share import datasets_basic_infos, ROOT_VOTING_SYSTEM_PATH
import numpy as np
import time
from TCACNet.tcanet_probs import tcanet_train, tcanet_test
from XDAWN.xdawn_probs import xdawn_train, xdawn_test
from sklearn.preprocessing import normalize
from diffE_training import diffE_train
from diffE_evaluation import diffE_evaluation
from Extraction.get_features_probs import extractions_train, extractions_test, by_frequency_band

import pandas as pd

def group_methods_train(
    dataset_name: str,
    subject_id: int,
    methods: dict,
    models_outputs: dict,
    data,
    epochs,
    labels,
    dataset_info,
):
    target_names = dataset_info["target_names"]
    processing_name: str = ''
    columns_list: list = []
    transform_methods: dict = {}
    # Standard methods:
    if methods["selected_transformers"]:
        print("selected_transformers")
        start_time = time.time()
        features_train_df, transform_methods = transform_data(data, dataset_info=dataset_info, labels=labels)
        models_outputs["selected_transformers_clf"], models_outputs["selected_transformers_accuracy"], columns_list = selected_transformers_train(features_train_df, labels)
        models_outputs["selected_transformers_train_timer"] = time.time() - start_time
    if methods["customized"]:
        print("customized")
        start_time = time.time()
        models_outputs["customized_clf"], models_outputs["customized_accuracy"], processing_name = customized_train(copy.deepcopy(data), labels)
        models_outputs["customized_train_timer"] = time.time() - start_time
    if methods["XDAWN_LogReg"]:
        print("XDAWN_LogReg")
        start_time = time.time()
        (
            models_outputs["XDAWN_LogReg_clf"],
            models_outputs["XDAWN_LogReg_accuracy"],
        ) = xdawn_train(epochs, labels, target_names)
        models_outputs["XDAWN_LogReg_train_timer"] = time.time() - start_time

    # No standard methods:
    if methods["TCANET_Global_Model"]:
        print("TCANET_Global_Model")
        start_time = time.time()
        (
            models_outputs["model_only_global"],
            models_outputs["model_only_local"],
            models_outputs["model_only_top"],
            models_outputs["TCANET_Global_Model_accuracy"],
        ) = tcanet_train(
            dataset_name,
            subject_id,
            data,
            labels,
            channels=dataset_info["#_channels"],
            ONLY_GLOBAL_MODEL=True,
        )
        models_outputs["TCANET_Global_Model_train_timer"] = time.time() - start_time
    if methods["TCANET"]:
        print("TCANET")
        start_time = time.time()
        (
            models_outputs["model_tcanet_global"],
            models_outputs["model_tcanet_local"],
            models_outputs["model_tcanet_top"],
            models_outputs["TCANET_accuracy"],
        ) = tcanet_train(
            dataset_name,
            subject_id,
            data,
            labels,
            channels=dataset_info["#_channels"],
            ONLY_GLOBAL_MODEL=False,
        )
        models_outputs["TCANET_train_timer"] = time.time() - start_time
    if methods["DeepConvNet"]: #todo: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy
        print("DeepConvNet")
        start_time = time.time()
        models_outputs["DeepConvNet_clf"] = ERP_train(dataset_name, copy.deepcopy(data), labels, dataset_info)
        models_outputs[
            "DeepConvNet_accuracy"
        ] = 0.25  # You can't trust the accuracy, so I don't even calculate it.
        models_outputs["DeepConvNet_train_timer"] = time.time() - start_time
    if methods["LSTM"]:
        print("LSTM")
        start_time = time.time()
        models_outputs["LSTM_clf"], models_outputs["LSTM_accuracy"] = LSTM_train(
            dataset_name, data, labels, dataset_info['#_class']
        )
        models_outputs["LSTM_train_timer"] = time.time() - start_time
    if methods["GRU"]:
        print("GRU")
        start_time = time.time()
        models_outputs["GRU_clf"], models_outputs["GRU_accuracy"] = GRU_train(dataset_name, data, labels,
                                                                              dataset_info['#_class'])
        models_outputs["GRU_train_timer"] = time.time() - start_time
    if methods["CNN_LSTM"]:
        print("CNN_LSTM")
        start_time = time.time()
        models_outputs["CNN_LSTM_clf"], models_outputs["CNN_LSTM_accuracy"] = CNN_LSTM_train(
            data, labels, dataset_info['#_class']
        )
        models_outputs["CNN_LSTM_train_timer"] = time.time() - start_time
    if methods["diffE"]:
        print("diffE")
        start_time = time.time()
        models_outputs["diffE_accuracy"] = diffE_train(subject_id=subject_id, X=data, Y=labels, dataset_info=dataset_info) # The trained clf is saved in a file
        models_outputs["diffE_train_timer"] = time.time() - start_time

    if methods["feature_extraction"]:
        print("feature_extraction")
        start_time = time.time()
        features_df = by_frequency_band(data, dataset_info)
        models_outputs["feature_extraction_clf"], models_outputs["feature_extraction_accuracy"] =extractions_train(features_df, labels)
        models_outputs["feature_extraction_train_timer"] = time.time() - start_time

    return models_outputs, processing_name, columns_list, transform_methods


def group_methods_test(methods: dict, columns_list: list, transform_methods: dict, models_outputs: dict, data_array, data_epoch):
    if methods["selected_transformers"] and models_outputs["selected_transformers_clf"]:
        print("selected_transformers")
        start_time = time.time()
        features_test_df, _ = transform_data(data_array, dataset_info=dataset_info, labels=None,
                                             transform_methods=transform_methods)
        models_outputs["selected_transformers_probabilities"] = normalize(
            selected_transformers_test(models_outputs["selected_transformers_clf"], features_test_df[columns_list])
        )
        models_outputs["selected_transformers_test_timer"] = time.time() - start_time
    if methods["customized"] and models_outputs["customized_clf"]:
        print("customized")
        start_time = time.time()
        models_outputs["customized_probabilities"] = normalize(
            customized_test(models_outputs["customized_clf"], data_array)
        )
        models_outputs["customized_test_timer"] = time.time() - start_time
    if methods["XDAWN_LogReg"] and models_outputs["XDAWN_LogReg_clf"]:
        print("XDAWN_LogReg")
        start_time = time.time()
        models_outputs["XDAWN_LogReg_probabilities"] = normalize(
            xdawn_test(models_outputs["XDAWN_LogReg_clf"], data_epoch)
        )
        models_outputs["XDAWN_LogReg_test_timer"] = time.time() - start_time
    if methods["TCANET_Global_Model"] and models_outputs["model_only_global"]:
        print("TCANET_Global_Model")
        start_time = time.time()
        models_outputs["TCANET_Global_Model_probabilities"] = normalize(tcanet_test(
            dataset_name,
            subject_id,
            models_outputs["model_only_global"],
            models_outputs["model_only_local"],
            models_outputs["model_only_top"],
            data_array,
            dataset_info["#_channels"],
            True,
        ))
        models_outputs["TCANET_Global_Model_test_timer"] = time.time() - start_time
    if methods["TCANET"] and models_outputs["model_tcanet_global"]:
        print("TCANET")
        start_time = time.time()
        models_outputs["TCANET_array"] = normalize(tcanet_test(
            dataset_name,
            subject_id,
            models_outputs["model_tcanet_global"],
            models_outputs["model_tcanet_local"],
            models_outputs["model_tcanet_top"],
            data_array,
            dataset_info["#_channels"],
            False,
        ))
        models_outputs["TCANET_test_timer"] = time.time() - start_time
    if methods["DeepConvNet"]:
        print("DeepConvNet")
        start_time = time.time()
        models_outputs["DeepConvNet_probabilities"] = normalize(ERP_test(
            models_outputs["DeepConvNet_clf"], copy.deepcopy(data_array)
        ))
        models_outputs["DeepConvNet_test_timer"] = time.time() - start_time
    if methods["LSTM"]:
        print("LSTM")
        start_time = time.time()
        models_outputs["LSTM_probabilities"] = normalize(LSTM_test(models_outputs["LSTM_clf"], data_array))
        models_outputs["LSTM_test_timer"] = time.time() - start_time
    if methods["GRU"]:
        print("GRU")
        start_time = time.time()
        models_outputs["GRU_probabilities"] = normalize(GRU_test(models_outputs["GRU_clf"], data_array))
        models_outputs["GRU_test_timer"] = time.time() - start_time
    if methods["CNN_LSTM"]:
        print("CNN_LSTM")
        start_time = time.time()
        models_outputs["CNN_LSTM_probabilities"] = normalize(CNN_LSTM_test(
            models_outputs["CNN_LSTM_clf"], data_array
        ))
        models_outputs["CNN_LSTM_test_timer"] = time.time() - start_time
    if methods["diffE"]:
        print("diffE")
        start_time = time.time()
        _, models_outputs["diffE_probabilities"] = normalize(diffE_evaluation(subject_id=subject_id, X=data_array, Y=[1], dataset_info=dataset_info)) # Y is just a filling to avoid editing the code further, but its not used
        models_outputs["diffE_test_timer"] = time.time() - start_time

    if methods["feature_extraction"]:
        print("feature_extraction")
        start_time = time.time()
        features_df = by_frequency_band(data_array, dataset_info)
        models_outputs["feature_extraction_probabilities"] = normalize(extractions_test(models_outputs["feature_extraction_clf"], features_df))
        models_outputs["feature_extraction_test_timer"] = time.time() - start_time

    return methods, models_outputs

def voting_decision(methods: dict, models_outputs: dict, voting_by_mode: bool = False):
    if voting_by_mode:
        probs_list = [
            np.argmax(models_outputs[f"{method}_probabilities"])
            for method in methods if models_outputs[f"{method}_accuracy"] is not np.nan
        ]
        return probs_list
    else:  # voting_by_array_probabilities
        probs_list = [
            np.multiply(
                models_outputs[f"{method}_probabilities"], models_outputs[f"{method}_accuracy"]
            )
            for method in methods
        ]

        # You need to select at least two for this to work
        print(probs_list)
        probs = np.nanmean(probs_list, axis=0)  # Mean over columns

        return probs

def probabilities_to_answer(probs_list: list, voting_by_mode: bool = False):
    if voting_by_mode:
        return max(set(probs_list), key=probs_list.count)
    else:  # voting_by_array_probabilities
        probs = np.nanmean(probs_list, axis=0)  # Mean over columns
        return np.argmax(probs)


if __name__ == "__main__":
    # Manual Inputs
    #dataset_name = "torres"  # Only two things I should be able to change
   # datasets = ['aguilera_gamified', 'aguilera_traditional', 'torres']
    datasets = ['braincommand']
    voting_by_mode = False
    for dataset_name in datasets:
        selected_classes = [0, 1, 2, 3]
        version_name = "multiple_classifier" # To keep track what the output processing alteration went through

        # Folders and paths
        dataset_foldername = dataset_name + "_dataset"
        computer_root_path = ROOT_VOTING_SYSTEM_PATH + "/Datasets/"
        data_path = computer_root_path + dataset_foldername
        print(data_path)
        # Initialize
        methods = {
            "selected_transformers": False, # Like customized but with frequency bands and selected columns
            "customized": True,
            "XDAWN_LogReg": False, # Todo: ValueError: Cannot apply correct_overlap if epochs were baselined.
            "TCANET_Global_Model": False, # BAD
            "TCANET": False, # BAD #todo It always gives answer 0. Even when the training is high. why?
            "diffE": False, #todo: for ic_bci_2020 diffE evaluation doesnt run because the array shape is weird? check! # ERROR IN EVALUATION OF TEST, BRAINCOMMAND: setting an array element with a sequence. The requested array has an inhomogeneous shape after 2 dimensions. The detected shape was (2, 1) + inhomogeneous part.
            "DeepConvNet": False, # BAD #todo: ValueError: Shapes (None, 5) and (None, 4) are incompatible
            "LSTM": False, # BAD
            "GRU": False, # BAD
            "CNN_LSTM": False, # BAD
            "feature_extraction": False,
        }
        keys = list(methods.keys())

        processing_name: str = ''
        if dataset_name not in datasets_basic_infos:
            raise Exception(
                f"Not supported dataset named '{dataset_name}', choose from the following: aguilera_traditional, aguilera_gamified, nieto, coretto or torres."
            )
        dataset_info: dict = datasets_basic_infos[dataset_name]

        dataset_info['#_class'] = len(selected_classes)

        models_outputs = dict.fromkeys([key + "_accuracy" for key in keys], np.nan)
        models_outputs.update(
            dict.fromkeys(
                [key + "_probabilities" for key in keys], np.asarray([[np.nan] * dataset_info['#_class']])
            )
        )

        mean_accuracy_per_subject: list = []
        results_df = pd.DataFrame()
        activated_methods: list = [k for k, v in methods.items() if v == True]

        for subject_id in range(
            1, dataset_info["subjects"] + 1
        ):  # Only two things I should be able to change
            print(subject_id)
            with open(
                f"{ROOT_VOTING_SYSTEM_PATH}/Results/{dataset_name}/{version_name}_{'_'.join(activated_methods)}_{dataset_name}.txt",
                "a",
            ) as f:
                f.write(f"Subject: {subject_id}\n\n")
            epochs, data, labels = load_data_labels_based_on_dataset(dataset_info, subject_id, data_path, selected_classes=selected_classes, threshold_for_bug = 0.00000001)  # could be any value, ex numpy.min

            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) # Do cross-validation
            acc_over_cv = []
            test_accuracy_dict_lists = dict((f'{k}_acc', []) for k in activated_methods)
            test_timer_dict_lists = dict((f'{k}_test_timer', []) for k in activated_methods)

            train_accuracy_dict_lists = dict((f'{k}_acc', []) for k in activated_methods)
            train_timer_dict_lists = dict((f'{k}_train_timer', []) for k in activated_methods)
            for train, test in cv.split(epochs, labels):
                print(
                    "******************************** Training ********************************"
                )
                # start = time.time()
                models_outputs, processing_name, columns_list, transform_methods = group_methods_train(
                    dataset_name,
                    subject_id,
                    methods,
                    models_outputs,
                    data[train],
                    epochs[train],
                    labels[train],
                    dataset_info,
                )
                # end = time.time()
                activated_outputs = dict((k, v) for k, v in models_outputs.items() if k[:-9] in activated_methods)
                with open(
                    f"{ROOT_VOTING_SYSTEM_PATH}/Results/{dataset_name}/{version_name}_{'_'.join(activated_methods)}_{dataset_name}.txt",
                    "a",
                ) as f:
                    f.write(f"{activated_outputs}\n")
                # print("Training time: ", end - start)

                for k, v in models_outputs.items():
                    if 'accuracy' in k and k[:-9] in activated_methods:
                        train_accuracy_dict_lists[f'{k[:-9]}_acc'].append(v)
                    if 'train_timer' in k and k[:-12] in activated_methods:
                        train_timer_dict_lists[f'{k[:-12]}_train_timer'].append(v)

                print(
                    "******************************** Test ********************************"
                )
                pred_list = []
                test_pred_methods = dict((f'{k}_acc', []) for k in activated_methods)
                test_timer_methods = dict((f'{k}_test_timer', []) for k in activated_methods)
                all_methods_pred_list = []
                for epoch_number in test:
                    # start = time.time()
                    methods, models_outputs = group_methods_test(
                        methods,
                        columns_list,
                        transform_methods,
                        models_outputs,
                        np.asarray([data[epoch_number]]),
                        epochs[epoch_number],
                    )
                    probs = voting_decision(methods, models_outputs, voting_by_mode)
                    voting_system_pred = probabilities_to_answer(probs, voting_by_mode)
                    # end = time.time()
                    # print("One epoch, testing time: ", end - start)
                    print(dataset_info["target_names"])
                    #print("Probability voting system: ", array)

                    for k, v in models_outputs.items():
                        if 'probabilities' in k and k[:-14] in activated_methods:
                            test_pred_methods[f'{k[:-14]}_acc'].append(np.argmax(v))
                        if 'test_timer' in k and k[:-11] in activated_methods:
                            test_timer_methods[f'{k[:-11]}_test_timer'].append(v)


                    pred_list.append(voting_system_pred)
                    print("Prediction: ", voting_system_pred)
                    print("Real: ", labels[epoch_number])

                for pred_method, value in test_pred_methods.items():
                    test_accuracy_dict_lists[pred_method].append(np.mean(value == labels[test])) # Accuracy per epoch

                for timer_method, value in test_timer_methods.items():
                    test_timer_dict_lists[timer_method].append(np.mean(value)) # Accuracy per epoch

                acc = np.mean(pred_list == labels[test])
                acc_over_cv.append(acc)
                with open(
                    f"{ROOT_VOTING_SYSTEM_PATH}/Results/{dataset_name}/{version_name}_{'_'.join(activated_methods)}_{dataset_name}.txt",
                    "a",
                ) as f:
                    f.write(f"Prediction: {pred_list}\n")
                    f.write(f"Real label:{labels[test]}\n")
                    f.write(f"Mean accuracy in KFold: {acc}\n")
                print("Mean accuracy in KFold: ", acc)
            mean_acc_over_cv = np.mean(acc_over_cv)
            test_final_accuracy = dict((k,np.mean(v)) for k, v in test_accuracy_dict_lists.items())
            test_final_timer = dict((k, np.mean(v)) for k, v in test_timer_dict_lists.items())

            train_final_accuracy = dict((k, np.nanmean(v)) for k, v in train_accuracy_dict_lists.items())
            train_final_timer = dict((k, np.mean(v)) for k, v in train_timer_dict_lists.items())

            with open(
                f"{ROOT_VOTING_SYSTEM_PATH}/Results/{dataset_name}/{version_name}_{'_'.join(activated_methods)}_{dataset_name}.txt",
                "a",
            ) as f:
                f.write(f"Final acc: {mean_acc_over_cv}\n\n\n\n")
            print(f"Final acc: {mean_acc_over_cv}")

            train_timer_outputs = dict((k, v) for k, v in models_outputs.items() if 'train_timer' in k)
            accuracy_outputs = dict((k, v) for k, v in models_outputs.items() if 'accuracy' in k and k[:-9] in activated_methods)

            # SAVE DATAFRAME
            activated_methods_list = [w.replace('customized', processing_name) for w in activated_methods]
            temp = pd.DataFrame({'Methods': activated_methods_list, 'Subject ID': [subject_id] * len(activated_methods), 'Version': [version_name] * len(activated_methods), 'Train Accuracy': train_final_accuracy.values(), 'Train Timer': train_final_timer.values(), 'Test Accuracy': test_final_accuracy.values(), 'Test Timer': test_final_timer.values()}) # mean_accuracy_per_subject
            results_df = pd.concat([results_df, temp])

        results_df.to_csv(f"{ROOT_VOTING_SYSTEM_PATH}/Results/{dataset_name}/{version_name}_{'_'.join(activated_methods)}_{dataset_name}.csv")

    print("Congrats! The processing methods are done processing.")
