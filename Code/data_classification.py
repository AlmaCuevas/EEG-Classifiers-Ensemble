# When I preprocess, the first and honestly only useful algorithms doesn't work. SVD did not converge. Why?
# Maybe if you can solve that you can use them and get better performance.

import copy

from sklearn.model_selection import StratifiedKFold

from CNN_LSTM_probs import CNN_LSTM_test, CNN_LSTM_train
from CSP_LDA_probs import CSP_LDA_train, CSP_LDA_test
from Code.data_loaders import load_data_labels_based_on_dataset
from ERP_probs import ERP_train, ERP_test
from GRU_probs import GRU_train, GRU_test
from LSTM_probs import LSTM_train, LSTM_test
from riemman_SVC_probs import riemman_train, riemman_test
from share import datasets_basic_infos
import numpy as np

from tcanet_probs import tcanet_train, tcanet_test
from xdawn_riemman_probs import xdawn_riemman_train, xdawn_riemman_test
from xdwan_probs import xdawn_train, xdawn_test
from sklearn.preprocessing import normalize
from pathlib import Path

ROOT_VOTING_SYSTEM_PATH: Path = Path(__file__).parent.parent.resolve()

threshold_for_bug = 0.00000001  # could be any value, ex numpy.min

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
    if methods["CSP_LDA"]:
        print("CSP_LDA")
        models_outputs["CSP_LDA_clf"], models_outputs["CSP_LDA_accuracy"] = CSP_LDA_train(
            copy.deepcopy(data), labels, target_names
        )
    if methods["RIEMMAN_SVC"]:
        print("RIEMMAN_SVC")
        (
            models_outputs["RIEMMAN_SVC_clf"],
            models_outputs["RIEMMAN_SVC_accuracy"],
        ) = riemman_train(data, labels, target_names)
    if methods["XDAWN_RIEMMAN"]:
        print("XDAWN_RIEMMAN")
        (
            models_outputs["XDAWN_RIEMMAN_clf"],
            models_outputs["XDAWN_RIEMMAN_accuracy"],
        ) = xdawn_riemman_train(data, labels, target_names)
    if methods["XDAWN_LogReg"]:
        print("XDAWN_LogReg")
        (
            models_outputs["XDAWN_LogReg_clf"],
            models_outputs["XDAWN_LogReg_accuracy"],
        ) = xdawn_train(epochs, labels, target_names)
    if methods["TCANET_Global_Model"]:
        print("TCANET_Global_Model")
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
    if methods["TCANET"]:
        print("TCANET")
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
    if methods["DeepConvNet"]: #todo: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy
        print("DeepConvNet")
        models_outputs["DeepConvNet_clf"] = ERP_train(dataset_name, copy.deepcopy(data), labels, dataset_info)
        models_outputs[
            "DeepConvNet_accuracy"
        ] = 0.25  # You can't trust the accuracy, so I don't even calculate it.
    if methods["LSTM"]:
        print("LSTM")
        models_outputs["LSTM_clf"], models_outputs["LSTM_accuracy"] = LSTM_train(
            dataset_name, data, labels
        )
    if methods["GRU"]:
        print("GRU")
        models_outputs["GRU_clf"], models_outputs["GRU_accuracy"] = GRU_train(dataset_name, data, labels)
    if methods["CNN_LSTM"]:
        models_outputs["CNN_LSTM_clf"], models_outputs["CNN_LSTM_accuracy"] = CNN_LSTM_train(
            data, labels
        )
    if methods["diffE"]:
        print("diffE")
        print("Not implemented")

    return models_outputs


def group_methods_test(methods: dict, models_outputs: dict, data_array, data_epoch):
    if methods["CSP_LDA"] and models_outputs["CSP_LDA_clf"]:
        print("CSP_LDA")
        models_outputs["CSP_LDA_probabilities"] = normalize(
            CSP_LDA_test(models_outputs["CSP_LDA_clf"], data_array)
        )
    if methods["RIEMMAN_SVC"] and models_outputs["RIEMMAN_SVC_clf"]:
        print("RIEMMAN_SVC")
        models_outputs["RIEMMAN_SVC_probabilities"] = normalize(
            riemman_test(models_outputs["RIEMMAN_SVC_clf"], data_array)
        )
    if methods["XDAWN_RIEMMAN"] and models_outputs["XDAWN_RIEMMAN_clf"]:
        print("XDAWN_RIEMMAN")
        models_outputs["XDAWN_RIEMMAN_probabilities"] = normalize(
            xdawn_riemman_test(models_outputs["XDAWN_RIEMMAN_clf"], data_array)
        )
    if methods["XDAWN_LogReg"] and models_outputs["XDAWN_LogReg_clf"]:
        print("XDAWN_LogReg")
        models_outputs["XDAWN_LogReg_probabilities"] = normalize(
            xdawn_test(models_outputs["XDAWN_LogReg_clf"], data_epoch)
        )
    if methods["TCANET_Global_Model"] and models_outputs["model_only_global"]:
        print("TCANET_Global_Model")
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
    if methods["TCANET"] and models_outputs["model_tcanet_global"]:
        print("TCANET")
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
    if methods["DeepConvNet"]:
        print("DeepConvNet")
        models_outputs["DeepConvNet_probabilities"] = normalize(ERP_test(
            models_outputs["DeepConvNet_clf"], copy.deepcopy(data_array)
        ))
    if methods["LSTM"]:
        models_outputs["LSTM_probabilities"] = normalize(LSTM_test(models_outputs["LSTM_clf"], data_array))
    if methods["GRU"]:
        print("GRU")
        models_outputs["GRU_probabilities"] = normalize(GRU_test(models_outputs["GRU_clf"], data_array))
    if methods["CNN_LSTM"]:
        print("CNN_LSTM")
        models_outputs["CNN_LSTM_probabilities"] = normalize(CNN_LSTM_test(
            models_outputs["CNN_LSTM_clf"], data_array
        ))
    if methods["diffE"]:
        print("diffE")
        print("Not implemented")

    probs_list = [
        np.multiply(
            models_outputs[f"{method}_probabilities"], models_outputs[f"{method}_accuracy"]
        )
        for method in methods
    ]

    # You need to select at least two for this to work
    probs = np.nanmean(probs_list, axis=0) # Mean over columns
    return probs


if __name__ == "__main__":
    # Manual Inputs
    dataset_name = "aguilera_gamified"  # Only two things I should be able to change

    # Folders and paths
    dataset_foldername = dataset_name + "_dataset"
    computer_root_path = str(ROOT_VOTING_SYSTEM_PATH) + "/Datasets/"
    data_path = computer_root_path + dataset_foldername
    print(data_path)
    # Initialize
    methods = {
        "CSP_LDA": True,
        "RIEMMAN_SVC": True,
        "XDAWN_RIEMMAN": True,
        "XDAWN_LogReg": True,
        "TCANET_Global_Model": False,
        "TCANET": False, #todo It always gives answer 0. Even when the training is high. why?
        "diffE": False,
        "DeepConvNet": False,
        "LSTM": False,
        "GRU": False,
        "CNN_LSTM": False,
    }
    keys = list(methods.keys())
    models_outputs = dict.fromkeys([key + "_accuracy" for key in keys], np.nan)
    models_outputs.update(
        dict.fromkeys(
            [key + "_probabilities" for key in keys], np.asarray([[0, 0, 0, 0]])
        )
    )

    if dataset_name not in [
        "aguilera_traditional",
        "aguilera_gamified",
        "nieto",
        "coretto",
        "torres",
    ]:
        raise Exception(
            f"Not supported dataset named '{dataset_name}', choose from the following: aguilera_traditional, aguilera_gamified, nieto, coretto or torres."
        )
    dataset_info = datasets_basic_infos[dataset_name]

    for subject_id in range(
        11, dataset_info["subjects"] + 1 #todo: run not normalized again. I think normalized is better though
    ):  # Only two things I should be able to change
        print(subject_id)
        activated_methods = [k for k, v in methods.items() if v == True]
        with open(
            f"{str(ROOT_VOTING_SYSTEM_PATH)}/Results/{'_'.join(activated_methods)}_{dataset_name}.txt",
            "a",
        ) as f:
            f.write(f"Subject: {subject_id}\n\n")
        epochs, data, labels = load_data_labels_based_on_dataset(dataset_name, subject_id, data_path)
        data[data < threshold_for_bug] = threshold_for_bug # To avoid the error "SVD did not convergence"
        # Do cross-validation
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        acc_over_cv = []
        for train, test in cv.split(epochs, labels):
            print(
                "******************************** Training ********************************"
            )
            # start = time.time()
            models_outputs = group_methods_train(
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
                f"{str(ROOT_VOTING_SYSTEM_PATH)}/Results/{'_'.join(activated_methods)}_{dataset_name}.txt",
                "a",
            ) as f:
                f.write(f"{activated_outputs}\n")
            # print("Training time: ", end - start)

            print(
                "******************************** Test ********************************"
            )
            pred_list = []
            for epoch_number in test:
                # start = time.time()
                array = group_methods_test(
                    methods,
                    models_outputs,
                    np.asarray([data[epoch_number]]),
                    epochs[epoch_number],
                )
                # end = time.time()
                # print("One epoch, testing time: ", end - start)
                print(dataset_info["target_names"])
                print("Probability: ", array)
                pred = np.argmax(array)
                pred_list.append(pred)
                print("Prediction: ", pred)
                print("Real: ", labels[epoch_number])
            acc = np.mean(pred_list == labels[test])
            acc_over_cv.append(acc)
            with open(
                f"{str(ROOT_VOTING_SYSTEM_PATH)}/Results/{'_'.join(activated_methods)}_{dataset_name}.txt",
                "a",
            ) as f:
                f.write(f"Prediction: {pred_list}\n")
                f.write(f"Real label:{labels[test]}\n")
                f.write(f"Mean accuracy in KFold: {acc}\n")
            print("Mean accuracy in KFold: ", acc)
        mean_acc_over_cv = np.mean(acc_over_cv)
        with open(
            f"{str(ROOT_VOTING_SYSTEM_PATH)}/Results/{'_'.join(activated_methods)}_{dataset_name}.txt",
            "a",
        ) as f:
            f.write(f"Final acc: {mean_acc_over_cv}\n\n\n\n")
        print(f"Final acc: {mean_acc_over_cv}")
        print("Congrats! The first draft of the voting system is now working.")
