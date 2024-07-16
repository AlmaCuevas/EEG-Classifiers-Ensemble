import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from data_loaders import load_data_labels_based_on_dataset
from data_utils import standard_saving_path
from DiffE.diffE_models import Decoder, DiffE, Encoder, LinearClassifier
from DiffE.diffE_utils import get_dataloader
from share import ROOT_VOTING_SYSTEM_PATH, datasets_basic_infos
from sklearn.model_selection import StratifiedKFold

# todo: add the test template
# todo: do the deap thing about the FFT: https://github.com/tongdaxu/EEG_Emotion_Classifier_DEAP/blob/master/Preprocess_Deap.ipynb

threshold_for_bug = 0.00000001  # could be any value, ex numpy.min


def diffE_train(data, labels) -> tuple[str, float]:
    pass


def diffE_test(subject_id: int, X, dataset_info: dict, device: str = "cuda:0"):
    # From diffe_evaluation
    model_path: str = standard_saving_path(
        dataset_info, "DiffE", "", file_ending="pt", subject_id=subject_id
    )

    X = X[
        :, :, : -1 * (X.shape[2] % 8)
    ]  # 2^3=8 because there are 3 downs and ups halves.
    # Dataloader
    batch_size = 32
    batch_size2 = 260
    seed = 42
    train_loader, _ = get_dataloader(
        X,
        [0] * (X.shape[0]),  # Y=0 JUST TO NOT LEAVE IT EMPTY, HERE IT ISN'T USED
        batch_size,
        batch_size2,
        seed,
        shuffle=True,
    )

    n_T = 1000
    ddpm_dim = 128
    encoder_dim = 256
    fc_dim = 512
    # Define model
    num_classes = dataset_info["#_class"]
    channels = dataset_info["#_channels"]

    encoder = Encoder(in_channels=channels, dim=encoder_dim).to(device)
    decoder = Decoder(
        in_channels=channels, n_feat=ddpm_dim, encoder_dim=encoder_dim
    ).to(device)
    fc = LinearClassifier(encoder_dim, fc_dim, emb_dim=num_classes).to(device)
    diffe = DiffE(encoder, decoder, fc).to(device)

    # load the pre-trained model from the file
    diffe.load_state_dict(torch.load(model_path))

    diffe.eval()

    with torch.no_grad():
        Y_hat = []
        for x, _ in train_loader:
            x = x.to(device).float()
            encoder_out = diffe.encoder(x)
            y_hat = diffe.fc(encoder_out[1])
            y_hat = F.softmax(y_hat, dim=1)

            Y_hat.append(y_hat.detach().cpu())
        Y_hat = torch.cat(Y_hat, dim=0).numpy()  # (N, 13): has to sum to 1 for each row
    return Y_hat


if __name__ == "__main__":
    # Manual Inputs
    # dataset_name = "torres"  # Only two things I should be able to change
    datasets = ["aguilera_traditional", "aguilera_gamified", "torres"]
    for dataset_name in datasets:
        version_name = "customized_only"  # To keep track what the output processing alteration went through

        # Folders and paths
        dataset_foldername = dataset_name + "_dataset"
        computer_root_path = ROOT_VOTING_SYSTEM_PATH + "/Datasets/"
        data_path = computer_root_path + dataset_foldername
        print(data_path)
        # Initialize
        processing_name: str = ""
        if dataset_name not in datasets_basic_infos:
            raise Exception(
                f"Not supported dataset named '{dataset_name}', choose from the following: aguilera_traditional, aguilera_gamified, nieto, coretto or torres."
            )
        dataset_info: dict = datasets_basic_infos[dataset_name]

        mean_accuracy_per_subject: list = []
        results_df = pd.DataFrame()

        for subject_id in range(
            1, dataset_info["subjects"] + 1
        ):  # Only two things I should be able to change
            print(subject_id)
            with open(
                f"{ROOT_VOTING_SYSTEM_PATH}/Results/{version_name}_{dataset_name}.txt",
                "a",
            ) as f:
                f.write(f"Subject: {subject_id}\n\n")
            epochs, data, labels = load_data_labels_based_on_dataset(
                dataset_info, subject_id, data_path
            )
            data[data < threshold_for_bug] = (
                threshold_for_bug  # To avoid the error "SVD did not convergence"
            )
            # Do cross-validation
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            acc_over_cv = []
            testing_time_over_cv = []
            training_time = []
            accuracy = 0
            for train, test in cv.split(epochs, labels):
                print(
                    "******************************** Training ********************************"
                )
                start = time.time()
                clf, accuracy, processing_name = diffE_train(data[train], labels[train])
                training_time.append(time.time() - start)
                with open(
                    f"{ROOT_VOTING_SYSTEM_PATH}/Results/{version_name}_{dataset_name}.txt",
                    "a",
                ) as f:
                    f.write(f"{processing_name}\n")
                    f.write(f"Accuracy of training: {accuracy}\n")
                print(
                    "******************************** Test ********************************"
                )
                pred_list = []
                testing_time = []
                for epoch_number in test:
                    start = time.time()
                    array = diffE_test(clf, np.asarray([data[epoch_number]]))
                    end = time.time()
                    testing_time.append(end - start)
                    print(dataset_info["target_names"])
                    print("Probability voting system: ", array)

                    voting_system_pred = np.argmax(array)
                    pred_list.append(voting_system_pred)
                    print("Prediction: ", voting_system_pred)
                    print("Real: ", labels[epoch_number])

                acc = np.mean(pred_list == labels[test])
                testing_time_over_cv.append(np.mean(testing_time))
                acc_over_cv.append(acc)
                with open(
                    f"{ROOT_VOTING_SYSTEM_PATH}/Results/{version_name}_{dataset_name}.txt",
                    "a",
                ) as f:
                    f.write(f"Prediction: {pred_list}\n")
                    f.write(f"Real label:{labels[test]}\n")
                    f.write(f"Mean accuracy in KFold: {acc}\n")
                print("Mean accuracy in KFold: ", acc)
            mean_acc_over_cv = np.mean(acc_over_cv)

            with open(
                f"{ROOT_VOTING_SYSTEM_PATH}/Results/{version_name}_{dataset_name}.txt",
                "a",
            ) as f:
                f.write(f"Final acc: {mean_acc_over_cv}\n\n\n\n")
            print(f"Final acc: {mean_acc_over_cv}")

            temp = pd.DataFrame(
                {
                    "Methods": [processing_name] * len(acc_over_cv),
                    "Subject ID": [subject_id] * len(acc_over_cv),
                    "Version": [version_name] * len(acc_over_cv),
                    "Training Accuracy": [accuracy] * len(acc_over_cv),
                    "Training Time": training_time,
                    "Testing Accuracy": acc_over_cv,
                    "Testing Time": testing_time_over_cv,
                }
            )  # The idea is that the most famous one is the one I use for this dataset
            results_df = pd.concat([results_df, temp])

        results_df.to_csv(
            f"{ROOT_VOTING_SYSTEM_PATH}/Results/{version_name}_{dataset_name}.csv"
        )

    print("Congrats! The processing methods are done processing.")
