from pathlib import Path

from data_loaders import load_data_labels_based_on_dataset
from diffE_models import *
from diffE_utils import *

import pickle
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from ema_pytorch import EMA
from tqdm import tqdm
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from share import datasets_basic_infos

ROOT_VOTING_SYSTEM_PATH: Path = Path(__file__).parent.parent.resolve()

dataset_name = "aguilera_traditional"  # Only two things I should be able to change

# Folders and paths
dataset_foldername = dataset_name + "_dataset"
computer_root_path = str(ROOT_VOTING_SYSTEM_PATH) + "/Datasets/"
data_path = computer_root_path + dataset_foldername

dataset_info = datasets_basic_infos[dataset_name]


def diffE_evaluation(subject_id: int, X, Y, dataset_info, device: str =  "cuda:0"):

        # create an argument parser for the data loader path
        model_path: str = f'{ROOT_VOTING_SYSTEM_PATH}/Results/diffe_{dataset_name}_{subject_id}.pt'  # diffE_{subject_ID}.pt

        X = X[:, :, : -1 * (X.shape[2] % 8)] # 2^3=8 because there are 3 downs and ups halves.
        # Dataloader
        batch_size = 32
        batch_size2 = 260
        seed = 42
        train_loader, test_loader = get_dataloader(
            X, Y, batch_size, batch_size2, seed, shuffle=True
        )

        n_T = 1000
        ddpm_dim = 128
        encoder_dim = 256
        fc_dim = 512
        # Define model
        num_classes = dataset_info['#_class']
        channels = dataset_info['#_channels']

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
            Y = []
            Y_hat = []
            for x, y in train_loader:
                x, y = x.to(device).float(), y.type(torch.LongTensor).to(device)
                encoder_out = diffe.encoder(x)
                y_hat = diffe.fc(encoder_out[1])
                y_hat = F.softmax(y_hat, dim=1)

                Y.append(y.detach().cpu())
                Y_hat.append(y_hat.detach().cpu())

            # List of tensors to tensor to numpy
            Y = torch.cat(Y, dim=0).numpy()  # (N, )
            Y_hat = torch.cat(Y_hat, dim=0).numpy()  # (N, 13): has to sum to 1 for each row

            return Y, Y_hat


if __name__ == "__main__":
    print(f'CUDA is available? {torch.cuda.is_available()}')

    dataset_name = "aguilera_traditional"  # Only two things I should be able to change

    # Folders and paths
    dataset_foldername = dataset_name + "_dataset"
    computer_root_path = str(ROOT_VOTING_SYSTEM_PATH) + "/Datasets/"
    data_path = computer_root_path + dataset_foldername
    print(data_path)
    dataset_info = datasets_basic_infos[dataset_name]
    all_subjects_accuracy: list = []

    for subject_id in range(1, dataset_info['subjects'] + 1):
        print(f"\nSubject: {subject_id}")
        _, X, Y = load_data_labels_based_on_dataset(dataset_name, subject_id, data_path)

        Y_returned, Y_hat = diffE_evaluation(subject_id=subject_id, X=X, Y=Y, dataset_info=dataset_info)

        # Accuracy and Confusion Matrix
        accuracy = top_k_accuracy_score(Y_returned, Y_hat, k=1, labels=np.arange(0, dataset_info['#_class']))
        all_subjects_accuracy.append(accuracy)
        print(f'Test accuracy: {accuracy:.2f}%')
    print(f'Test accuracy: {all_subjects_accuracy}')