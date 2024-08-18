import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_utils import standard_saving_path
from diffusion.diffE_models import (
    DDPM,
    ConditionalUNet,
    Decoder,
    DiffE,
    Encoder,
    LinearClassifier,
)
from diffusion.diffE_utils import get_dataloader
from ema_pytorch import EMA
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from tqdm import tqdm


# Evaluate function
def evaluate(encoder, fc, generator, device, number_of_labels: int = 4):
    labels = np.arange(0, number_of_labels)
    Y = []
    Y_hat = []
    for x, y in generator:
        x, y = x.to(device), y.type(torch.LongTensor).to(device)
        encoder_out = encoder(x)
        y_hat = fc(encoder_out[1])
        y_hat = F.softmax(y_hat, dim=1)

        Y.append(y.detach().cpu())
        Y_hat.append(y_hat.detach().cpu())

    # List of tensors to tensor to numpy
    Y = torch.cat(Y, dim=0).numpy()  # (N, )
    Y_hat = torch.cat(Y_hat, dim=0).numpy()  # (N, 13): has to sum to 1 for each row

    # Accuracy and Confusion Matrix
    accuracy = top_k_accuracy_score(Y, Y_hat, k=1, labels=labels)
    f1 = f1_score(Y, Y_hat.argmax(axis=1), average="macro", labels=labels)
    recall = recall_score(Y, Y_hat.argmax(axis=1), average="macro", labels=labels)
    precision = precision_score(Y, Y_hat.argmax(axis=1), average="macro", labels=labels)
    auc = roc_auc_score(Y, Y_hat, average="macro", multi_class="ovo", labels=labels)

    metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "auc": auc,
    }
    # df_cm = pd.DataFrame(confusion_matrix(Y, Y_hat.argmax(axis=1)))
    return metrics


def diffE_train(subject_id: int, X, Y, dataset_info, device: str = "cuda:0"):
    model_path: str = standard_saving_path(
        dataset_info, "DiffE", "", file_ending="pt", subject_id=subject_id
    )

    # This saves the training in a file
    X = X[
        :, :, : -1 * (X.shape[2] % 8)
    ]  # 2^3=8 because there are 3 downs and ups halves.

    # Dataloader
    device = torch.device(device)
    batch_size = 32
    batch_size2 = 260
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    print("Random Seed: ", seed)
    train_loader, test_loader = get_dataloader(
        X, Y, batch_size, batch_size2, seed, shuffle=True
    )

    # Define model
    num_classes = dataset_info["#_class"]
    channels = X.shape[1]
    print(channels)

    n_T = 1000  # Steps in the diffusion process, Timesteps for the Betas
    ddpm_dim = 128
    encoder_dim = 256
    fc_dim = 512
    num_groups = 1

    ddpm_model = ConditionalUNet(
        in_channels=channels, n_feat=ddpm_dim, num_groups=num_groups
    ).to(device)
    ddpm = DDPM(nn_model=ddpm_model, betas=(1e-6, 1e-2), n_T=n_T, device=device).to(
        device
    )  # Betas tell us how much noise we want to add. It starts at 1.e-6 at increases up to 1e-2
    encoder = Encoder(in_channels=channels, dim=encoder_dim, num_groups=num_groups).to(
        device
    )
    decoder = Decoder(
        in_channels=channels,
        n_feat=ddpm_dim,
        encoder_dim=encoder_dim,
        num_groups=num_groups,
    ).to(device)
    fc = LinearClassifier(encoder_dim, fc_dim, emb_dim=num_classes).to(device)
    diffe = DiffE(encoder, decoder, fc).to(device)

    print("ddpm size: ", sum(p.numel() for p in ddpm.parameters()))
    print("encoder size: ", sum(p.numel() for p in encoder.parameters()))
    print("decoder size: ", sum(p.numel() for p in decoder.parameters()))
    print("fc size: ", sum(p.numel() for p in fc.parameters()))

    # Criterion
    criterion = nn.L1Loss()
    criterion_class = nn.MSELoss()

    # Define optimizer
    base_lr, lr = 9e-5, 1.5e-3
    optim1 = optim.RMSprop(ddpm.parameters(), lr=base_lr)
    optim2 = optim.RMSprop(diffe.parameters(), lr=base_lr)

    # EMAs
    fc_ema = EMA(
        diffe.fc,
        beta=0.95,
        update_after_step=100,
        update_every=10,
    )

    step_size = 150
    scheduler1 = optim.lr_scheduler.CyclicLR(
        optimizer=optim1,
        base_lr=base_lr,
        max_lr=lr,
        step_size_up=step_size,
        mode="exp_range",
        cycle_momentum=False,
        gamma=0.9998,
    )
    scheduler2 = optim.lr_scheduler.CyclicLR(
        optimizer=optim2,
        base_lr=base_lr,
        max_lr=lr,
        step_size_up=step_size,
        mode="exp_range",
        cycle_momentum=False,
        gamma=0.9998,
    )
    # Train & Evaluate
    num_epochs = dataset_info["total_trials"]
    test_period = 1
    start_test = test_period
    alpha = 0.1

    best_acc = 0
    best_f1 = 0
    best_recall = 0
    best_precision = 0
    best_auc = 0

    with tqdm(
        total=num_epochs, desc=f"Method ALL - Processing subject_id {subject_id}"
    ) as pbar:
        for epoch in range(num_epochs):
            ddpm.train()
            diffe.train()

            # ***************************** Train *****************************
            for x, y in train_loader:
                x, y = x.to(device).float(), y.type(torch.LongTensor).to(device)
                y_cat = (
                    F.one_hot(y, num_classes=num_classes)
                    .type(torch.FloatTensor)
                    .to(device)
                )
                # Train DDPM
                optim1.zero_grad()

                x_hat, down, up, noise, t = ddpm(x)

                loss_ddpm = F.l1_loss(x_hat, x, reduction="none")
                loss_ddpm.mean().backward()
                optim1.step()
                ddpm_out = x_hat, down, up, t

                # Train Diff-E
                optim2.zero_grad()
                decoder_out, fc_out = diffe(x, ddpm_out)

                loss_gap = criterion(decoder_out, loss_ddpm.detach())
                loss_c = criterion_class(fc_out, y_cat)
                loss = loss_gap + alpha * loss_c
                loss.backward()
                optim2.step()

                # Optimizer scheduler step
                scheduler1.step()
                scheduler2.step()

                # EMA update
                fc_ema.update()

            # ***************************** Test *****************************
            with torch.no_grad():
                if epoch > start_test:
                    test_period = 1
                if epoch % test_period == 0:
                    ddpm.eval()
                    diffe.eval()

                    metrics_test = evaluate(
                        diffe.encoder, fc_ema, test_loader, device, num_classes
                    )

                    acc = metrics_test["accuracy"]
                    f1 = metrics_test["f1"]
                    recall = metrics_test["recall"]
                    precision = metrics_test["precision"]
                    auc = metrics_test["auc"]

                    best_acc_bool = acc > best_acc
                    best_f1_bool = f1 > best_f1
                    best_recall_bool = recall > best_recall
                    best_precision_bool = precision > best_precision
                    best_auc_bool = auc > best_auc

                    if best_acc_bool:
                        print("Saving model...")
                        best_acc = acc
                        torch.save(
                            diffe.state_dict(),
                            model_path,
                        )
                    if best_f1_bool:
                        best_f1 = f1
                    if best_recall_bool:
                        best_recall = recall
                    if best_precision_bool:
                        best_precision = precision
                    if best_auc_bool:
                        best_auc = auc

                    # print("Subject: {0}".format(subject_id))
                    # # print("ddpm test loss: {0:.4f}".format(t_test_loss_ddpm/len(test_generator)))
                    # # print("encoder test loss: {0:.4f}".format(t_test_loss_ed/len(test_generator)))
                    # print("accuracy:  {0:.2f}%".format(acc*100), "best: {0:.2f}%".format(best_acc*100))
                    # print("f1-score:  {0:.2f}%".format(f1*100), "best: {0:.2f}%".format(best_f1*100))
                    # print("recall:    {0:.2f}%".format(recall*100), "best: {0:.2f}%".format(best_recall*100))
                    # print("precision: {0:.2f}%".format(precision*100), "best: {0:.2f}%".format(best_precision*100))
                    # print("auc:       {0:.2f}%".format(auc*100), "best: {0:.2f}%".format(best_auc*100))
                    # writer.add_scalar(f"EEGNet/Accuracy/subject_{subject_id}", acc*100, epoch)
                    # writer.add_scalar(f"EEGNet/F1-score/subject_{subject_id}", f1*100, epoch)
                    # writer.add_scalar(f"EEGNet/Recall/subject_{subject_id}", recall*100, epoch)
                    # writer.add_scalar(f"EEGNet/Precision/subject_{subject_id}", precision*100, epoch)
                    # writer.add_scalar(f"EEGNet/AUC/subject_{subject_id}", auc*100, epoch)

                    # if best_acc_bool or best_f1_bool or best_recall_bool or best_precision_bool or best_auc_bool:
                    #     performance = {'subject_id': subject_id,
                    #                 'epoch': epoch,
                    #                 'accuracy': best_acc*100,
                    #                 'f1_score': best_f1*100,
                    #                 'recall': best_recall*100,
                    #                 'precision': best_precision*100,
                    #                 'auc': best_auc*100
                    #                 }
                    #     with open(output_file, 'a') as f:
                    #         f.write(f"{performance['subject_id']}, {performance['epoch']}, {performance['accuracy']}, {performance['f1_score']}, {performance['recall']}, {performance['precision']}, {performance['auc']}\n")
                    description = f"Best accuracy: {best_acc*100:.2f}%"
                    pbar.set_description(
                        f"Method ALL - Processing subject_id {subject_id} - {description}"
                    )
            pbar.update(1)
    return best_acc
