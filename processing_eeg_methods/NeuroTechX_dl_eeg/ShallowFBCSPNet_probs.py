import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split

from data_utils import create_folder, convert_into_binary
from share import datasets_basic_infos, ROOT_VOTING_SYSTEM_PATH
from data_loaders import load_data_labels_based_on_dataset
import time
import torch
from os import path
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from torch import nn
from braindecode.torch_ext.util import set_random_seeds
from torch import optim
from braindecode.torch_ext.util import np_to_var, var_to_np
from braindecode.datautil.iterators import get_balanced_batches
import torch.nn.functional as F
from numpy.random import RandomState

threshold_for_bug = 0.00000001  # could be any value, ex numpy.min
accelerator = 'cu80' if path.exists('/opt/bin/nvidia-smi') else 'cpu'

def ShallowFBCSPNet_train(data, label) -> tuple[str, float]:
    rng = RandomState(None)
    # rng = RandomState((2017,6,30))

    nb_epoch = 160
    loss_rec = np.zeros((nb_epoch, 2))
    accuracy_rec = np.zeros((nb_epoch, 2))

    cuda = torch.cuda.is_available()
    set_random_seeds(seed=20180505, cuda=cuda)
    n_classes = 2

    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)

    train_set = SignalAndTarget(x_train, y=y_train)
    test_set = SignalAndTarget(x_test, y=y_test)

    # final_conv_length = auto ensures we only get a single output in the time dimension
    model = ShallowFBCSPNet(in_chans=train_set.X.shape[1], n_classes=n_classes,
                            input_time_length=train_set.X.shape[2],

                            n_filters_time=10,
                            filter_time_length=75,
                            n_filters_spat=5,
                            pool_time_length=60,
                            pool_time_stride=30,

                            # n_filters_time=10,
                            # filter_time_length=90,
                            # n_filters_spat=1,
                            # pool_time_length=45,
                            # pool_time_stride=15,

                            final_conv_length='auto'
                            ).create_network()
    if cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters())

    for i_epoch in range(nb_epoch):

        # get a set of balanced batches
        i_trials_in_batch = get_balanced_batches(len(train_set.X), rng, shuffle=True,
                                                 batch_size=32)

        # Set model to training mode
        model.train()

        # go through all batches
        for i_trials in i_trials_in_batch:
            # Have to add empty fourth dimension to X
            batch_X = train_set.X[i_trials][:, :, :, None]
            batch_y = train_set.y[i_trials]
            net_in = np_to_var(batch_X)
            if cuda:
                net_in = net_in.cuda()
            net_target = np_to_var(batch_y).type(torch.LongTensor)
            if cuda:
                net_target = net_target.cuda()
            # Remove gradients of last backward pass from all parameters
            optimizer.zero_grad()
            # Compute outputs of the network
            outputs = model(net_in)
            print(net_target)
            # Compute the loss
            loss = F.nll_loss(outputs, net_target)
            # Do the backpropagation
            loss.backward()
            # Update parameters with the optimizer
            optimizer.step()

        # Print some statistics each epoch
        model.eval()
        print("Epoch {:d}".format(i_epoch))

        sets = {'Train': 0, 'Test': 1}
        for setname, dataset in (('Train', train_set), ('Test', test_set)):
            i_trials_in_batch = get_balanced_batches(len(dataset.X), rng, batch_size=32, shuffle=False)
            outputs = []
            net_targets = []
            for i_trials in i_trials_in_batch:
                batch_X = dataset.X[i_trials][:, :, :, None]
                batch_y = dataset.y[i_trials]

                net_in = np_to_var(batch_X)
                if cuda:
                    net_in = net_in.cuda()
                net_target = np_to_var(batch_y)
                if cuda:
                    net_target = net_target.cuda()
                net_target = var_to_np(net_target)
                output = var_to_np(model(net_in))
                outputs.append(output)
                net_targets.append(net_target)
            net_targets = np_to_var(np.concatenate(net_targets))
            outputs = np_to_var(np.concatenate(outputs))
            loss = F.nll_loss(outputs, net_targets)

            print("{:6s} Loss: {:.5f}".format(
                setname, float(var_to_np(loss))))
            loss_rec[i_epoch, sets[setname]] = var_to_np(loss)

            predicted_labels = np.argmax(var_to_np(outputs), axis=1)
            accuracy = np.mean(dataset.y == predicted_labels)
            print("{:6s} Accuracy: {:.1f}%".format(
                setname, accuracy * 100))
            accuracy_rec[i_epoch, sets[setname]] = accuracy


    # save/load only the model parameters(prefered solution)
    model_path: str = f'{ROOT_VOTING_SYSTEM_PATH}/Results/{dataset_info["dataset_name"]}/ShallowFBCSPNet/ShallowFBCSPNet_{dataset_info["dataset_name"]}_{subject_id}.pth'
    torch.save(model.state_dict(), model_path)

    acc = accuracy_rec[:, 1].mean()
    return acc

def ShallowFBCSPNet_test(subject_id: int, data, dataset_info: dict):
    model_path: str = f'{ROOT_VOTING_SYSTEM_PATH}/Results/{dataset_info["dataset_name"]}/ShallowFBCSPNet/ShallowFBCSPNet_{dataset_info["dataset_name"]}_{subject_id}.pth'

    test_set = SignalAndTarget(data, y=[0]) # y=0 just to not leave it empty, but it is not used.

    rng = RandomState(None)
    n_classes = 2
    # final_conv_length = auto ensures we only get a single output in the time dimension
    model = ShallowFBCSPNet(in_chans=test_set.X.shape[1], n_classes=n_classes,
                            input_time_length=test_set.X.shape[2],

                            n_filters_time=10,
                            filter_time_length=75,
                            n_filters_spat=5,
                            pool_time_length=60,
                            pool_time_stride=30,

                            # n_filters_time=10,
                            # filter_time_length=90,
                            # n_filters_spat=1,
                            # pool_time_length=45,
                            # pool_time_stride=15,

                            final_conv_length='auto'
                            ).create_network()
    cuda = torch.cuda.is_available()
    set_random_seeds(seed=20180505, cuda=cuda)
    if cuda:
        model.cuda()

    model.load_state_dict(torch.load(model_path))

    # Print some statistics each epoch
    model.eval()

    dataset = test_set

    i_trials_in_batch = get_balanced_batches(len(dataset.X), rng, batch_size=32, shuffle=False)
    outputs = []
    for i_trials in i_trials_in_batch:
        batch_X = dataset.X[i_trials][:,:,:,None]

        net_in = np_to_var(batch_X)
        if cuda:
            net_in = net_in.cuda()
        output = var_to_np(model(net_in))
        outputs.append(output)
    outputs = np_to_var(np.concatenate(outputs))
    return var_to_np(outputs)


if __name__ == "__main__":
    # Manual Inputs
    datasets = ['braincommand']#, 'aguilera_traditional', 'aguilera_gamified', 'torres']
    for dataset_name in datasets:
        chosen_numbered_label = 3
        version_name = str(chosen_numbered_label) # To keep track what the output processing alteration went through

        # Folders and paths
        dataset_foldername = dataset_name + "_dataset"
        computer_root_path = ROOT_VOTING_SYSTEM_PATH + "/Datasets/"
        data_path = computer_root_path + dataset_foldername
        print(data_path)
        # Initialize
        processing_name: str = 'ShallowFBCSPNet'
        if dataset_name not in datasets_basic_infos:
            raise Exception(
                f"Not supported dataset named '{dataset_name}', choose from the following: braincommand, aguilera_traditional, aguilera_gamified, nieto, coretto or torres."
            )
        dataset_info: dict = datasets_basic_infos[dataset_name]

        mean_accuracy_per_subject: list = []
        results_df = pd.DataFrame()

        for subject_id in range(
            29,30
        ):
            create_folder(dataset_name, processing_name)
            print(subject_id)
            with open(
                f"{ROOT_VOTING_SYSTEM_PATH}/Results/{version_name}_{processing_name}_{dataset_name}.txt",
                "a",
            ) as f:
                f.write(f"Subject: {subject_id}\n\n")
            epochs, _, _ = load_data_labels_based_on_dataset(dataset_info, subject_id, data_path)

            data = (epochs.get_data() * 1e6).astype(np.float32)
            labels = epochs.events[:, 2].astype(np.int64)
            labels = convert_into_binary(labels, chosen_numbered_label=chosen_numbered_label)

            data[data < threshold_for_bug] = threshold_for_bug # To avoid the error "SVD did not convergence"
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
                accuracy = ShallowFBCSPNet_train(data[train], labels[train])
                training_time.append(time.time() - start)
                with open(
                    f"{ROOT_VOTING_SYSTEM_PATH}/Results/{version_name}_{processing_name}_{dataset_name}.txt",
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
                    array = ShallowFBCSPNet_test(subject_id, np.asarray([data[epoch_number]]), dataset_info)
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
                    f"{ROOT_VOTING_SYSTEM_PATH}/Results/{version_name}_{processing_name}_{dataset_name}.txt",
                    "a",
                ) as f:
                    f.write(f"Prediction: {pred_list}\n")
                    f.write(f"Real label:{labels[test]}\n")
                    f.write(f"Mean accuracy in KFold: {acc}\n")
                print("Mean accuracy in KFold: ", acc)
            mean_acc_over_cv = np.mean(acc_over_cv)

            with open(
                f"{ROOT_VOTING_SYSTEM_PATH}/Results/{version_name}_{processing_name}_{dataset_name}.txt",
                "a",
            ) as f:
                f.write(f"Final acc: {mean_acc_over_cv}\n\n\n\n")
            print(f"Final acc: {mean_acc_over_cv}")

            temp = pd.DataFrame({'Methods': [processing_name] * len(acc_over_cv), 'Subject ID': [subject_id] * len(acc_over_cv),
                                 'Version': [version_name] * len(acc_over_cv), 'Training Accuracy': [accuracy] * len(acc_over_cv), 'Training Time': training_time,
                                 'Testing Accuracy': acc_over_cv, 'Testing Time': testing_time_over_cv}) # The idea is that the most famous one is the one I use for this dataset
            results_df = pd.concat([results_df, temp])

        results_df.to_csv(
            f"{ROOT_VOTING_SYSTEM_PATH}/Results/{version_name}_{processing_name}_{dataset_name}.csv")

    print("Congrats! The processing methods are done processing.")
