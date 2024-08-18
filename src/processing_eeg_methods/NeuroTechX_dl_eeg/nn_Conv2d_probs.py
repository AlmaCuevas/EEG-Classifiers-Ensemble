# Not good, at least for BrainCommand
import numpy as np
import torch
import torch.nn.functional as F
from braindecode.datautil.iterators import get_balanced_batches
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.torch_ext.util import np_to_var, set_random_seeds, var_to_np
from numpy.random import RandomState
from share import ROOT_VOTING_SYSTEM_PATH
from sklearn.model_selection import train_test_split
from torch import nn, optim


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10% every 30 epochs"""
    lr = 0.00006 * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def nn_Conv2d_train(data, label, dataset_info, subject_id) -> tuple[str, float]:
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
    model = ShallowFBCSPNet(
        in_chans=train_set.X.shape[1],
        n_classes=n_classes,
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
        final_conv_length="auto",
    ).create_network()
    if cuda:
        model.cuda()

    for param in model.conv_classifier.parameters():
        param.requires_grad = False

    model.conv_classifier = nn.Conv2d(5, 2, (8, 1), bias=True).cuda()

    optimizer = optim.Adam(model.conv_classifier.parameters(), lr=0.00006)

    for i_epoch in range(nb_epoch):
        i_trials_in_batch = get_balanced_batches(
            len(train_set.X), rng, shuffle=True, batch_size=32
        )

        adjust_learning_rate(optimizer, i_epoch)

        # Set model to training mode
        model.train()

        for i_trials in i_trials_in_batch:
            # Have to add empty fourth dimension to X
            batch_X = train_set.X[i_trials][:, :, :, None]
            batch_y = train_set.y[i_trials]
            net_in = np_to_var(batch_X)
            if cuda:
                net_in = net_in.cuda()
            net_target = np_to_var(batch_y)
            if cuda:
                net_target = net_target.cuda()
            # Remove gradients of last backward pass from all parameters
            optimizer.zero_grad()
            # Compute outputs of the network
            outputs = model(net_in)
            # Compute the loss
            loss = F.nll_loss(outputs, net_target)
            # Do the backpropagation
            loss.backward()
            # Update parameters with the optimizer
            optimizer.step()

        # Print some statistics each epoch
        model.eval()
        print("Epoch {:d}".format(i_epoch))

        sets = {"Train": 0, "Test": 1}
        for setname, dataset in (("Train", train_set), ("Test", test_set)):
            i_trials_in_batch = get_balanced_batches(
                len(dataset.X), rng, batch_size=32, shuffle=False
            )
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

            print("{:6s} Loss: {:.5f}".format(setname, float(var_to_np(loss))))
            loss_rec[i_epoch, sets[setname]] = var_to_np(loss)

            predicted_labels = np.argmax(var_to_np(outputs), axis=1)
            accuracy = np.mean(dataset.y == predicted_labels)
            print("{:6s} Accuracy: {:.1f}%".format(setname, accuracy * 100))
            accuracy_rec[i_epoch, sets[setname]] = accuracy

    # save/load only the model parameters(preferred solution)
    model_path: str = (
        f'{ROOT_VOTING_SYSTEM_PATH}/Results/{dataset_info["dataset_name"]}/nn_Conv2d/nn_Conv2d_{dataset_info["dataset_name"]}_{subject_id}.pth'
    )
    torch.save(model.state_dict(), model_path)

    acc = accuracy_rec[:, 1].mean()
    return acc


def nn_Conv2d_test(subject_id: int, data, dataset_info: dict):
    model_path: str = (
        f'{ROOT_VOTING_SYSTEM_PATH}/Results/{dataset_info["dataset_name"]}/nn_Conv2d/nn_Conv2d_{dataset_info["dataset_name"]}_{subject_id}.pth'
    )

    test_set = SignalAndTarget(
        data, y=[0]
    )  # y=0 just to not leave it empty, but it is not used.

    rng = RandomState(None)
    n_classes = 2
    # final_conv_length = auto ensures we only get a single output in the time dimension
    model = ShallowFBCSPNet(
        in_chans=test_set.X.shape[1],
        n_classes=n_classes,
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
        final_conv_length="auto",
    ).create_network()
    cuda = torch.cuda.is_available()
    set_random_seeds(seed=20180505, cuda=cuda)
    if cuda:
        model.cuda()

    model.load_state_dict(torch.load(model_path))

    # Print some statistics each epoch
    model.eval()

    dataset = test_set

    i_trials_in_batch = get_balanced_batches(
        len(dataset.X), rng, batch_size=32, shuffle=False
    )
    outputs = []
    for i_trials in i_trials_in_batch:
        batch_X = dataset.X[i_trials][:, :, :, None]

        net_in = np_to_var(batch_X)
        if cuda:
            net_in = net_in.cuda()
        output = var_to_np(model(net_in))
        outputs.append(output)
    outputs = np_to_var(np.concatenate(outputs))
    return var_to_np(outputs)
