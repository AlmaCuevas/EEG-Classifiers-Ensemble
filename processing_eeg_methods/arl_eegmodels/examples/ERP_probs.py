# I tried doing the accuracy check, but accuracy is supposed to be a predictive measure for how good the model is.
# Well it's not. In EEGNet the accuracy can be 0.2 but with the test it'll be 0.42.
# Or the accuracy can be 0.3 and the test 0.3 too.
# Everyting was well done as far as I know, as since there was nothing to do, I simply removed the accuracy to have more data to train.
# I also tried doing voting system, but its simply bad.

import numpy as np

# EEGNet-specific imports
from arl_eegmodels.EEGModels import EEGNet, ShallowConvNet, DeepConvNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
import time

from data_utils import train_test_val_split
from share import datasets_basic_infos, ROOT_VOTING_SYSTEM_PATH
from data_loaders import load_data_labels_based_on_dataset

from pathlib import Path


def ERP_train(dataset_name, data, label, dataset_info):
    X_train, X_test, X_validate, Y_train, Y_test, Y_validate = train_test_val_split(
        dataX=data, dataY=label, valid_flag=True)

    kernels, chans, samples = 1, dataset_info['#_channels'], dataset_info['samples']

    ############################# EEGNet portion ##################################

    # convert labels to one-hot encodings.
    Y_train = np_utils.to_categorical(Y_train)
    Y_validate = np_utils.to_categorical(Y_validate)

    # convert data to NHWC (trials, channels, samples, kernels) format. Data
    # contains 60 channels and 151 time-points. Set the number of kernels to 1.
    X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
    X_validate = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')

    # configure the EEGNet-8,2,16 model with kernel length of 32 samples (other
    # model configurations may do better, but this is a good starting point)
    model = DeepConvNet(nb_classes=4, Chans=chans, Samples=samples,
                   dropoutRate=0.5)

    # compile the model and set the optimizers
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    # set a valid path for your system to record model checkpoints
    checkpointer = ModelCheckpoint(filepath=f'{ROOT_VOTING_SYSTEM_PATH}/processing_eeg_methods/BigProject/ERP_model_{dataset_name}.h5', verbose=1,
                                   save_best_only=True)

    ###############################################################################
    # if the classification task was imbalanced (significantly more trials in one
    # class versus the others) you can assign a weight to each class during
    # optimization to balance it out. This data is approximately balanced so we
    # don't need to do this, but is shown here for illustration/completeness.
    ###############################################################################

    # the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
    # the weights all to be 1
    class_weights = {0: 1, 1: 1, 2: 1, 3: 1}

    ################################################################################
    # fit the model. Due to very small sample sizes this can get
    # pretty noisy run-to-run, but most runs should be comparable to xDAWN +
    # Riemannian geometry classification (below)
    ################################################################################
    model.fit(X_train, Y_train, batch_size=16, epochs=X_train.shape[0],
                            verbose=2, validation_data=(X_validate, Y_validate),
                            callbacks=[checkpointer], class_weight=class_weights)

    # load optimal weights
    model.load_weights(f'{ROOT_VOTING_SYSTEM_PATH}/processing_eeg_methods/BigProject/ERP_model_{dataset_name}.h5')
    return model


def ERP_test(model, X_test):
    probs = model.predict(X_test)
    return probs


if __name__ == '__main__':
    # Manual Inputs
    subject_id = 1  # Only two things I should be able to change
    dataset_name = 'aguilera_gamified'  # Only two things I should be able to change
    array_format = True

    # Folders and paths
    dataset_foldername = dataset_name + '_dataset'
    computer_root_path = ROOT_VOTING_SYSTEM_PATH + "/Datasets/"
    data_path = computer_root_path + dataset_foldername
    dataset_info = datasets_basic_infos[dataset_name]

    # while the default tensorflow ordering is 'channels_last' we set it here
    # to be explicit in case if the user has changed the default ordering
    K.set_image_data_format('channels_last')

    data, label = load_data_labels_based_on_dataset(dataset_info, subject_id, data_path)
    target_names = dataset_info['target_names']

    data_train, data_test, _, labels_train, labels_test, _ = train_test_val_split(
        dataX=data, dataY=label, valid_flag=False)

    kernels, chans, samples = 1, dataset_info['#_channels'], dataset_info['samples']

    labels_test = np_utils.to_categorical(labels_test)
    data_test = data_test.reshape(data_test.shape[0], chans, samples, kernels)

    print("******************************** Training ********************************")
    start = time.time()
    model = ERP_train(dataset_name, data_train, labels_train, dataset_info)
    end = time.time()
    print("Training time: ", end - start)

    print("******************************** Test ********************************")
    pred_list = []
    for data_chosen, trial_chosen in zip(data_test, labels_test):
        start = time.time()
        probs = ERP_test(model, np.asarray([data_chosen]))
        preds = probs.argmax(axis=-1)

        end = time.time()
        print("One epoch, testing time: ", end - start)
        print("Answer: ", preds)
        print("Real: ", trial_chosen.argmax(axis = -1))
        pred_list.append(preds)

    test_voting_acc = np.mean(pred_list == labels_test)
    print("Final Acc voting: ", test_voting_acc)

