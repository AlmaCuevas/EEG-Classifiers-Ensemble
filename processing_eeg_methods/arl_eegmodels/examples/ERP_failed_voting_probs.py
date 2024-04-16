# I tried doing the accuracy check, but accuracy is supposed to be a predictive measure for how good the model is.
# Well it's not. In EEGNet the accuracy can be 0.2 but with the test it'll be 0.42.
# Or the accuracy can be 0.3 and the test 0.3 too.
# Everyting was well done as far as I know, as since there was nothing to do, I simply removed the accuracy to have more data to train.
# I also tried doing voting system, but its simply bad.

import numpy as np

# EEGNet-specific imports
from EEGModels import EEGNet, ShallowConvNet, DeepConvNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
import time

from Code.data_utils import train_test_val_split
from share import datasets_basic_infos
from data_loaders import load_data_labels_based_on_dataset

def ERP_train(data, label, dataset_info):
    X_train, X_test, X_validate, Y_train, Y_test, Y_validate = train_test_val_split(
        dataX=data, dataY=label, valid_flag=True)

    kernels, chans, samples = 1, dataset_info['#_channels'], dataset_info['samples']

    ############################# EEGNet portion ##################################

    # convert labels to one-hot encodings.
    Y_train = np_utils.to_categorical(Y_train)
    Y_validate = np_utils.to_categorical(Y_validate)
    Y_test = np_utils.to_categorical(Y_test)

    # convert data to NHWC (trials, channels, samples, kernels) format. Data
    # contains 60 channels and 151 time-points. Set the number of kernels to 1.
    X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
    X_validate = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
    X_test = X_test.reshape(X_test.shape[0], chans, samples, kernels)

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # configure the EEGNet-8,2,16 model with kernel length of 32 samples (other
    # model configurations may do better, but this is a good starting point)
    model = EEGNet(nb_classes=4, Chans=chans, Samples=samples,
                   dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16,
                   dropoutType='Dropout')

    # compile the model and set the optimizers
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    # count number of parameters in the model
    numParams = model.count_params()

    # set a valid path for your system to record model checkpoints
    checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
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
    model.load_weights('/tmp/checkpoint.h5')

    ###############################################################################
    # can alternatively used the weights provided in the repo. If so it should get
    # you 93% accuracy. Change the WEIGHTS_PATH variable to wherever it is on your
    # system.
    ###############################################################################

    # WEIGHTS_PATH = /path/to/EEGNet-8-2-weights.h5
    # model.load_weights(WEIGHTS_PATH)

    ###############################################################################
    # make prediction on test set.
    ###############################################################################
    probs = model.predict(X_test)
    preds = probs.argmax(axis=-1)
    acc = np.mean(preds == Y_test.argmax(axis=-1))
    print(preds)
    print(Y_test.argmax(axis=-1))
    print("Classification accuracy: %f " % (acc))
    return model, acc


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
    # computer_root_path = "/Users/rosit/Documents/MCC/voting_system_platform/Datasets/"  # OMEN
    computer_root_path = "//"  # MAC
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
    model, acc = ERP_train(data_train, labels_train, dataset_info)
    end = time.time()
    print("Training time: ", end - start)
    print("Accuracy: ", acc)
    print("******************************** Training ********************************")
    model_2, acc_2 = ERP_train(data_train, labels_train, dataset_info)


    print("******************************** Training ********************************")
    model_3, acc_3 = ERP_train(data_train, labels_train, dataset_info)

    print("******************************** Training ********************************")
    model_4, acc_4 = ERP_train(data_train, labels_train, dataset_info)

    print("******************************** Test ********************************")
    pred_list = []
    pred_list_1 = []
    pred_list_2 = []
    pred_list_3 = []
    pred_list_4 = []
    for data_chosen, trial_chosen in zip(data_test, labels_test):
        start = time.time()
        probs = ERP_test(model, np.asarray([data_chosen]))
        probs_2 = ERP_test(model_2, np.asarray([data_chosen]))
        probs_3 = ERP_test(model_3, np.asarray([data_chosen]))
        probs_4 = ERP_test(model_4, np.asarray([data_chosen]))

        probs_list = [np.multiply(probs, acc),
                      np.multiply(probs_2, acc_2),
                      np.multiply(probs_3, acc_3),
                      np.multiply(probs_4, acc_4),
                      ]
        probs_voting = np.nanmean(probs_list, axis=0)

        preds = probs_voting.argmax(axis=-1)
        preds_1 = probs.argmax(axis=-1)
        preds_2 = probs_2.argmax(axis=-1)
        preds_3 = probs_3.argmax(axis=-1)
        preds_4 = probs_4.argmax(axis=-1)

        end = time.time()
        print("One epoch, testing time: ", end - start)
        print("Answer: ", preds)
        print("Real: ", trial_chosen.argmax(axis = -1))
        pred_list.append(preds)
        pred_list_1.append(preds_1)
        pred_list_2.append(preds_2)
        pred_list_3.append(preds_3)
        pred_list_4.append(preds_4)

    test_voting_acc = np.mean(pred_list == labels_test)
    test_1_acc = np.mean(pred_list_1 == labels_test)
    test_2_acc = np.mean(pred_list_2 == labels_test)
    test_3_acc = np.mean(pred_list_3 == labels_test)
    test_4_acc = np.mean(pred_list_4 == labels_test)
    print("Final Acc voting: ", test_voting_acc)
    print("Final Acc 1: ", test_1_acc)
    print("Final Acc 2: ", test_2_acc)
    print("Final Acc 3: ", test_3_acc)
    print("Final Acc 4: ", test_4_acc)

    print("Acc 1: ", acc)
    print("Acc 2: ", acc_2)
    print("Acc 3: ", acc_3)
    print("Acc 4: ", acc_4)

