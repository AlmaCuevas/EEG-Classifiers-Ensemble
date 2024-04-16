from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.layers import Conv1D, Activation, Flatten
from keras.utils import to_categorical
import numpy as np

from data_loaders import load_data_labels_based_on_dataset
from share import datasets_basic_infos, ROOT_VOTING_SYSTEM_PATH
from data_utils import train_test_val_split
import time

def CNN_LSTM_train(data, labels, num_classes: int):
    # substract data from list
    X_train, X_test, X_val, y_train, y_test, y_val = train_test_val_split(dataX=data, dataY=labels, valid_flag=True)

    # get data dimension
    N_train, T_train, C_train = X_train.shape
    N_val, T_val, C_val = X_val.shape
    N_test, T_test, C_test = X_test.shape

    # add dummy zeros for y classification, convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # construct X_total and y_total based on sub-sampling of X_train and y_train

    # take sub-sampling on the time sequence to reduce dimension for RNN
    sampling = 1

    X_train = X_train.reshape(N_train, int(T_train / sampling), sampling, C_train)[:, :, 0, :]
    X_val = X_val.reshape(N_val, int(T_val / sampling), sampling, C_val)[:, :, 0, :]
    X_test = X_test.reshape(N_test, int(T_test / sampling), sampling, C_test)[:, :, 0, :]

    # get new data dimension
    N_train, T_train, C_train = X_train.shape
    N_test, T_test, C_test = X_test.shape

    # Expected input batch shape: (batch_size, timesteps, data_dim)
    # Note that we have to provide the full batch_input_shape since the network is stateful.
    # the sample of index i in batch k is the follow-up for the sample i in batch k-1.

    # perhaps should try masking layer

    data_dim = C_train
    timesteps = T_train
    batch_size = y_train.shape[0]
    num_epoch = y_train.shape[0]

    # make a sequential model
    model = Sequential()

    # add 1-layer cnn
    model.add(Conv1D(40, kernel_size=20, strides=4,
                     input_shape=(timesteps, data_dim)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    # model.add(MaxPooling1D(pool_size=4, strides=4))

    # add 2-layer lstm
    model.add(LSTM(30, return_sequences=True, stateful=False))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(LSTM(20, return_sequences=True, stateful=False))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    # set loss function and optimizer
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # train the data with validation
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=num_epoch,
                        shuffle=False,
                        validation_data=(X_val, y_val))

    # test set
    results = model.evaluate(X_test, y_test, batch_size=N_test)
    print("test loss, test acc:", results)
    acc = results[1]
    return model, acc

def CNN_LSTM_test(model, trial_data):
    N_test, T_test, C_test = trial_data.shape
    sampling = 1
    trial_data = trial_data.reshape(N_test,int(T_test/sampling), sampling, C_test)[:,:,0,:]
    output_array = model.predict(trial_data)
    return output_array

if __name__ == '__main__':
    # Manual Inputs
    subject_id = 2  # Only two things I should be able to change
    dataset_name = 'aguilera_gamified'  # Only two things I should be able to change
    ONLY_GLOBAL_MODEL = True

    array_format = True

    # Folders and paths
    dataset_foldername = dataset_name + '_dataset'
    computer_root_path = f"{ROOT_VOTING_SYSTEM_PATH}/Datasets/"
    data_path = computer_root_path + dataset_foldername
    dataset_info = datasets_basic_infos[dataset_name]

    data, label = load_data_labels_based_on_dataset(dataset_info, subject_id, data_path)
    data_train, data_test, _, labels_train, labels_test, _ = train_test_val_split(
        dataX=data, dataY=label, valid_flag=False)
    target_names = dataset_info['target_names']

    print("******************************** Training ********************************")
    start = time.time()
    model, acc = CNN_LSTM_train(data_train, labels_train, dataset_info['#_class'])
    end = time.time()
    print("Training time: ", end - start)

    print("******************************** Test ********************************")
    pred_list=[]
    for data_chosen, labels_chosen in zip(data_test, labels_test):
        start = time.time()
        array = CNN_LSTM_test(model, np.asarray([data_chosen]))
        end = time.time()
        print("One epoch, testing time: ", end - start)
        print(target_names)
        print("Probability: " , data_chosen) # We select the last one, the last epoch which is the current one.
        print("Real: ", labels_chosen)
        pred=np.argmax(array)
        pred_list.append(pred)
        print("Prediction: ", pred)
    acc = np.mean(pred_list == labels_test)
    print("Final Acc: ", acc)