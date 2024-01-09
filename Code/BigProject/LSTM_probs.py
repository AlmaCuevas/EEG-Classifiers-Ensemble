from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Flatten, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.fftpack import dct, idct
from scipy import signal

from Code.data_loaders import load_data_labels_based_on_dataset
from share import datasets_basic_infos
from Code.data_utils import train_test_val_split
import time

def LSTM_train(dataset_name, data, labels):
    # substract data from list
    X_train, X_test, _, y_train, y_test, _ = train_test_val_split(dataX=data, dataY=labels, valid_flag=False)

    # get data dimension
    N_train, T_train, C_train = X_train.shape
    N_test, T_test, C_test = X_test.shape

    # add dummy zeros for y classification
    lb = preprocessing.LabelBinarizer()
    lb.fit([0, 1, 2, 3, ])
    y_train = lb.transform(y_train)
    y_test = lb.transform(y_test)

    # Filtering through FFT(discrete cosine transform)

    def filter(x, low=0, high=1, plot=False):
        N = x.shape[0]
        t = np.linspace(0, N, N)
        y = dct(x, norm='ortho')
        window = np.zeros(N)
        window[int(low * N):int(high * N)] = 1
        yr = idct(y * window, norm='ortho')
        sum(abs(x - yr) ** 2) / sum(abs(x) ** 2)
        if plot:
            plt.plot(t, x, '-b')
            plt.plot(t, yr, 'r')
        return x

    # Filter band
    low_freq = 0.02
    high_freq = 0.4

    for i in np.arange(N_train):
        for j in np.arange(C_train):
            X_train[i, :, j] = filter(X_train[i, :, j], low_freq, high_freq)

    for i in np.arange(N_test):
        for j in np.arange(C_test):
            X_test[i, :, j] = filter(X_test[i, :, j], low_freq, high_freq)

    # Downsampling in time through FFT

    t_sample = 50
    X_train_sub = signal.resample(X_train, t_sample, axis=1)
    X_test_sub = signal.resample(X_test, t_sample, axis=1)

    data_dim = C_train
    seq_split = 1  # Set to one when using FFT to down sample
    seq_len = int(X_train_sub.shape[1] * seq_split)
    timesteps = seq_len
    num_classes = 4
    batch_size = 200
    num_epoch = 100

    model = Sequential()
    # 1
    model.add(LSTM(200, return_sequences=True, stateful=False,
                   recurrent_dropout=0.6, dropout=0.6, input_shape=(timesteps, data_dim)))

    # 2
    model.add(LSTM(100, return_sequences=True, stateful=False,
                   recurrent_dropout=0.5, dropout=0.5))

    # 3
    model.add(LSTM(50, return_sequences=True, stateful=False,
                   recurrent_dropout=0.4, dropout=0.4))
    model.add(Flatten())

    # 4
    model.add(Dense(100))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # 5
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # define early stopping callback
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=30, mode='auto')

    # saves the model weights after each epoch if the validation loss decreased
    checkpointer = ModelCheckpoint(
        filepath=f'/Users/almacuevas/work_projects/voting_system_platform/Code/BigProject/LSTM_model_{dataset_name}.hdf5',
        monitor='val_acc', verbose=1)  # , initial_value_threshold=0.4)

    callbacks_list = [earlystop, checkpointer]

    model.fit(X_train_sub[:, :seq_len, :], y_train,
                        batch_size=batch_size, epochs=num_epoch, shuffle=True,
                        validation_split=0.15, callbacks=callbacks_list)

    results = model.evaluate(X_train_sub, y_train, batch_size=N_train)
    print("Training: test loss, test acc:", results)
    results = model.evaluate(X_test_sub, y_test, batch_size=N_test)
    print("Testing: test loss, test acc:", results)
    acc = results[1]
    return model, acc

def LSTM_test(model, trial_data):
    t_sample = 50
    trial_data = signal.resample(trial_data, t_sample, axis=1)
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
    # computer_root_path = "/Users/rosit/Documents/MCC/voting_system_platform/Datasets/"  # OMEN
    computer_root_path = "/Users/almacuevas/work_projects/voting_system_platform/Datasets/"  # MAC
    data_path = computer_root_path + dataset_foldername
    dataset_info = datasets_basic_infos[dataset_name]

    data, label = load_data_labels_based_on_dataset(dataset_name, subject_id, data_path)
    data_train, data_test, _, labels_train, labels_test, _ = train_test_val_split(
        dataX=data, dataY=label, valid_flag=False)
    target_names = dataset_info['target_names']

    print("******************************** Training ********************************")
    start = time.time()
    model, acc = LSTM_train(dataset_name, data_train, labels_train)
    end = time.time()
    print("Training time: ", end - start)

    print("******************************** Test ********************************")
    pred_list=[]
    for data_chosen, labels_chosen in zip(data_test, labels_test):
        start = time.time()
        array = LSTM_test(model, [data_chosen])
        end = time.time()
        print("One epoch, testing time: ", end - start)
        print(target_names)
        print("Probability: " , labels_chosen) # We select the last one, the last epoch which is the current one.
        print("Real: ", labels_chosen)
        pred=np.argmax(array)
        pred_list.append(pred)
        print("Prediction: ", pred)
    acc = np.mean(pred_list == labels_test)
    print("Final Acc: ", acc)