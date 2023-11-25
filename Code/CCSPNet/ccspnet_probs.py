from ccspnet import manual_seed, EEG_dataset, MVLoss, CCSP, train, evaluate, predict_answer, EEG_dataset_online
import torch
from torch.utils.data import DataLoader
import numpy as np

from share import datasets_basic_infos
from data_loaders import load_data_labels_based_on_dataset
import time

#TODO: Load the CCSPNET file and divide it in train and test, find a way to provide the array from pairs.
# project_combine_cnn_lstm_run_3.ipynb does something similar, it didn't gave good result but you could try.
# We set the seed to 2045 because the Singularity is near!
# manual_seed(2045)

if (torch.cuda.is_available()):
    device = 'cuda'
    workers = 2
else:
    device = 'cpu'
    workers = 0
print(f'Device is set to {device}\nNumber of workers: {workers}')

def ccspnet_train(data, labels, dataset_info, true_label=1): # true_label=1, so 1 wil be 1 and everyone else 0
    """#Run Network
    Mode parameter: 'SD' for subject-dependent
    """

    dataset = EEG_dataset(data, labels, true_label)
    ### Define Dataloader
    dataloader = DataLoader(dataset, batch_size=5300,
                            num_workers=workers, pin_memory=True,
                            shuffle=True)

    hists_acc = []
    hists_loss = []
    manual_seed(2045)
    net_args = {"kernLength": 32,
                "timepoints": 250,
                "wavelet_filters": 4,
                "wavelet_kernel": 64,
                "nn_layers": [4],
                "feature_reduction_network": [16, 8, 4],
                "n_CSP_subspace_vectors": 2,
                "nb_classes": dataset_info['#_class'],
                "nchans": dataset_info['#_channels'],
                }

    train_args = {"dataloader": dataloader,
                  "epochs": 20,
                  "lr": 0.01,
                  "wavelet_lr": 0.001,
                  "loss_ratio": 0.3,
                  "criterion": MVLoss,
                  "verbose": 0,
                  "tensorboard": False,
                  "m": net_args['n_CSP_subspace_vectors'],
                  'lambda1': 0.01,
                  'lambda2': 0.1
                  }

    net = CCSP(**net_args).to(device)

    history = train(dataset=dataset, model=net, **train_args)
    ### Evaluate
    best_model_state = history['model'][np.argmin(history['train_loss'])]
    print('The Best Epoch:', np.argmin(history['train_loss']) + 1)
    test_loss, ts_acc = evaluate(net, best_model_state, dataloader, m=net_args['n_CSP_subspace_vectors'],
                                 criterion=MVLoss,
                                 loss_ratio=train_args["loss_ratio"], verbose=1, )
    hists_acc.append([test_loss, ts_acc])

    best_model_state = history['model'][np.argmax(history['train_accuracy'])]
    print('The Best Epoch based on accuracy:', np.argmax(history['train_accuracy']) + 1)
    test_loss, ts_acc = evaluate(net, best_model_state, dataloader, m=net_args['n_CSP_subspace_vectors'],
                                 criterion=MVLoss,
                                 loss_ratio=train_args["loss_ratio"], verbose=1, )
    hists_loss.append([test_loss, ts_acc])
    return net, best_model_state

def ccspnet_test(model, epoch, best_model_state):
    """
    This is what the real-time BCI will call.
    Parameters
    ----------
    model : classifier trained for the specific subject
    epoch: one epoch, the current one that represents the intention of movement of the user.

    Returns Array of classification with 4 floats representing the target classification
    -------

    """
    # To load the model, just in case
    #loaded_model = pickle.load(open(filename, 'rb'))

    # To see the array of predictions
    dataset = EEG_dataset_online(epoch)
    dataloader = DataLoader(dataset, batch_size=5300,
                            num_workers=workers, pin_memory=True,
                            shuffle=True)
    answer = predict_answer(model, best_model_state, dataloader)

    return answer # This is not an array. It's a 1 or 0

if __name__ == '__main__':
    # Manual Inputs
    subject_id = 10  # Only two things I should be able to change
    dataset_name = 'aguilera_traditional'  # Only two things I should be able to change
    array_format = True

    # Folders and paths
    dataset_foldername = dataset_name + '_dataset'
    # computer_root_path = "/Users/rosit/Documents/MCC/voting_system_platform/Datasets/"  # OMEN
    computer_root_path = "/Users/almacuevas/work_projects/voting_system_platform/Datasets/"  # MAC
    data_path = computer_root_path + dataset_foldername
    dataset_info = datasets_basic_infos[dataset_name]

    data, labels = load_data_labels_based_on_dataset(dataset_name, subject_id, data_path, array_format=array_format)
    target_names = dataset_info['target_names']

    print("******************************** Training ********************************")
    start = time.time()
    model, best_model_state = ccspnet_train(data, labels, dataset_info, true_label=0)
    end = time.time()
    print("Training time: ", end - start)

    print("******************************** Test ********************************")
    epoch_number = 0
    start = time.time()
    answer = ccspnet_test(model, np.asarray([data[epoch_number]]), best_model_state)
    end = time.time()
    print("One epoch, testing time: ", end - start)
    print(target_names)
    print("Answer: " , answer)
    print("Real: ", labels[epoch_number])