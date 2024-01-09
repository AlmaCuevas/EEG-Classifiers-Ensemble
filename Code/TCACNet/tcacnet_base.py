import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from Code.data_utils import train_test_val_split
import torch.nn as nn
import torch.optim as optim
from tcacnet_utils.network import globalnetwork, localnetwork, topnetwork
from share import datasets_basic_infos
from data_loaders import load_data_labels_based_on_dataset
import numpy as np

# If a GPU is available, use it
if torch.cuda.is_available():
    device = torch.device("cuda")
    use_cuda = True
    print('Using cuda !')
else:
    device = torch.device("cpu")
    use_cuda = False
    print('GPU not available !')

class CustomDataset(Dataset):

    def __init__(self, data, label):
        self.data = data
        self.data_len = len(self.data)
        self.label_arr = label

    def __getitem__(self, index):
        label = int(self.label_arr[index])
        sample = np.asarray([self.data[index]])
        sample = torch.from_numpy(sample)
        return sample, label

    def __len__(self):
        return self.data_len

def EEGdata_loader(data, label, batch_size = 16, batch_size_eval = 16):
    x_train, x_test, x_valid, y_train, y_test, y_valid = train_test_val_split(dataX=data, dataY=label, valid_flag=True)

    train_data = CustomDataset(x_train, y_train)
    valid_data = CustomDataset(x_valid, y_valid)
    test_data = CustomDataset(x_test, y_test)
    print(x_train.shape)
    train_loader = DataLoader(train_data, batch_size=batch_size, pin_memory=use_cuda, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size_eval, pin_memory=use_cuda)
    test_loader = DataLoader(test_data, batch_size=batch_size_eval, pin_memory=use_cuda)

    return train_loader, valid_loader, test_loader

from tcacnet_utils.attention import inference

def train(model_global, model_local, model_top, optimizer, loss_fn_local_top, epoch, only_global_model, train_loader, channels, n_slices = 1):

    model_global.train()
    model_local.train()
    model_top.train()

    for batch_idx, (inputs, target) in enumerate(train_loader):
        inputs, target = inputs.to(device), target.to(device)
        inputs = inputs.float()
        target = target.long()
        optimizer.zero_grad()

        wpser = inputs[:,:,:,-1]    # WPSER corresponding to each channel
        inputs = inputs[:,:,:,0:inputs.shape[3]-1]    # raw EEG signal
        output_merged, hint_loss, channel_loss = inference(inputs, wpser, model_global, model_local, model_top, n_slices, channels, device,
                                                           only_global_model, is_training=True)

        loss_local_and_top = loss_fn_local_top(output_merged, target)
        loss_global_model = loss_local_and_top + hint_loss + channel_loss

        for param in model_local.parameters():
            param.requires_grad = False
        for param in model_top.parameters():
            param.requires_grad = False
        loss_global_model.backward(retain_graph=True)
        for param in model_local.parameters():
            param.requires_grad = True
        for param in model_top.parameters():
            param.requires_grad = True
        for param in model_global.parameters():
            param.requires_grad = False
        loss_local_and_top.backward()
        for param in model_global.parameters():
            param.requires_grad = True

        optimizer.step()

    if epoch % 10 == 0:
        print('\rTrain Epoch: {}'
              '  Total_Loss: {:.4f} (CrossEntropy: {:.2f} Hint: {:.2f} Ch: {:.2f})'
              ''.format(epoch, loss_local_and_top.item()+hint_loss.item(), loss_local_and_top.item(), hint_loss.item(), channel_loss.item()),
              end='')

    return loss_local_and_top.item()+hint_loss.item()+channel_loss.item(), loss_local_and_top.item()

def tcanet_test(model_global, model_local, model_top, test_loss_fn_local_top, epoch, loader, only_global_model, channels, verbose=True, n_slices = 1):
    model_global.eval()
    model_local.eval()
    model_top.eval()

    avg_test_loss, avg_hint_loss, avg_channel_loss = 0, 0, 0
    correct = 0
    test_size = 0
    with torch.no_grad():
        for inputs, target in loader:
            inputs, target = inputs.to(device), target.to(device)

            inputs = inputs.float()
            target = target.long()

            wpser = inputs[:,:,:,-1]
            inputs = inputs[:,:,:,0:inputs.shape[3]-1]

            output_merged, hint_loss, channel_loss = inference(inputs, wpser, model_global, model_local, model_top, n_slices, channels, device,
                                                               only_global_model, is_training=False)

            test_size += len(inputs)
            avg_test_loss += test_loss_fn_local_top(output_merged, target).item()
            avg_hint_loss += len(inputs) * hint_loss.item()
            avg_channel_loss += len(inputs) * channel_loss.item()
            pred = output_merged.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_test_loss /= test_size
    avg_hint_loss /= test_size
    avg_channel_loss /= test_size
    accuracy = correct / test_size

    if epoch % 10 == 0 and verbose:
        print('\nTest set: Avg_Total_Loss: {:.4f} (CrossEntropy: {:.2f} Hint: {:.2f} Ch: {:.2f})'
              '  Accuracy: {}/{} ({:.0f}%)\n'
              .format(avg_test_loss + avg_hint_loss + avg_channel_loss, avg_test_loss, avg_hint_loss, avg_channel_loss,
                      correct, test_size, 100. * accuracy))

    return avg_test_loss+avg_hint_loss+avg_channel_loss, avg_test_loss, accuracy, pred.numpy().flatten()

def call_tcanet(dataset_name, subject_id, train_loader, valid_loader, test_loader, channels, only_global_model: bool = True):  # only use global model
    n_epochs = 200
    loss_fn_local_top = nn.NLLLoss()
    test_loss_fn_local_top = nn.NLLLoss(reduction='sum')
    learning_rate = 0.0625 * 0.01

    if dataset_name == 'torres':
        kernel_size = 3
    elif 'aguilera' in dataset_name:
        kernel_size = 6
    elif dataset_name == 'coretto':
        kernel_size = 2
    elif dataset_name == 'nieto':
        kernel_size = 4

    model_global = globalnetwork(channels).to(device)
    model_local = localnetwork(channels).to(device)
    model_top = topnetwork(kernel_size).to(device)
    try:
        if only_global_model:
            optimizer = optim.Adam(list(model_global.parameters())
                                   + list(model_top.parameters()), lr=learning_rate)
        else:
            optimizer = optim.Adam(list(model_global.parameters())
                                   + list(model_local.parameters())
                                   + list(model_top.parameters()), lr=learning_rate)

        min_cross_entropy = 100000

        for ep in range(n_epochs):
            train_total_loss, train_cross_entropy = train(model_global, model_local, model_top, optimizer,
                                                          loss_fn_local_top, ep, only_global_model, train_loader, channels)
            valid_total_loss, valid_cross_entropy, valid_acc, preds_tcanet = tcanet_test(model_global, model_local, model_top,
                                                                          test_loss_fn_local_top, ep, valid_loader, only_global_model, channels, verbose=False)
            if valid_cross_entropy < min_cross_entropy:
                min_cross_entropy = valid_cross_entropy
                torch.save(model_global.state_dict(), f'model_global_cross_entropy_{dataset_name}_{subject_id}.pth')
                torch.save(model_local.state_dict(), f'model_local_cross_entropy_{dataset_name}_{subject_id}.pth')
                torch.save(model_top.state_dict(), f'model_top_cross_entropy_{dataset_name}_{subject_id}.pth')

        if only_global_model:
            print('\nUse global model:')
        else:
            print('\nUse TCACNet:')

        model_global.load_state_dict(torch.load(f'model_global_cross_entropy_{dataset_name}_{subject_id}.pth'))
        model_local.load_state_dict(torch.load(f'model_local_cross_entropy_{dataset_name}_{subject_id}.pth'))
        model_top.load_state_dict(torch.load(f'model_top_cross_entropy_{dataset_name}_{subject_id}.pth'))
        valid_total_loss, valid_cross_entropy, valid_acc, preds_tcanet = tcanet_test(model_global, model_local, model_top,
                                                                                    test_loss_fn_local_top, 0, test_loader, only_global_model, channels,
                                                                                    verbose=True)
        return model_global, model_local, model_top, valid_acc
    except:
        return False, False, False, np.nan

def tcanet_online_pred(model_global, model_local, model_top, loader, channels, only_global_model, n_slices = 1):
    model_global.eval()
    model_local.eval()
    model_top.eval()

    with torch.no_grad():
        for inputs, target in loader:
            inputs, target = inputs.to(device), target.to(device)

            inputs = inputs.float()

            wpser = inputs[:,:,:,-1]
            inputs = inputs[:,:,:,0:inputs.shape[3]-1]

            output_merged, hint_loss, channel_loss = inference(inputs, wpser, model_global, model_local, model_top, n_slices, channels, device,
                                                               only_global_model, is_training=False)
    return output_merged #.numpy().flatten()

if __name__ == '__main__':
    # Manual Inputs
    subject_id = 2  # Only two things I should be able to change
    dataset_name = 'aguilera_gamified'  # Only two things I should be able to change
    array_format = True

    # Folders and paths
    dataset_foldername = dataset_name + '_dataset'
    # computer_root_path = "/Users/rosit/Documents/MCC/voting_system_platform/Datasets/"  # OMEN
    computer_root_path = "//"  # MAC
    data_path = computer_root_path + dataset_foldername
    dataset_info = datasets_basic_infos[dataset_name]

    data, label = load_data_labels_based_on_dataset(dataset_name, subject_id, data_path)
    data_train, data_test, _, labels_train, labels_test, _ = train_test_val_split(
        dataX=data, dataY=label, valid_flag=False)
    train_loader, valid_loader, test_loader = EEGdata_loader(data_train, labels_train)
    model_global, model_local, model_top, acc = call_tcanet(dataset_name, subject_id, train_loader, valid_loader, test_loader, dataset_info['#_channels'], True)
    print(acc)
    model_global, model_local, model_top, acc = call_tcanet(dataset_name, subject_id, train_loader, valid_loader, test_loader, dataset_info['#_channels'], False)
    print(acc)
    pred_list_true = []
    pred_list_false = []

    for data_chosen, trial_chosen in zip(data_test, labels_test):
        trial_data = CustomDataset([data_chosen], [0])  # The label is dummy, it's just for the code to work
        use_cuda = False
        loader = DataLoader(trial_data, pin_memory=use_cuda)
        output_array_true = tcanet_online_pred(model_global, model_local, model_top, loader, dataset_info['#_channels'], only_global_model=True)
        output_array_false = tcanet_online_pred(model_global, model_local, model_top, loader, dataset_info['#_channels'],
                                          only_global_model=False)
        pred_true=np.argmax(output_array_true)
        pred_false = np.argmax(output_array_false)
        pred_list_true.append(pred_true)
        pred_list_false.append(pred_false)

        print("Answer true: ", pred_true)
        print("Answer false: ", pred_false)
        print("Real: ", trial_chosen)
    acc_true = np.mean(pred_list_true == labels_test)
    acc_false = np.mean(pred_list_false == labels_test)
    print("Accuracy true: ", acc_true)
    print("Accuracy false: ", acc_false)