from compared_models import weights_init, MaxNormDefaultConstraint, EEGNet, ShallowConvNet
from lmda_model import LMDA
from experiment import EEGDataLoader, Experiment, setup_seed
# dataloader and preprocess
from Code.data_preprocess import preprocess4mi
# tools for pytorch
from torch.utils.data import DataLoader
import torch
# tools for numpy as scipy and sys
import logging
import os
import time
import datetime
# tools for plotting confusion matrices and t-SNE
from torchsummary import summary
from voting_system_platform.Code.data_loaders import load_data_labels_based_on_dataset
from voting_system_platform.share import datasets_basic_infos
from voting_system_platform.Code.data_utils import train_test_val_split

import warnings
# ========================= LMDA general run =====================================

def data_and_model(dataset_name: str, valid_flag: bool = False):
    data, label = load_data_labels_based_on_dataset(dataset_name, subject_id, dataset_info, data_path)
    data = preprocess4mi(data) # This was for MI, try it. If it doesn't work, delete it.
    x_train, x_test, x_val, y_train, y_test, y_val = train_test_val_split(dataX = data, dataY= label, valid_flag=valid_flag)

    if valid_flag:
        valid_loader = EEGDataLoader(x_val, y_val)
        valid_dl = DataLoader(valid_loader, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
    else:
        valid_dl = None

    train_loader = EEGDataLoader(x_train, y_train)
    test_loader = EEGDataLoader(x_test, y_test)

    train_dl = DataLoader(train_loader, batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
    test_dl = DataLoader(test_loader, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)


    model_id = '%s' % share_model_name
    folder_path = './%s/%s/' % (dataset_foldername, subject_id)  # mkdir in current folder, and name it by target's num
    folder = os.path.exists(folder_path)
    if not folder:
        os.makedirs(folder_path)
    output_file = os.path.join(folder_path, '%s%s.log' % (model_id, code_num))
    fig_path = folder_path + str(model_id) + code_num  # 用来代码命名
    print(fig_path)
    logging.basicConfig(
        datefmt='%Y/%m/%d %H:%M:%S',
        format="%(asctime)s %(levelname)s : %(message)s",
        level=logging.INFO,
        filename=output_file,
    )
    # log.info自带format, 因此只需要使用响应的占位符即可.
    logging.info("****************  %s for %s! ***************************", model_id, subject_id)

    if share_model_name == 'LMDA':
        Net = LMDA(num_classes=model_para['num_classes'], chans=model_para['chans'], samples=data.shape[3],
                   channel_depth1=model_para['channel_depth1'],
                   channel_depth2=model_para['channel_depth2'],
                   kernel=model_para['kernel'], depth=model_para['depth'],
                   ave_depth=model_para['pool_depth'], avepool=model_para['avepool'],
                   ).to(device)
        logging.info(model_para)

    elif share_model_name == 'EEGNet':
        Net = EEGNet(num_classes=model_para['num_classes'], chans=model_para['chans'], samples=data.shape[3]).to(device)

    else:  # ConvNet
        Net = ShallowConvNet(num_classes=model_para['num_classes'], chans=model_para['chans'], samples=data.shape[3]).to(device)

    Net.apply(weights_init)
    Net.apply(weights_init)

    logging.info(summary(Net, show_input=False))

    model_optimizer = torch.optim.AdamW(Net.parameters(), lr=lr_model)
    model_constraint = MaxNormDefaultConstraint()
    return train_dl, valid_dl, test_dl, Net, model_optimizer, model_constraint, fig_path


if __name__ == "__main__":
    #mne.set_log_level(verbose='warning')  # to avoid info at terminal
    warnings.filterwarnings(action="ignore", category=DeprecationWarning)
    warnings.filterwarnings(action="ignore", category=FutureWarning)
    start_time = time.time()
    setup_seed(521)  # 521, 322
    print('* * ' * 20)


    # Manual Inputs
    #subject_id = 1                      # Only 3 things I should be able to change
    dataset_name = 'nieto'           # Only 3 things I should be able to change
    share_model_name = 'LMDA'           # Only 3 things I should be able to change
    assert share_model_name in ['LMDA', 'EEGNet', 'ConvNet']

    dataset_info = datasets_basic_infos[dataset_name]
    times=[]
    for subject_id in range(10, dataset_info['subjects']+1):
        # Folders and paths
        dataset_foldername = dataset_name + '_dataset'
        #computer_root_path = "/Users/almacuevas/work_projects/voting_system_platform/Datasets/" # MAC
        computer_root_path = "/Users/rosit/Documents/MCC/voting_system_platform/Datasets/"  # OMEN
        data_path = computer_root_path + dataset_foldername


        assert dataset_info['subjects'] >= int(subject_id)

        device = torch.device('cuda')


        print('subject_id: ', subject_id)

        model_para = {
            'num_classes': dataset_info['#_class'],
            'chans': dataset_info['#_channels'],
            'channel_depth1': dataset_info['#_channels'] + 2,  # It is recommended that the number of convolutional layers in the time domain is more than the number of convolutional layers in the spatial domain.
            'channel_depth2': 9,
            'kernel': 75,
            'depth': 9,
            'pool_depth': 1,
            'avepool': dataset_info['sample_rate'] // 10,  # 还是推荐两步pooling的
            'avgpool_step1': 1,
        }


        today = datetime.date.today().strftime('%m%d')
        if share_model_name == 'LMDA':

            code_num = 'D{depth}_D{depth1}_D{depth2}_pool{pldp}'.format(depth=model_para['depth'],
                                                                        depth1=model_para['channel_depth1'],
                                                                        depth2=model_para['channel_depth2'],
                                                                        pldp=model_para['avepool'] * model_para[
                                                                            'avgpool_step1'])
        else:
            code_num = ''
        print(share_model_name + code_num)
        print(device)
        print(model_para)
        print('* * ' * 20)

        # ===============================  超 参 数 设 置 ================================
        lr_model = 1e-3
        step_one_epochs = 300
        batch_size = 23
        kwargs = {'num_workers': 1, 'pin_memory': True}  # 因为pycharm开启了多进行运行main, num_works设置多个会报错
        # ================================================================================

        train_dl, valid_dl, test_dl, Net, model_optimizer, model_constraint, fig_path = data_and_model(dataset_name)

        exp = Experiment(model=Net,
                         device=device,
                         optimizer=model_optimizer,
                         train_dl=train_dl,
                         test_dl=test_dl,
                         val_dl=valid_dl,
                         fig_path=fig_path,
                         model_constraint=model_constraint,
                         step_one=step_one_epochs,
                         classes=model_para['num_classes'],
                         )
        exp.run()

        end_time = time.time()
        times.append(end_time)
        # print('Net channel weight:', ShallowNet.channel_weight)
    logging.info('Param again {}'.format(model_para))
    logging.info('Done! Running time %.5f', end_time - start_time)
    print(times)
