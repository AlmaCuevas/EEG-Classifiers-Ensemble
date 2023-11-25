from CSP_LDA_probs import CSP_LDA_train, CSP_LDA_test
from Code.data_loaders import load_data_labels_based_on_dataset
from ccspnet_probs import ccspnet_train, ccspnet_test
from riemman_SVC_probs import riemman_train, riemman_test
from share import datasets_basic_infos
import time
import numpy as np

from tcanet_probs import tcanet_train, tcanet_test
from xdawn_riemman_probs import xdawn_riemman_train, xdawn_riemman_test
from xdwan_probs import xdawn_train, xdawn_test
from sklearn.preprocessing import normalize
from copy import deepcopy

def group_methods_train(dataset_name, subject_id, data, epochs, labels, dataset_info):
    models_outputs = {}
    print("TCANET Global Model")
    models_outputs['model_only_global'], models_outputs['model_only_local'], models_outputs['model_only_top'] = tcanet_train(dataset_name, subject_id, data, labels, channels=dataset_info['#_channels'], ONLY_GLOBAL_MODEL=True)
    print("TCANET")
    models_outputs['model_tcanet_global'], models_outputs['model_tcanet_local'], models_outputs['model_tcanet_top'] = tcanet_train(dataset_name, subject_id, data, labels, channels=dataset_info['#_channels'], ONLY_GLOBAL_MODEL=False)
    print("CSP-LDA")
    models_outputs['CSP_LDA_clf'] = CSP_LDA_train(data, labels, target_names)
    #print("CSSPNET")
    #models_outputs['net_0'], models_outputs['best_model_state_0'] = ccspnet_train(data, labels, dataset_info, true_label=0)
    #models_outputs['net_1'], models_outputs['best_model_state_1'] = ccspnet_train(data, labels, dataset_info, true_label=1)
    #models_outputs['net_2'], models_outputs['best_model_state_2'] = ccspnet_train(data, labels, dataset_info, true_label=2)
    #models_outputs['net_3'], models_outputs['best_model_state_3'] = ccspnet_train(data, labels, dataset_info, true_label=3)
    print("RIEMMAN SVC")
    models_outputs['riemman_clf'] = riemman_train(data, labels, target_names)
    print("XDAWN RIEMMAN")
    models_outputs['xdawn_riemman_clf'] = xdawn_riemman_train(data, labels, target_names)
    print("XDAWN LogReg")
    models_outputs['xdawn_clf'] = xdawn_train(epochs, labels, target_names)

    return models_outputs

def group_methods_test(dataset_name, subject_id, models_outputs: dict, data_array, data_epoch):
    tcanet_only_global_model_array = tcanet_test(dataset_name, subject_id, models_outputs['model_only_global'],
                                                 models_outputs['model_only_local'], models_outputs['model_only_top'],
                                                 data_array, dataset_info['#_channels'], True)
    tcanet_array = tcanet_test(dataset_name, subject_id, models_outputs['model_tcanet_global'],
                               models_outputs['model_tcanet_local'], models_outputs['model_tcanet_top'], data_array,
                               dataset_info['#_channels'], False)
    CSP_LDA_array = CSP_LDA_test(models_outputs['CSP_LDA_clf'], data_array)
    #answer_0 = ccspnet_test(models_outputs['net_0'], deepcopy(data_array), models_outputs['best_model_state_0'])
    ##answer_1 = ccspnet_test(models_outputs['net_1'], deepcopy(data_array), models_outputs['best_model_state_1'])
    ##answer_2 = ccspnet_test(models_outputs['net_2'], deepcopy(data_array), models_outputs['best_model_state_2'])
    ##answer_3 = ccspnet_test(models_outputs['net_3'], deepcopy(data_array), models_outputs['best_model_state_3'])
    #ccspnet_array = [answer_0, answer_1, answer_2, answer_3]
    riemman_array = riemman_test(models_outputs['riemman_clf'], data_array)
    xdawn_riemman_array = xdawn_riemman_test(models_outputs['xdawn_riemman_clf'], data_array)
    xdawn_array = xdawn_test(models_outputs['xdawn_clf'], data_epoch)

    #ccspnet_array #todo: is it always 1s? 2 times have happened so far
    probs_list = [normalize(array) for array in [tcanet_only_global_model_array, tcanet_array, CSP_LDA_array,
                                            riemman_array, xdawn_riemman_array, xdawn_array]]
    probs = np.mean(probs_list, axis=0)
    return probs

if __name__ == '__main__':
    # Manual Inputs
    dataset_name = 'aguilera_traditional'  # Only two things I should be able to change
    # array_format = True

    # Folders and paths
    dataset_foldername = dataset_name + '_dataset'
    # computer_root_path = "/Users/rosit/Documents/MCC/voting_system_platform/Datasets/"  # OMEN
    computer_root_path = "/Users/almacuevas/work_projects/voting_system_platform/Datasets/"  # MAC
    data_path = computer_root_path + dataset_foldername

    if dataset_name not in ['aguilera_traditional', 'aguilera_gamified', 'nieto', 'coretto', 'torres']:
        raise Exception(
            f"Not supported dataset named '{dataset_name}', choose from the following: aguilera_traditional, aguilera_gamified, nieto, coretto or torres.")
    dataset_info = datasets_basic_infos[dataset_name]

    for subject_id in range(1, 2):#dataset_info['subjects']+1):  # Only two things I should be able to change
        print(subject_id)
        data, labels = load_data_labels_based_on_dataset(dataset_name, subject_id, data_path, array_format=True)
        epochs, _ = load_data_labels_based_on_dataset(dataset_name, subject_id, data_path, array_format=False)
        target_names = dataset_info['target_names']

        print("******************************** Training ********************************")
        start = time.time()
        models_outputs = group_methods_train(dataset_name, subject_id, data, epochs, labels, dataset_info)
        end = time.time()
        print("Training time: ", end - start)

        print("******************************** Test ********************************")
        pred_list=[]
        for epoch_number in range(0, dataset_info['total_trials']):
            start = time.time()
            array = group_methods_test(dataset_name, subject_id, models_outputs, np.asarray([data[epoch_number]]), epochs[epoch_number])
            end = time.time()
            print("One epoch, testing time: ", end - start)
            print(target_names)
            print("Probability: " , array)
            pred=np.argmax(array)
            pred_list.append(pred)
            print("Prediction: ", pred)
            print("Real: ", labels[epoch_number])
        acc = np.mean(pred_list == labels)
        print("Final accuracy of all epochs: ", acc)
        print("Congrats! The first draft of the voting system is now working.")
        #todo: check if ccspnet actually works, and correct the final format
        #todo: check the 3 subjects of coretto that i DONWLOAD, TO JUSTIFY THAT IT DOESNT WORK AND DELETE THEM
