from sklearn.model_selection import StratifiedKFold

from CSP_LDA_probs import CSP_LDA_train, CSP_LDA_test
from Code.data_loaders import load_data_labels_based_on_dataset
from riemman_SVC_probs import riemman_train, riemman_test
from share import datasets_basic_infos
import numpy as np

from xdawn_riemman_probs import xdawn_riemman_train, xdawn_riemman_test
from xdwan_probs import xdawn_train, xdawn_test
from sklearn.preprocessing import normalize

def group_methods_train(data, epochs, labels, target_names):
    models_outputs = {}
    print("CSP-LDA")
    models_outputs['CSP_LDA_clf'], models_outputs['CSP_LDA_acc'] = CSP_LDA_train(data, labels, target_names)
    print("RIEMMAN SVC")
    models_outputs['riemman_clf'], models_outputs['riemman_acc'] = riemman_train(data, labels, target_names)
    print("XDAWN RIEMMAN")
    models_outputs['xdawn_riemman_clf'], models_outputs['xdawn_riemman_acc'] = xdawn_riemman_train(data, labels, target_names)
    print("XDAWN LogReg")
    models_outputs['xdawn_clf'], models_outputs['xdawn_acc'] = xdawn_train(epochs, labels, target_names)

    return models_outputs

def group_methods_test(models_outputs: dict, data_array, data_epoch):
    CSP_LDA_array = normalize(CSP_LDA_test(models_outputs['CSP_LDA_clf'], data_array))
    riemman_array = normalize(riemman_test(models_outputs['riemman_clf'], data_array))
    xdawn_riemman_array = normalize(xdawn_riemman_test(models_outputs['xdawn_riemman_clf'], data_array))
    xdawn_array = normalize(xdawn_test(models_outputs['xdawn_clf'], data_epoch))

    probs_list = [np.multiply(CSP_LDA_array, models_outputs['CSP_LDA_acc']),
                  np.multiply(riemman_array, models_outputs['riemman_acc']),
                  np.multiply(xdawn_riemman_array, models_outputs['xdawn_riemman_acc']),
                  np.multiply(xdawn_array, models_outputs['xdawn_acc']),
                  ]

    probs = np.nanmean(probs_list, axis=0)
    return probs

if __name__ == '__main__':
    # Manual Inputs
    dataset_name = 'aguilera_gamified'  # Only two things I should be able to change
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

    for subject_id in range(1, dataset_info['subjects']+1):  # Only two things I should be able to change
        print(subject_id)
        with open(
                f'/Users/almacuevas/work_projects/voting_system_platform/Results/notcanets_weighted_min25_{dataset_name}_nobaseline.txt',
                'a') as f:
            f.write(f'Subject: {subject_id}\n\n')
        data, labels = load_data_labels_based_on_dataset(dataset_name, subject_id, data_path, array_format=True)
        epochs, _ = load_data_labels_based_on_dataset(dataset_name, subject_id, data_path, array_format=False)
        target_names = dataset_info['target_names']

        # Do cross-validation
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        acc_over_cv = []
        for train, test in cv.split(epochs, labels):
            print("******************************** Training ********************************")
            #start = time.time()
            models_outputs = group_methods_train(data[train], epochs[train], labels[train], target_names)
            #end = time.time()
            with open(
                    f'/Users/almacuevas/work_projects/voting_system_platform/Results/notcanets_weighted_min25_{dataset_name}_nobaseline.txt',
                    'a') as f:
                f.write(f"{models_outputs['CSP_LDA_acc']}, {models_outputs['riemman_acc']}, {models_outputs['xdawn_riemman_acc']}, "
                        f"{models_outputs['xdawn_acc']}\n")
            #print("Training time: ", end - start)

            print("******************************** Test ********************************")
            pred_list=[]
            for epoch_number in test:
                #start = time.time()
                array = group_methods_test(models_outputs, np.asarray([data[epoch_number]]), epochs[epoch_number])
                #end = time.time()
                #print("One epoch, testing time: ", end - start)
                print(target_names)
                print("Probability: " , array)
                pred=np.argmax(array)
                pred_list.append(pred)
                print("Prediction: ", pred)
                print("Real: ", labels[epoch_number])
            acc = np.mean(pred_list == labels[test])
            acc_over_cv.append(acc)
            with open(
                    f'/Users/almacuevas/work_projects/voting_system_platform/Results/notcanets_weighted_min25_{dataset_name}_nobaseline.txt',
                    'a') as f:
                f.write(f'Prediction: {pred_list}\n')
                f.write(f'Real label:{labels[test]}\n')
                f.write(f'Mean accuracy in KFold: {acc}\n')
            print("Mean accuracy in KFold: ", acc)
        mean_acc_over_cv = np.mean(acc_over_cv)
        with open(
                f'/Users/almacuevas/work_projects/voting_system_platform/Results/notcanets_weighted_min25_{dataset_name}_nobaseline.txt',
                'a') as f:
            f.write(f'Final acc: {mean_acc_over_cv}\n\n\n\n')
        print(f'Final acc: {mean_acc_over_cv}')
        print("Congrats! The first draft of the voting system is now working.")
        #todo: check the 3 subjects of coretto that i DONWLOAD, TO JUSTIFY THAT IT DOESNT WORK AND DELETE THEM
