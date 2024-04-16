# The first part can be the suggested model, the second is XDAWN+Riemman, exactly the one we already have.

import numpy as np

# EEGNet-specific imports
from EEGModels import EEGNet, ShallowConvNet, DeepConvNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

from Code.data_utils import train_test_val_split
from share import datasets_basic_infos
from data_loaders import load_data_labels_based_on_dataset
from pathlib import Path

ROOT_VOTING_SYSTEM_PATH: Path = Path(__file__).parent.parent.resolve()

# while the default tensorflow ordering is 'channels_last' we set it here
# to be explicit in case if the user has changed the default ordering
K.set_image_data_format('channels_last')

# trials, channels, samples
# Manual Inputs
subject_id = 1  # Only two things I should be able to change
dataset_name = 'aguilera_gamified'  # Only two things I should be able to change
array_format = True

# Folders and paths
dataset_foldername = dataset_name + '_dataset'
computer_root_path = ROOT_VOTING_SYSTEM_PATH + "/Datasets/"
data_path = computer_root_path + dataset_foldername
dataset_info = datasets_basic_infos[dataset_name]

data, label = load_data_labels_based_on_dataset(dataset_info, subject_id, data_path)
label = label +1 # It can't read 0,1,2,3; it has to be 1,2,3,4
target_names = dataset_info['target_names']


X_train, X_test, X_validate, Y_train, Y_test, Y_validate = train_test_val_split(
            dataX=data, dataY=label, valid_flag=True)

kernels, chans, samples = 1, dataset_info['#_channels'], dataset_info['samples']


############################# EEGNet portion ##################################

# convert labels to one-hot encodings.
Y_train      = np_utils.to_categorical(Y_train-1)
Y_validate   = np_utils.to_categorical(Y_validate-1)
Y_test       = np_utils.to_categorical(Y_test-1)

# convert data to NHWC (trials, channels, samples, kernels) format. Data 
# contains 60 channels and 151 time-points. Set the number of kernels to 1.
X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
X_validate   = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)
   
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# configure the EEGNet-8,2,16 model with kernel length of 32 samples (other 
# model configurations may do better, but this is a good starting point)
model = EEGNet(nb_classes = 4, Chans = chans, Samples = samples,
               dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16,
               dropoutType = 'Dropout')

# compile the model and set the optimizers
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics = ['accuracy'])

# count number of parameters in the model
numParams    = model.count_params()    

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
class_weights = {0:1, 1:1, 2:1, 3:1}

################################################################################
# fit the model. Due to very small sample sizes this can get
# pretty noisy run-to-run, but most runs should be comparable to xDAWN + 
# Riemannian geometry classification (below)
################################################################################
model.fit(X_train, Y_train, batch_size = 16, epochs = X_train.shape[0],
                        verbose = 2, validation_data=(X_validate, Y_validate),
                        callbacks=[checkpointer], class_weight = class_weights)

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

probs       = model.predict(X_test)
preds       = probs.argmin(axis = -1)
acc         = np.mean(preds == Y_test.argmax(axis=-1))
print(preds)
print(Y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))

# plot the confusion matrices for both classifiers
#plt.figure(0)
#cm = confusion_matrix(Y_test.argmax(axis = -1), preds, labels=[1,2,3,4])
#ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)


############################## PyRiemann Portion ##############################
#
## code is taken from PyRiemann's ERP sample script, which is decoding in
## the tangent space with a logistic regression
#
#n_components = 2  # pick some components
#
## set up sklearn pipeline
#clf = make_pipeline(XdawnCovariances(n_components),
#                    TangentSpace(metric='riemann'),
#                    LogisticRegression())
#
#preds_rg     = np.zeros(len(Y_test))
#
## reshape back to (trials, channels, samples)
#X_train      = X_train.reshape(X_train.shape[0], chans, samples)
#X_test       = X_test.reshape(X_test.shape[0], chans, samples)
#
## train a classifier with xDAWN spatial filtering + Riemannian Geometry (RG)
## labels need to be back in single-column format
#clf.fit(X_train, Y_train.argmax(axis = -1))
#preds_rg     = clf.predict(X_test)
#
## Printing the results
#acc2         = np.mean(preds_rg == Y_test.argmax(axis = -1))
#print("Classification accuracy: %f " % (acc2))
#
#
#
#plt.figure(1)
#cm_rg = confusion_matrix(Y_test.argmax(axis = -1), preds_rg, labels=[1,2,3,4])
#ConfusionMatrixDisplay(confusion_matrix=cm_rg, display_labels=target_names)



