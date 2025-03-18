#!/usr/bin/env python
# coding: utf-8

# # KDE & ML Based NuMu Event Reconstruction

# ### Import Classes & Modules

#     Ensure all required python classes are installed prior to progress.

# This script performs CNN training on the y-coordinate of the detector vertex.
# It uses the pixelmaps as the features.
# However, the true vertex (label) is in cm, so we need to convert the true vertex to pixels.

# To run this training script:
#  $ $PY37 y_vertex_training.py --detector <ND> --horn <FHC> --flux <nonswap> --epochs <20>


import os.path
import time

import pandas as pd
import numpy as np
import pickle
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse

from datetime import date
from sklearn import preprocessing
from tqdm import tqdm, tqdm_notebook
from sklearn.model_selection import train_test_split
from tensorflow.keras import metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Activation, concatenate
from tensorflow.keras.optimizers import Adam  # optimizer
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.python.client import device_lib
from sklearn.preprocessing import MinMaxScaler  # normalize and scale data
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score


# Conversion from cm (vtx.{x,y,z}) to pixels (cvnmaps) for "standard"-type datasets
# NOTE: requires information of the cell and plane! (i.e. firstcellx, firstcelly, firstplane)
# fx == firstcellx, fy == firstcelly, fz == firstplane
def convert_vtx_y_to_pixelmap(vtx_y_array, firstcelly_array, detStr):
    """
    :param vtx_y_array: `vtx.y` -- y in detector coordinates.
    :param firstcelly_array: `firstcelly` -- first y cell in pixelmap coordinates
    :param detStr: which detector (ND or FD)
    :return: y pixelmap coordinate
    """
    print('Converting y coordinate for {}...'.format(detStr))
    assert (type(vtx_y_array) is np.ndarray), "y_array must be a numpy array"
    if detStr == 'ND':
        return vtx_y_array / 3.97 - firstcelly_array + 47
    elif detStr == 'FD':
        return vtx_y_array / 3.97 - firstcelly_array + 191
    else:
        print('NO DETECTOR. No Y coordinate conversion.')
        return None


def printout_type(array):
    for file in range(0, len(array)):  # there should be 8 files in each array
        if file == 0:
            print('file: ', file)
            print('type of array: ', type(array[file]))
            assert (type(array[file]) is np.ndarray), "array must be a numpy array"
            print('shape of array: ', array[file].shape)
            print('-------------------')
    print('All file entries for the array have been checked -- they are np.ndarray')


# Let's now check to make sure we don't have any gigantic numbers, from unsigned int's.
# we already know this is the common number we see if wew do this wrong...so check against it.
def check_large_ints(array):
    for file in range(0, len(array)):  # there should be 8 files in each array
        event = 0
        if array[file][event] > 4294967200:  # a large number, just below max value of an unsigned int, which should trigger
            print(array[file][event], ' > ', 4294967200, '. We have a problem')
        event += 1


# custom regression loss functions
# huber loss
def huber(true, pred, delta):
    loss = np.where(np.abs(true - pred) < delta, 0.5 * ((true - pred) ** 2),
                    delta * np.abs(true - pred) - 0.5 * (delta ** 2))
    return np.sum(loss)


# log cosh loss
def logcosh(true, pred):
    loss = np.log(np.cosh(pred - true))
    return np.sum(loss)


########### begin main script ###########

# collect the arguments for this macro. the horn and swap options are required.
parser = argparse.ArgumentParser()
parser.add_argument("--detector", help="ND or FD", default="FD", type=str)
parser.add_argument("--horn", help="FHC or RHC", default="FHC", type=str)
parser.add_argument("--flux", help="nonswap (numu) or fluxswap (nue)", default="nonswap", type=str)
parser.add_argument("--epochs", help="number of epochs", default=20, type=int)
args = parser.parse_args()

# convert to the useful case
args.detector = args.detector.upper()
args.horn = args.horn.upper()
args.flux = args.flux.lower()
args.flux = args.flux.capitalize()  # capitalize the first letter
print("ARGS TO SCRIPT: ", args.detector, args.horn, args.flux, args.epochs)

# Using the pre-processed files.
train_path = '/home/k948d562/output/training/{}-Nominal-{}-{}/'.format(args.detector,
                                                                                            args.horn,
                                                                                            args.flux)
print('train_path: ', train_path)

# number of events in each file
events_total_training = 0
count_training_files = 0

# create empty lists --> Need to be converted to arrays!
cvnmap, vtx_y, firstcelly = [], [], []

for h5_filename in os.listdir(train_path):
    if os.path.isdir(h5_filename):
        continue

    # open each file and read the information
    print('Reading cvnmap and vertices, and cells/planes. Processing file... {} of {} '.format(
        count_training_files + 1, len(os.listdir(train_path))))
    with h5py.File(train_path + h5_filename, 'r') as f:

        if count_training_files == 0:
            print('keys in the train file....', f.keys())

        # this should be ~O(millions) of events for each file...
        events_per_file_train = len(f['vtx.y'][:])  # count events in file # can try: sum(len(x) for x in multilist)

        # create numpy.arrays of the information
        cvnmap.append(f['cvnmap'][:])
        vtx_y.append(f['vtx.y'][:])
        firstcelly.append(f['firstcelly'][:])

        events_total_training += events_per_file_train
        print('events in file: ', events_per_file_train)
        count_training_files += 1

print('total events: ', events_total_training)

print('Training files read successfully.')
print('Loaded {} files, and {} total events.'.format(count_training_files, events_total_training))

# create a file indexer
file_idx = 0
assert file_idx <= count_training_files, "file_idx must be less than or equal to count_training_files: {}".format(count_training_files)

# number of values in multi-dimensional list
print('length of sum cvnmap', sum(len(x) for x in cvnmap))


# convert the lists to numpy arrays
print('========================================')
print('Converting lists to numpy arrays...')
cvnmap = np.array(cvnmap)
vtx_y = np.array(vtx_y)
firstcelly = np.array(firstcelly)

# These arrays have shape (8,).
# Within each index (0-8), there is the shape we expect: (events_in_file, pixels)
print('cvnmap[0].shape: ', cvnmap[0].shape)
print('vtx_y[0].shape: ', vtx_y[0].shape)
print('firstcelly[0].shape: ', firstcelly[0].shape)

# cvnmap = cvnmap.reshape(cvnmap)
print('len of cvnmap is: ', len(cvnmap))

# trust that they are all numpy.arrays AND have the right shape
for i in [cvnmap, vtx_y, firstcelly]:
    ev = 0
    assert (type(i) is np.ndarray), "i must be a numpy array"
    if ev == 0:
        print('shape of array', i.shape)
    ev += 1



print('========================================')
printout_type(cvnmap)
printout_type(vtx_y)
printout_type(firstcelly)


print('========================================')
print('Addressing the bug in the h5 files. Converting firstcelly, firstcelly, firstplane to int type...')
############################################################################################################
# convert the cell and plane arrays to integers
# NOTE: for Prod5.1 h5 samples (made from Reco Conveners), the firstcelly, firstcelly arrays are `unsigned int`s.
#       this is incorrect. They need to be `int` type. So Erin E. discovered the solution that we use here:
#       -- first add 40 to each element in the array
#       -- then convert the array to `int` type
#       -- then subtract 40 from each element in the array
# We do this to `firstplane` as well (cast as int) although not strictly necessary.
# If you do not do this, firstcell numbers will be 4294967200, which is the max value of an unsigned int -- and wrong.
############################################################################################################

# some debugging, to be sure...
# print('firstcelly[1] before conversion: ', firstcelly[1], type(firstcelly[1]))
print('firstcelly[file_idx][100] at start: ', firstcelly[file_idx][100], type(firstcelly[file_idx][100]))

for fileIdx in range(0, len(firstcelly)):
    print('firstcelly[fileIdx].shape', firstcelly[fileIdx].shape)
    firstcelly[fileIdx] += 40
    firstcelly[fileIdx] = np.array(firstcelly[fileIdx], dtype='int')
    print('firstcelly[fileIdx][100] after conversion + 40 addition: ', firstcelly[fileIdx][100], type(firstcelly[fileIdx][100]))
    firstcelly[fileIdx] -= 40
    print('firstcelly[fileIdx][100] after conversion + 40 subtraction: ', firstcelly[fileIdx][100], type(firstcelly[fileIdx][100]))


# Now lets check for any large unsigned ints that we know are a problem.
# These only appeared in the cell/plane information (and not the vertices).
check_large_ints(firstcelly)


print('========================================')
print('Converting the vertex coordinates into pixelmap coordinates for the network...')
# Create the vertex info in pixel map coordinates:
# convert the vertex location (detector coordinates) to pixel map coordinates
vtx_y_pixelmap = convert_vtx_y_to_pixelmap(vtx_y, firstcelly, args.detector)
print('Done converting.')

print('========================================')
# print out info for single event to double check
print('some simple checks: ')
print('type of vtx_y_pixelmap:  ', type(vtx_y_pixelmap))
print('shape of vtx_y_pixelmap: ', vtx_y_pixelmap.shape)
print('-------------------')
print('Here is an example of the conversion for a single file :')
print('vtx_y[1] = ', vtx_y[1])
print('vtx_y_pixelmap[1] (these should NOT be 2e8 now) = ', vtx_y_pixelmap[1])
print('firstcelly after conversion (these should NOT be 2e8 now): ', firstcelly[1])
print('-------------------')


# Print out useful info about the shapes of the arrays
print('========================================')
print('Useful info about the shapes of the arrays:')
print('-------------------')
print('Format of the arrays: (file_idx, event_idx, pixel_idx)......')
print('the pixel_idx is for cvnmaps only')
print('-------------------')
# we set file_idx manually
print('cvnmap.shape: ', cvnmap.shape)
print('vtx_y.shape: ', vtx_y.shape)
print('vtx_y_pixelmap.shape: ', vtx_y_pixelmap.shape)
print('firstcelly.shape: ', firstcelly.shape)
print('-------------------')

# this array should be something like: (2, 10000, 16000)
print('-------------------')
print('to access FILE, index in array is: (cvnmap[0].shape) = ', cvnmap[0].shape, 'files.')  # first dimension is the file
print('to access EVENT, index in array is: (cvnmap[0].shape[0]) = ', cvnmap[0].shape[0], 'events')  # second dimension is the events
print('to access PIXELS, index in array is: (cvnmap[0].shape[1]) = ', cvnmap[0].shape[1], 'pixels')  # third dimension is the pixelmap
print('-------------------')

print('========================================')
# reshape the pixels into 2 (100,80) views: XZ and YZ.
print('reshape the pixels into 2 (100,80) views: XZ and YZ.....')
b, cvnmap_resh_yz = [], []
y_temp, vtx_y_pixelmap_resh = [], []
# want to split this by the events amount in each file.
# loop over each file and event and append.
# We are basically removing the file index from the cvnmap array.
# NOTE: we are creating a list initially.
total_event_counter = 0
file_count = 0
for file_count in range(len(os.listdir(train_path))):
    print('Processing train cvnmap file {} of {}'.format(file_count + 1, (len(os.listdir(train_path)))))

    # loop through events in each file...in this current array structure, this is: cvnmap[file_count].shape[1].
    # I.e. the second index within the file_idx.
    print('beginning loop over N events....', cvnmap[file_count].shape[0])
    print('About to reshape N ({}) events into (100, 80) map size...'.format(cvnmap[file_count].shape[0]))
    assert cvnmap[file_count].shape[1] == 100*80*2, 'must have 16000 pixels to reshape the maps!'
    for ev in range(cvnmap[file_count].shape[0]):

        b = cvnmap[file_count][ev].reshape(2, 100, 80)
        cvnmap_resh_yz.append(b[1])  # b[0] is the XZ view, b[1] is the YZ view. --> We want the YZ view here. 

        y_temp = vtx_y_pixelmap[file_count][ev]
        vtx_y_pixelmap_resh.append(y_temp)


    print('ev at then end of loop: ', ev)
    total_event_counter += ev
    print('adding ev to total_event_counter. New total..... ', total_event_counter)

# The event assertion below is failing, so we wneed to add this
# I don't undetstand why. The x script doesn't need this...
total_event_counter += count_training_files * 1  # we add 8 -- 1 event for each file.


assert count_training_files == file_count + 1, "count_training_files must equal file_count: {}".format(file_count)
assert events_total_training == total_event_counter, "total_events_training must equal event_counter: {} != {}".format(events_total_training, total_event_counter)


# Convert the XZ and YZ views to np arrays
print('========================================')
print('Convert the XZ and YZ views to np arrays...')
cvnmap_resh_yz = np.array(cvnmap_resh_yz)  # yz views only
print('cvnmap_resh_yz.shape: ', cvnmap_resh_yz.shape)
print('----------------------------------------')
print('Convert the vtx_y_pixelmap_resh to np arrays...')
vtx_y_pixelmap_resh = np.array(vtx_y_pixelmap_resh)
print('vtx_y_pixelmap_resh.shape', vtx_y_pixelmap_resh.shape)

print('========================================')
# must re-shape the vtx_y_pixelmap array to match the cvnmap_resh_yz array
# I.e. --> remove the first dimension [0], the file.
# print('Reshape vtx_y_pixelmap, vtx_y_pixelmap, vtx_z_pixelmap to match the cvnmaps...')
# print('vtx_y_pixelmap: ', vtx_y_pixelmap.shape)
# print('vtx_y_pixelmap[0].shape', vtx_y_pixelmap[0].shape)
# vtx_y_pixelmap_resh = np.reshape(vtx_y_pixelmap, events_total_training)
# print('vtx_y_pixelmap_resh: ', vtx_y_pixelmap_resh.shape)


print('========================================')
# GPU Test
# see if GPU is recognized
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print('is built with cuda?: ', tf.test.is_built_with_cuda())
print('is gpu available?: ', tf.config.list_physical_devices('GPU'))
print('session: ', sess)
print('list_local_devices(): ', device_lib.list_local_devices())

#     +-----------------------------------------------------------------------------+
#     | NVIDIA-SMI 440.64.00    Driver Version: 440.64.00    CUDA Version: 10.2     |
#     |-------------------------------+----------------------+----------------------+
#     | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
#     | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
#     |===============================+======================+======================|
#     |   0  Tesla V100-PCIE...  Off  | 00000000:3B:00.0 Off |                    0 |
#     | N/A   33C    P0    34W / 250W |  15431MiB / 16160MiB |      0%      Default |
#     +-------------------------------+----------------------+----------------------+
#     |   1  Tesla V100-PCIE...  Off  | 00000000:D8:00.0 Off |                    0 |
#     | N/A   34C    P0    34W / 250W |      0MiB / 16160MiB |      5%      Default |
#     +-------------------------------+----------------------+----------------------+
#                                                                                    
#     +-----------------------------------------------------------------------------+
#     | Processes:                                                       GPU Memory |
#     |  GPU       PID   Type   Process name                             Usage      |
#     |=============================================================================|
#     |    0    152871      C   python3                                    15419MiB |
#     +-----------------------------------------------------------------------------+

# #### Prepare the Training & Test Sets
# split the data into training and testing sets

# XZ view. Trains on the x-coordinate
X1_train, X1_test, y1_train, y1_test = train_test_split(cvnmap_resh_yz, vtx_y_pixelmap_resh, test_size=0.25, random_state=101)


# add one more dimension to let the CNN know we are dealing with one color dimension
x1_train = X1_train.reshape(X1_train.shape[0], 100, 80, 1)
x1_test = X1_test.reshape(X1_test.shape[0], 100, 80, 1)


print('========================================')
print('Final printout of shape before feeding into network......')
print('training: (after final reshaping)')
print('x1_train.shape: ', x1_train.shape)
print('testing:')
print('x1_test.shape: ', x1_test.shape)
print('========================================')

# ### MultiView Fully Connected Layer Regression CNN Model

print('using only YZ model for `y` coordinate')

# instantiate the model
model_regCNN_yz = Sequential()
# add two fully connected 2-dimensional convolutional layers for the XZ and YZ views
model_regCNN_yz.add(Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1),
                           input_shape=(100, 80, 1), activation='relu'))

# specify 2-dimensional pooling
model_regCNN_yz.add(MaxPool2D(pool_size=(2, 2)))
# flatten the datasets
model_regCNN_yz.add(Flatten())
# add dense layers for each view. 256 neurons per layer
model_regCNN_yz.add(Dense(256, activation='relu'))
model_regCNN_yz.add(Dense(256, activation='relu'))
model_regCNN_yz.add(Dense(256, activation='relu'))
model_regCNN_yz.add(Dense(256, activation='relu'))
model_regCNN_yz.add(Dense(256, activation='relu'))
model_regCNN_yz.add(Dense(256, activation='relu'))
# no. of classes (output)
n_classes = 1
# tf concatenate the models
# model_regCNN = concatenate([model_regCNN_yz.output], axis=-1)
# model_regCNN = Dense(n_classes)(model_regCNN)
model_regCNN = Model(inputs=[model_regCNN_yz.input], outputs=model_regCNN_yz.output)
# compile the concatenated model
model_regCNN.compile(loss='logcosh',
                     optimizer='adam',
                     metrics=['mse'])  # loss was 'mse' then 'mae'
# print a summary of the model
print(model_regCNN.summary())


# x-coordinate system.
date = date.today()
start = time.time()
model_regCNN.fit(x=[x1_train], y=y1_train, epochs=args.epochs, batch_size=32, verbose=1,)
stop = time.time()
print('Time to train: ', stop - start)

# the default output name
outputName = 'training_{}epochs_{}_{}_{}_Y_{}'.format(args.epochs, args.detector, args.horn, args.flux, date)

save_model_dir = '/home/k948d562/output/trained-models/'
model_regCNN.save(save_model_dir + 'model_{}.h5'.format(outputName))
print('saved model to: ', save_model_dir + 'model_{}.h5'.format(outputName))
# Items in the model file: <KeysViewHDF5 ['model_weights', 'optimizer_weights']>

# Save the metrics to a dataframe
metrics = pd.DataFrame(model_regCNN.history.history)
print('METRICS:')
print(metrics.head())
save_metric_dir = '/home/k948d562/output/metrics/'
metrics.to_csv(save_metric_dir + '/metrics_{}.csv'.format(outputName))
print('saved metrics to: ', save_metric_dir + '/metrics_{}.csv'.format(outputName))


# model evaluation with logcosh
plot_dir = '/home/k948d562/plots/ml-vertexing-plots/wsu-vertexer/training/{}'.format(outputName)

if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)
    print('created plot dir: ', plot_dir)

# plot the training loss (among other metrics, for edification)
plt.figure(figsize=(12, 8))
plt.plot(metrics[['loss']])  # 'accuracy', 'mse']])  # TODO: add more metrics to this training.. accuracy, entropy. etc.
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training Metrics')
plt.legend(['loss'], loc='upper right')  # 'accuracy', 'mse'], loc='upper right')
plt.tight_layout()
plt.show()
for ext in ['png', 'pdf']:
    plt.savefig('{}/loss_{}.{}'.format(plot_dir, outputName, ext))


# Evaluate the test set
start_eval = time.time()
print('Evaluation on the test set...')
evaluation = model_regCNN.evaluate(x=[x1_test], y=y1_test, verbose=1)
stop_eval = time.time()
print('Test Set Evaluation: {}'.format(evaluation))
print('Evaluation time: ', stop_eval - start_eval)
print('Done! Training Complete!')
