#!/usr/bin/env python
# coding: utf-8

# # KDE & ML Based NuMu Event Reconstruction

# ### Import Classes & Modules

#     Ensure all required python classes are installed prior to progress.

# This script performs CNN training on the z-coordinate of the detector vertex.
# It uses the pixelmaps to as the features.
# However, the true vertex (label) is in cm, so we need to convert the true vertex to pixels.

# To run this training script:
#  $ $PY37 z_vertex_training.py --detector <ND> --horn <FHC> --flux <nonswap> --epochs <20>


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
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Activation, concatenate, Permute, Average, Lambda
from tensorflow.keras.optimizers import Adam  # optimizer
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.python.client import device_lib
from sklearn.preprocessing import MinMaxScaler  # normalize and scale data
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
#from keras.layers import GlobalAveragePooling1D


# Conversion from cm (vtx.{x,y,z}) to pixels (cvnmaps) for "standard"-type datasets
# NOTE: requires information of the cell and plane! (i.e. firstcellx, firstcelly, firstplane)
# fx == firstcellx, fy == firstcelly, fz == firstplane
def convert_vtx_z_to_pixelmap(vtx_z_array, firstplane_array, det):
    """
    :param vtx_z_array: `vtx.z` -- z in detector coordinates.
    :param firstplane_array: `firstplane` -- first plane in pixelmap coordinates
    :param det: which detector (ND or FD)
    :return: z pixelmap coordinate
    """
    print('Converting z coordinate for {}...'.format(det))
    assert (type(vtx_z_array) is np.ndarray), "z_array must be a numpy array"
    if det == 'ND':
       return vtx_z_array / 6.61 - firstplane_array    #Make sure that conversion factor 6.61 is the same in when you try to convert back to the detector coordinate in your predictions script 
    elif det == 'FD':
        return vtx_z_array / 6.664 - firstplane_array  #maintain the conversion factor 6.664 for FD when you are trying to convert back to the detector coordinate in your prediction script
    else:
        print('NO DETECTOR. No Z coordinate conversion.')
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
print(args.detector, args.horn, args.flux)

# Using the pre-processed files.
train_path = '/homes/m962g264/wsu_Nova_Vertexer/training_files_preprocess/files/training_file/{}-Nominal-{}-{}/'.format(args.detector,
                                                                                                                            args.horn,
                                                                                                                            args.flux)
print('train_path: ', train_path)

# number of events in each file
events_total_training = 0
count_training_files = 0

# create empty lists --> Need to be converted to arrays!
cvnmap, vtx_z, firstplane = [], [], []

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
        events_per_file_train = len(f['vtx.x'][:])  # count events in file # can try: sum(len(x) for x in multilist)

        # create numpy.arrays of the information
        cvnmap.append(f['cvnmap'][:])
        vtx_z.append(f['vtx.z'][:])
        firstplane.append(f['firstplane'][:])

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
vtx_z = np.array(vtx_z)
firstplane = np.array(firstplane)

# These arrays have shape (8,).
# Within each index (0-8), there is the shape we expect: (events_in_file, pixels)
print('cvnmap[0].shape: ', cvnmap[0].shape)
print('vtx_z[0].shape: ', vtx_z[0].shape)
print('firstplane[0].shape: ', firstplane[0].shape)

# cvnmap = cvnmap.reshape(cvnmap)
print('len of cvnmap is: ', len(cvnmap))

# trust that they are all numpy.arrays AND have the right shape
for i in [cvnmap, vtx_z, firstplane]:
    ev = 0
    assert (type(i) is np.ndarray), "i must be a numpy array"
    if ev == 0:
        print('shape of array', i.shape)
    ev += 1



print('========================================')
printout_type(cvnmap)
printout_type(vtx_z)
printout_type(firstplane)


print('========================================')
print('Addressing the bug in the h5 files. Converting firstcellx, firstcelly, firstplane to int type...')
############################################################################################################
# convert the cell and plane arrays to integers
# NOTE: for Prod5.1 h5 samples (made from Reco Conveners), the firstcellx, firstcelly arrays are `unsigned int`s.
#       this is incorrect. They need to be `int` type. So Erin E. discovered the solution that we use here:
#       -- first add 40 to each element in the array
#       -- then convert the array to `int` type
#       -- then subtract 40 from each element in the array
# We do this to `firstplane` as well (cast as int) although not strictly necessary.
# If you do not do this, firstcell numbers will be 4294967200, which is the max value of an unsigned int -- and wrong.
############################################################################################################

# some debugging, to be sure...

# loop over the events
for fileIdx in range(0, len(firstplane)):
    print('casting the `firstplane` array to int type...')
    firstplane[fileIdx] = np.array(firstplane[fileIdx], dtype='int')  # not strictly necessary, Erin doesn't do it...


# Now lets check for any large unsigned ints that we know are a problem.
# These only appeared in the cell/plane information (and not the vertices).
check_large_ints(firstplane)


print('========================================')
print('Converting the vertex coordinates into pixelmap coordinates for the network...')
# Create the vertex info in pixel map coordinates:
# convert the vertex location (detector coordinates) to pixel map coordinates
vtx_z_pixelmap = convert_vtx_z_to_pixelmap(vtx_z, firstplane, args.detector)
print('Done converting.')

print('========================================')
# print out info for single event to double check
print('some simple checks: ')
print('Here is an example of the conversion for a single file :')
print('vtx_z[1] = ', vtx_z[1])
print('vtx_z_pixelmap[1] = ', vtx_z_pixelmap[1])
print('-------------------')


# Print out useful info about the shapes of the arrays
print('========================================')
print('Useful info about the shapes of the arrays:')
print('-------------------')
print('Format of the arrays: (file_idx, event_idx, pixel_idx)......')
print('the pixel_idx is for cvnmaps only')
print('-------------------')
# we set file_idx manually
print('-------------------')
print('vtx_z.shape: ', vtx_z.shape)
print('vtx_z_pixelmap.shape: ', vtx_z_pixelmap.shape)
print('firstplane.shape: ', firstplane.shape)

# this array should be something like: (2, 10000, 16000)
print('-------------------')
print('to access FILE, index in array is: (cvnmap[0].shape) = ', cvnmap[0].shape, 'files.')  # first dimension is the file
print('to access EVENT, index in array is: (cvnmap[0].shape[0]) = ', cvnmap[0].shape[0], 'events')  # second dimension is the events
print('to access PIXELS, index in array is: (cvnmap[0].shape[1]) = ', cvnmap[0].shape[1], 'pixels')  # third dimension is the pixelmap
print('-------------------')

print('========================================')
# reshape the pixels into 2 (100,80) views: XZ and YZ.
print('reshape the pixels into 2 (100,80) views: XZ and YZ.....')
b, cvnmap_resh_xz, cvnmap_resh_yz = [], [], []
z_temp, vtx_z_pixelmap_resh = [], []
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
    event_counter_in_file = 0
    print('beginning loop over N events....', cvnmap[file_count].shape[0])
    print('About to reshape N ({}) events into (100, 80) map size...'.format(cvnmap[file_count].shape[0]))
    assert cvnmap[file_count].shape[1] == 100*80*2, 'must have 16000 pixels to reshape the maps!'
    for ev in range(cvnmap[file_count].shape[0]):

        b = cvnmap[file_count][ev].reshape(2, 100, 80)
        cvnmap_resh_xz.append(b[0])  # b[0] is the XZ view.
        cvnmap_resh_yz.append(b[1])  # b[0] is the YZ view.

        z_temp = vtx_z_pixelmap[file_count][ev]
        vtx_z_pixelmap_resh.append(z_temp)

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
cvnmap_resh_xz = np.array(cvnmap_resh_xz)  # xz views only
cvnmap_resh_yz = np.array(cvnmap_resh_yz)  # yz views only
print('cvnmap_resh_xz.shape: ', cvnmap_resh_xz.shape)
print('cvnmap_resh_yz.shape: ', cvnmap_resh_yz.shape)
print('----------------------------------------')
print('Convert the vtx_x_pixelmap_resh to np arrays...')
vtx_z_pixelmap_resh = np.array(vtx_z_pixelmap_resh)
print('vtx_z_pixelmap_resh.shape', vtx_z_pixelmap_resh.shape)


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

# XZ view. Trains on the z-coordinate
XZ_train_feature_x, XZ_test_feature_x, XZ_train_label_y, XZ_test_label_y = train_test_split(cvnmap_resh_xz, vtx_z_pixelmap_resh, test_size=0.25, random_state=101)

# YZ view. Trains on the z-coordinate
YZ_train_feature_x, YZ_test_feature_x, YZ_train_label_y, YZ_test_label_y = train_test_split(cvnmap_resh_yz, vtx_z_pixelmap_resh, test_size=0.25, random_state=101)

# add one more dimension to let the CNN know we are dealing with one color dimension
XZ_train_feature_x = XZ_train_feature_x.reshape(XZ_train_feature_x.shape[0], 100, 80, 1)
XZ_test_feature_x = XZ_test_feature_x.reshape(XZ_test_feature_x.shape[0], 100, 80, 1)
YZ_train_feature_x = YZ_train_feature_x.reshape(YZ_train_feature_x.shape[0], 100, 80, 1)
YZ_test_feature_x = YZ_test_feature_x.reshape(YZ_test_feature_x.shape[0], 100, 80, 1)


print('========================================')
print('Final printout of shape before feeding into network......')
print('training: (after final reshaping)')
print('XZ_train_feature.shape: ', XZ_train_feature_x.shape)
#print('YZ_train_feature.shape: ', YZ_train_feature_x.shape)
print('testing:')
print('XZ_test_feature.shape: ', XZ_test_feature_x.shape)
#print('YZ_test_feature.shape: ', YZ_test_feature_x.shape)
print('========================================')

# ### MultiView Fully Connected Layer Regression CNN Model

print('using XZ and YZ views for training the model on `z` coordinate')

# instantiate the model
model_regCNN_xz = Sequential()
#model_regCNN_yz = Sequential()
# add two fully connected 2-dimensional convolutional layers for the XZ and YZ views
model_regCNN_xz.add(Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1),
                           input_shape=(100, 80, 1), activation='relu'))
model_regCNN_yz.add(Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1),
                           input_shape=(100, 80, 1), activation='relu'))

# specify 2-dimensional pooling
model_regCNN_xz.add(MaxPool2D(pool_size=(2, 2)))
model_regCNN_yz.add(MaxPool2D(pool_size=(2, 2)))
# flatten the datasets
model_regCNN_xz.add(Flatten())
model_regCNN_yz.add(Flatten())
# add dense layers for each view. 256 neurons per layer
model_regCNN_xz.add(Dense(256, activation='relu'))
model_regCNN_xz.add(Dense(256, activation='relu'))
model_regCNN_xz.add(Dense(256, activation='relu'))
model_regCNN_xz.add(Dense(256, activation='relu'))
model_regCNN_xz.add(Dense(256, activation='relu'))
model_regCNN_xz.add(Dense(256, activation='relu'))
#model_regCNN_xz.add(AveragePooling2D(pool_size=(2,2)))


model_regCNN_yz.add(Dense(256, activation='relu'))
model_regCNN_yz.add(Dense(256, activation='relu'))
model_regCNN_yz.add(Dense(256, activation='relu'))
model_regCNN_yz.add(Dense(256, activation='relu'))
model_regCNN_yz.add(Dense(256, activation='relu'))
model_regCNN_yz.add(Dense(256, activation='relu'))
#model_regCNN_yz.add(AveragePooling2D(pool_size=(2,2)))

#class for a dense layer
n_classes = 1

#trying to output each model first (xz and yz)
output_xz=Dense(n_classes)(model_regCNN_xz.output)
output_yz=Dense(n_classes)(model_regCNN_yz.output)

#finding the mean for each output
#model_mean_output_xz=tf.reduce_mean(output_xz, axis=0)
#model_mean_output_yz=tf.reduce_mean(output_yz, axis=0)
model_regCNN_xz_Dense= Dense(n_classes)(model_regCNN_xz.output)


# To find the average of each (XZ and YZ) output, comment line 415 and 414 and uncomment line 424, 425, 429 and 430
#output_xz=model_regCNN_xz.output
#averaging_xz = Average()(output_xz) #This is averaging the output of xz
#extracted_value_xz= Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(output_xz)   #This is to extract only the z value from the xz coordinate


#output_yz=model_regCNN_yz.output
#averaging_yz = Average()(output_yz) #This is averaging the output of yz
#extracted_value_yz= Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(output_yz)   #This is to extract only the z value from the yz coordinate
#model_regCNN_concat= concatenate([model_regCNN_xz.output, model_regCNN_yz.output], axis=-1)

#model_regCNN_concat= concatenate([extracted_value_xz, extracted_value_yz], axis=-1)
#model_regCNN_concat= Average()([output_xz, output_yz])
#print("Shape of model_regCNN_concat:", model_regCNN_concat.shape)
#print("Shape of model_regCNN_xz output :", model_regCNN_xz.output.shape)
#print("Shape of model_regCNN_yz output :", model_regCNN_yz.output.shape)
#model_regCNN_Average= AveragePooling2D(pool_size=(2,2))(model_regCNN_xz.output, model_regCNN_yz.output], axis=-1)

#Abdul adding this flatten layer to make the concat a 1D before the Dense layer
#model_regCNN_mean_flatten = Flatten()(model_regCNN_mean)
#input_layer_flatten = Input(shape=(100, 80, 1))

#print("Shape of flatten_input:", flatten_input.shape)
#print("Shape of model_regCNN_mean_flatten:", model_regCNN_mean_flatten.shape)
#print("Shape of model_regCNN_concat:", model_regCNN_concat.shape)

#number of class output
#print("Shape of model_regCNN_concat before  Dense:", model_regCNN_concat.shape)
#model_regCNN_concat= Dense(n_classes)(model_regCNN_concat)
#print("Shape of model_regCNN_concat after Dense:", model_regCNN_concat.shape)
#model_regCNN = Model(inputs=[model_regCNN_xz.input, model_regCNN_yz.input], outputs=model_regCNN_concat)


#This is for training the XZ and YZ separately on the Z-Planes
model_regCNN_YZ = Model(inputs=[model_regCNN_yz.input], outputs=output_yz)
model_regCNN_XZ = Model(inputs=[model_regCNN_xz.input], outputs=output_xz)
# compile the concatenated model
#model_regCNN.compile(loss='logcosh',
#                     optimizer='adam',
#                     metrics=['mse'])  # loss was 'mse' then 'mae'
# print a summary of the model

#Compiling for the individual training
model_regCNN_YZ.compile(loss='logcosh',
                     optimizer='adam',
                     metrics=['mse'])  # loss was 'mse' then 'mae
print(model_regCNN_YZ.summary())
model_regCNN_XZ.compile(loss='logcosh',
                     optimizer='adam',
                     metrics=['mse'])  # loss was 'mse' then 'mae
print(model_regCNN_XZ.summary())

# x-coordinate system.
date = date.today()
start = time.time()
# todo: do we need to include both y (labels) here? or just one?
#model_regCNN.fit(x=[XZ_train_feature_x, YZ_train_feature_x], y=[XZ_train_label_y, YZ_train_label_y], epochs=args.epochs, batch_size=32, verbose=1, )  #This line is for combining the XZ and YZ for the training

#Training XZ and YZ separately
model_regCNN_XZ.fit(x=[XZ_train_feature_x], y=[XZ_train_label_y], epochs=args.epochs, batch_size=32, verbose=1, ) #This is for XZ maps
model_regCNN_YZ.fit(x=[YZ_train_feature_x], y=[YZ_train_label_y], epochs=args.epochs, batch_size=32, verbose=1, ) #This is for the YZ maps
stop = time.time()
print('Time to train: ', stop - start)

# the default output name
outputName = 'training_{}epochs_{}_{}_{}_Z_{}'.format(args.epochs, args.detector, args.horn, args.flux, date)
save_model_dir = '/homes/m962g264/wsu_Nova_Vertexer/output/New-trained-model/'
#model_regCNN.save(save_model_dir + 'model_{}.h5'.format(outputName))
model_regCNN_YZ.save(save_model_dir + 'model_YZ_{}.h5'.format(outputName))
model_regCNN_XZ.save(save_model_dir + 'model_XZ_{}.h5'.format(outputName))
print('saved model to: ', save_model_dir + 'model_{}.h5'.format(outputName))
# Items in the model file: <KeysViewHDF5 ['model_weights', 'optimizer_weights']>

# Save the metrics to a dataframe
#metrics = pd.DataFrame(model_regCNN.history.history)
metrics_xz = pd.DataFrame(model_regCNN_XZ.history.history)
metrics_yz = pd.DataFrame(model_regCNN_YZ.history.history)

print('METRICS:')
print(metrics_xz.head())
#print(metrics_yz.head())
save_metric_dir = '/homes/m962g264/wsu_Nova_Vertexer/output/metrics/'
metrics_xz.to_csv(save_metric_dir + '/metrics_xz_{}.csv'.format(outputName))
metrics_yz.to_csv(save_metric_dir + '/metrics_yz_{}.csv'.format(outputName))
print('saved metrics to: ', save_metric_dir + '/metrics_{}.csv'.format(outputName))


plot_dir = '/homes/m962g264/wsu_Nova_Vertexer/output/plots/New-trained-plots/{}'.format(outputName)

if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)
    print('created plot dir: ', plot_dir)

# plot the training loss (among other metrics, for edification)
plt.figure(figsize=(12, 8))
#plt.plot(metrics[['loss']])  # 'accuracy', 'mse']])  # TODO: add more metrics to this training.. accuracy, entropy. etc.
plt.plot(metrics_xz[['loss']], label='XZ')
plt.plot(metrics_yz[['loss']], label='YZ')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training Metrics')
#plt.legend(['loss'], loc='upper right')  # 'accuracy', 'mse'], loc='upper right')
plt.legend(['XZ-loss','YZ-loss'], loc='upper right')  # 'accuracy', 'mse'], loc='upper right')
plt.tight_layout()
plt.show()
for ext in ['png', 'pdf']:
    plt.savefig('{}/loss_{}.{}'.format(plot_dir, outputName, ext))


# Evaluate the test set
start_eval = time.time()
print('Evaluation on the test set...')
#evaluation = model_regCNN.evaluate(x=[XZ_test_feature_x, YZ_test_feature_x], y=[XZ_test_label_y, YZ_test_label_y], verbose=1) #This is for evaluating the combine maps for training
evaluation_xz = model_regCNN_XZ.evaluate(x=[XZ_test_feature_x], y=[XZ_test_label_y], verbose=1)
evaluation_yz = model_regCNN_YZ.evaluate(x=[YZ_test_feature_x], y=[YZ_test_label_y], verbose=1)  
stop_eval = time.time()
print('Test Set Evaluation: {}'.format(evaluation_xz))
print('Test Set Evaluation: {}'.format(evaluation_yz))
print('Evaluation time: ', stop_eval - start_eval)
print('Done! Training Complete!')
