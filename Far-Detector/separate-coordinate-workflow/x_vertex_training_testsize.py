#!/usr/bin/env python
# coding: utf-8

# # KDE & ML Based NuMu Event Reconstruction

# ### Import Classes & Modules

#     Ensure all required python classes are installed prior to progress.

# This script performs CNN training on the x-coordinate of the detector vertex.
# It uses the pixelmaps to as the features.
# However, the true vertex (label) is in cm, so we need to convert the true vertex to pixels.

# To run this training script:
#  $ $PY37 x_vertex_training_testsize.py --detector <ND> --horn <FHC> --flux <nonswap> --epochs <200>


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

def two_scales(ax1, time, data1, data2, c1, c2):
    ax2 = ax1.twinx()
    ax1.plot(time, data1, color=c1)
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('exp')
    ax2.plot(time, data2, color=c2)
    ax2.set_ylabel('sin')
    return ax1, ax2


# Change color of each axis
def color_y_axis(ax, color):
    """Color your axes."""
    for t in ax.get_yticklabels():
        t.set_color(color)



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


# TODO: I cannot get the function to work, so do it within the script
# # Conversion from cm (vtx.{x,y,z}) to pixels (cvnmaps) for "standard"-type datasets
# def convXdp(x, fx, detector):
#     if detector == 'ND':
#         return x / 3.99 - fx + 48
#     elif detector == 'FD':
#         return x / 3.97 - fx + 192
#     else:
#         print('NO DETECTOR. No coordinate conversion.')
#         return None


# def convYdp(y, fy, detector):
#   if detector == 'ND':
#     return y/3.97 - fy + 47
#   elif detector== 'FD':
#     return y/3.97 - fy + 191
#   else:
#     print('NO DETECTOR')
#     return None
# def convZdp(z, fz, detector):
#   if detector == 'ND':
#     return z/6.61 - fz
#   elif detector== 'FD':
#     return z/6.664 - fz
#   else:
#     print('NO DETECTOR')
#     return None


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

# Using the pre-processed files for the testsize files.
train_path = '/home/k948d562/output/wsu-vertexer/training/{}-Nominal-{}-{}/testsize/'.format(args.detector,
                                                                                            args.horn,
                                                                                            args.flux)
print('train_path: ', train_path)
# print this out just purely for verifying what exactly is in the files
train_file = h5py.File(train_path + os.listdir(train_path)[0], 'r')
print('keys in the train file....', train_file.keys())

# number of events in each file
events_total_training = 0
count_training_files = 0

cvnmap, vtx_x, firstcellx = [], [], []
for h5_filename in os.listdir(train_path):
    if os.path.isdir(h5_filename):
        continue

    print('Reading cvnmap and vtx arrays. Processing file... {} '.format(len(os.listdir(train_path))))
    f = h5py.File(train_path + h5_filename, 'r')
    cvnmap.append(f['cvnmap'][:])
    vtx_x.append(f['vtx.x'][:])
    firstcellx.append(f['firstcellx'][:])

    # this should be ~O(millions) of events for each file...
    events_per_file_train = len(f['vtx.x'][:])  # count events in each file # can try: sum(len(x) for x in multilist)
    events_total_training += events_per_file_train
    print('events in file: ', events_per_file_train)
    print('total events: ', events_total_training)
    count_training_files += 1


# Convert data into np arrays
cvnmap = np.array(cvnmap)
vtx_x = np.array(vtx_x)

# need to convert the cells, as Reco. group advises.
for entry in range(len(firstcellx)):
    entry += 40   # this is the +40 bit Erin also has. Unclear why.
firstcellx = np.array(firstcellx, dtype='int')
firstcellx -= 40  # this is the +40 bit Erin also has. Unclear why.
# TODO: this does not seem right. the X position is always on the edge or not visible, I don't think this is actually doing anything.....

print('Validation files read successful')
print('Validation files read successful:')
print('loaded {} files, and {} total events.'.format(count_training_files, events_total_training))


# Create the vertex info in pixel map coordinates:
vtx_x_pixelmap = vtx_x
if type(vtx_x_pixelmap) is np.ndarray:
    vtx_x_pixelmap = np.array(vtx_x_pixelmap)  # make sure its an array if it isn't already
# This function isn't working, need to figure out why.
# convXdp(vtx_x_pixelmap, firstcellx, args.detector)

# Convert the true vertex to pixelmap coordinates
if args.detector == 'ND':
    print('ND detector not setup yet. Exiting')
    exit()
elif args.detector == 'FD':
    vtx_x_pixelmap = vtx_x / 3.99 - firstcellx + 192
    # vtx_y_pixelmap = vtx_y / 3.97 - firstcelly + 191
    # vtx_z_pixelmap = vtx_z / 6.664 - firstplane
else:
    print('NO DETECTOR. No coordinate conversion.')
    exit(1)

print('vtx_x_pixelmap: ', vtx_x_pixelmap.shape)
print('vtx_x_pixelmap[0][1] = ', vtx_x_pixelmap[0][1])
assert vtx_x_pixelmap[0][1] == 75.02478790283203, "vtx_x_pixelmap[0][1] should be 75.02478790283203"

# TODO: Unclear if this is needed only for this "testsize" training, or for all training ( i.e. all files)
# NOTE: must re-shape the vtx_x array to match the cvnmap_norm_resh_xz array
print('vtx.shape: ', vtx_x.shape)
vtx_x_resh = np.reshape(vtx_x, (events_total_training, count_training_files))
print('vtx_x_resh: ', vtx_x_resh.shape)
vtx_x_pixelmap_resh = np.reshape(vtx_x_pixelmap, (events_total_training, count_training_files))

# Print out useful info about the shapes of the arrays
print('-------------------')
print('Format of the arrays: (file_idx, event_idx, pixel_idx)')
print('the pixel_idx is for cvnmaps only')
print('-------------------')
# we expect file_idx to be just 1 here, because we are only loading 1 file for testing.
print('cvnmap.shape: ', cvnmap.shape)
print('vtx_x.shape: ', vtx_x.shape)
print('last 5 entries of vtx_x from first file', vtx_x[0][:4])
print('firstcellx.shape: ', firstcellx.shape)
print('-------------------')

# split normalized cvnmap into reshaped events with multi-views: XZ and YZ.
print('split normalized cvnmap into reshaped events with multi-views....')
a, b, cvnmap_norm_resh_xz, cvnmap_norm_resh_yz = [], [], [], []
cvnmap_resh = []
# training CVN map view split ##
# want to split this by the events amount in each file

file_counter = event_counter = 0
for h5_filename in os.listdir(train_path):
    print('looping through file....{} of {}'.format(file_counter, len(os.listdir(train_path))))
    a = cvnmap[file_counter]
    print('cvnmap.shape[file_counter] = ', a.shape[file_counter])
    print('Processing train cvnmap file {} of {}'.format(file_counter + 1, (len(os.listdir(train_path)))))
    event_counter = 0
    while event_counter < a.shape[0]:
        # b = cvnmap[file_counter][event_counter].reshape(2, 100, 80)  # HAVE: 501409x16000=8022544000. Want: 2 views, 100x80 = 16000
        b = a[event_counter].reshape(2, 100, 80)
        cvnmap_norm_resh_xz.append(b[0])
        cvnmap_norm_resh_yz.append(b[1])
        cvnmap_resh.append(b)
        event_counter += 1
    file_counter += 1

assert count_training_files == file_counter, "count_training_files must equal file_counter: {}".format(file_counter)
assert events_total_training == event_counter, "total_events_training must equal event_counter: {}".format(event_counter)


# Convert the XZ and YZ views to np arrays
cvnmap_norm_resh_xz = np.array(cvnmap_norm_resh_xz)  # xz views only
print('cvnmap_norm_resh_xz.shape: ', cvnmap_norm_resh_xz.shape)
cvnmap_norm_resh_yz = np.array(cvnmap_norm_resh_yz)  # yz views only
print('cvnmap_norm_resh_xy.shape: ', cvnmap_norm_resh_yz.shape)

cvnmap_resh = np.array(cvnmap_resh)  # all views
print('cvnmap_resh.shape: ', cvnmap_resh.shape)

# GPU Test
# see if GPU is recognized
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print('is built with cuda?: ', tf.test.is_built_with_cuda())
print('is gpu available?: ', tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
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

# In[ ]:

# split the data into training and testing sets for the unified view
x_train, x_test, y_train, y_test = train_test_split(cvnmap_resh, vtx_x_pixelmap_resh, test_size=0.25, random_state=101)

# split the data into training and testing sets

# XZ view. Trains on the x-coordinate
X1_train, X1_test, y1_train, y1_test = train_test_split(cvnmap_norm_resh_xz, vtx_x_pixelmap_resh, test_size=0.25, random_state=101)

# YZ view.
X2_train, X2_test, y2_train, y2_test = train_test_split(cvnmap_norm_resh_yz, vtx_x_pixelmap_resh, test_size=0.25, random_state=101)

# In[ ]:


# add one more dimension to let the CNN know we are dealing with one color dimension
x1_train = X1_train.reshape(X1_train.shape[0], 100, 80, 1)
x1_test = X1_test.reshape(X1_test.shape[0], 100, 80, 1)
x2_train = X2_train.reshape(X2_train.shape[0], 100, 80, 1)
x2_test = X2_test.reshape(X2_test.shape[0], 100, 80, 1)

x_train = x_train.reshape(x_train.shape[0], 2, 100, 80, 1)
# batch_size,width,height,color_channels

# print what they look like and the true vertex

# In[ ]:


# ### MultiView Fully Connected Layer Regression CNN Model

# In[ ]:


# instantiate the models
model_regCNN_xz = Sequential()
model_regCNN_yz = Sequential()
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
model_regCNN_yz.add(Dense(256, activation='relu'))
model_regCNN_xz.add(Dense(256, activation='relu'))
model_regCNN_yz.add(Dense(256, activation='relu'))
model_regCNN_xz.add(Dense(256, activation='relu'))
model_regCNN_yz.add(Dense(256, activation='relu'))
model_regCNN_xz.add(Dense(256, activation='relu'))
model_regCNN_yz.add(Dense(256, activation='relu'))
model_regCNN_xz.add(Dense(256, activation='relu'))
model_regCNN_yz.add(Dense(256, activation='relu'))
model_regCNN_xz.add(Dense(256, activation='relu'))
model_regCNN_yz.add(Dense(256, activation='relu'))
# no. of classes (output)
n_classes = 1
# tf concatenate the models
# model_regCNN_concat = concatenate([model_regCNN_xz.output], axis=-1)
# model_regCNN_concat = Dense(n_classes)(model_regCNN_concat)
model_regCNN = Model(inputs=[model_regCNN_xz.input], outputs=model_regCNN_xz.output)
# compile the concatenated model
model_regCNN.compile(loss='logcosh',
                     optimizer='adam',
                      metrics=[
                            'accuracy',
                          'mse',
                          # metrics.MeanSquaredError(name='MSE'),
                          # metrics.MeanAbsoluteError(name='MAE'),
                          # metrics.AUC(name='AUC'),
                      ])  # loss was 'mse' then 'mae'
# print a summary of the model
print(model_regCNN.summary())


# x-coordinate system.
date = date.today()
start = time.time()
model_regCNN.fit(x=[x_train], y=y_train, epochs=args.epochs, batch_size=32, verbose=1,)
stop = time.time()
print('Time to train: ', stop - start)

# the default output name
outputName = 'testsize_{}epochs_{}_{}_{}_X'.format(args.epochs, args.detector, args.horn, args.flux)

save_model_dir = '/home/k948d562/output/wsu-vertexer/trained-models/'
model_regCNN.save(save_model_dir + 'model_{}.h5'.format(outputName))
print('saved model to: ', save_model_dir + 'model_{}.h5'.format(outputName))
# Items in the model file: <KeysViewHDF5 ['model_weights', 'optimizer_weights']>

# Save the metrics to a dataframe
metrics = pd.DataFrame(model_regCNN.history.history)
print(metrics.head())
metrics.to_hdf(save_model_dir + '/metrics_{}.h5'.format(outputName), key='metrics')
print('saved metrics to: ', save_model_dir + '/metrics_{}.h5'.format(outputName))

# plt.savefig('/home/m962g264/wsu_Nova_Vertexer/output/plots/loss_plots/X-train-{}_{}_{}_-E220-loss_2.png')

# model evaluation with logcosh 200 epoch
plot_dir = '/home/k948d562/plots/ml-vertexing-plots/wsu-vertexer/small-scale-testing/{}'.format(outputName)

if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)
    print('created directory: ', plot_dir)

# plot the training loss (among other metrics, for edification)
plt.figure(figsize=(12, 8))
plt.plot(metrics[['loss', 'accuracy', 'mse']])  # TODO: add more metrics to this training... accuracy, entropy... etc.
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Metrics')
plt.legend(['loss', 'accuracy', 'mse'], loc='upper right')
plt.tight_layout()
plt.show()
for ext in ['png', 'pdf']:
    plt.savefig('{}/train_loss_testsize_{}epochs_{}_{}_{}_{}_X.{}'.format(plot_dir, args.epochs, args.detector, args.horn, args.flux, date, ext))


# # Create axes
# fig, ax1 = plt.subplots(1,1, figsize=(12,8))
# ax1, ax1a = two_scales(ax1, t, s1, s2, 'r', 'b')
# ax1.set_xlabel('Epoch')
# color_y_axis(ax1, 'r')
# color_y_axis(ax1a, 'b')


# Evaluate the test set
start_eval = time.time()
print('Evaluation on the test set...')
evaluation = model_regCNN.evaluate(x=[x1_test, x2_test], y=y1_test, verbose=1)
stop_eval = time.time()
print('Test Set Evaluation: {}'.format(evaluation))
print('Evaluation time: ', stop_eval - start_eval)
