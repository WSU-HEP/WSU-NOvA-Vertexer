#!/usr/bin/env python
# coding: utf-8

# testing of the joint XYZ training together.

# To run this training script:
#  $ $PY37 xyz_vertex_training_testsize.py --data_train_path --epochs <20>
# generally: $OUTPUT/training/FD-Nominal-FHC-Fluxswap-testsize/

# ML Vtx utils
import utils.iomanager as io
import utils.model
import utils.plot
import utils.data_processing as dp

import argparse
from datetime import date
import numpy as np
import os
import pandas as pd
from tensorflow.keras.optimizers import Adam  # optimizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
import time

########### begin main script ###########

# collect the arguments for this macro. the horn and swap options are required.
parser = argparse.ArgumentParser()
parser.add_argument("--data_train_path", help="path to train data", type=str, required=True)
parser.add_argument("--epochs", help="number of epochs", default=20, type=int)
args = parser.parse_args()

# want the final dir, and extract its strings.
train_path = args.data_train_path
train_path_dir = os.path.basename(os.path.normpath(train_path))
det, horn, flux = io.IOManager.get_det_horn_and_flux_from_string(train_path_dir)

print('WARNING: You are doing testing, be sure you have the correct path to train data! ')
print('data_train_path: ', train_path)
datasets, total_events, total_files = io.load_data(train_path, False)

# convert the lists to numpy arrays
print('========================================')
print('Converting lists to numpy arrays...')
datasets = {key: np.array(datasets[key]) for key in datasets}

# trust that they are all numpy.arrays AND have the right shape
for key in datasets:
    print(key)
    dp.Debug(datasets[key]).printout_type()

print('========================================')
print('Addressing the bug in the h5 files. Converting firstcellx, firstcelly, firstplane to int type...')
datasets['firstcellx'] = dp.DataCleaning(datasets['firstcellx'], 'x').remove_unsigned_ints()
datasets['firstcelly'] = dp.DataCleaning(datasets['firstcelly'], 'y').remove_unsigned_ints()
datasets['firstplane'] = dp.DataCleaning(datasets['firstplane'], 'z').remove_unsigned_ints()
# Let's now check to make sure we don't have any gigantic numbers, from unsigned int's.
# we already know this is the common number we see if wew do this wrong...so check against it.
file_idx = 0
for i in [datasets['firstcellx'], datasets['firstcelly']]:
    event = 0
    if i[file_idx][event] > 4294967200:  # this a large number, just below the max value of an unsigned int, which should trigger
        print('i: ', i)
    event += 1

print('========================================')
print('Converting the vertex coordinates into pixelmap coordinates for the network...')
# Create the vertex info in pixel map coordinates:
# convert the vertex location (detector coordinates) to pixel map coordinates
vtx_x_pixelmap = dp.ConvertFarDetCoords(det, 'x').convert_fd_vtx_to_pixelmap(datasets['vtx.x'], datasets['firstcellx'])
vtx_y_pixelmap = dp.ConvertFarDetCoords(det, 'y').convert_fd_vtx_to_pixelmap(datasets['vtx.y'], datasets['firstcelly'])
vtx_z_pixelmap = dp.ConvertFarDetCoords(det, 'z').convert_fd_vtx_to_pixelmap(datasets['vtx.z'], datasets['firstplane'])
print('Done converting.')

print('========================================')
# Print out useful info about the shapes of the arrays
print('========================================')
print('Useful info about the shapes of the arrays:')
print('-------------------')
print('Format of the arrays: (file_idx, event_idx, pixel_idx)......')
print('the pixel_idx is for cvnmaps only')
print('-------------------')
# we set file_idx manually
print('cvnmap.shape: ', datasets['cvnmap'].shape)
print('vtx_x.shape: ', datasets['vtx.x'].shape)
print('vtx_x_pixelmap.shape: ', vtx_x_pixelmap.shape)
print('firstcellx.shape: ', datasets['firstcellx'].shape)
print('-------------------')

# this array should be something like: (2, 10000, 16000)
cvnmap = datasets['cvnmap']
print('-------------------')
print('to access FILE, index in array is: (cvnmap.shape[0]) = ', cvnmap.shape[0], 'files.')  # first dimension is the file
print('to access EVENT, index in array is: (cvnmap.shape[1]) = ', cvnmap.shape[1], 'events')  # second dimension is the events
print('to access PIXELS, index in array is: (cvnmap.shape[2]) = ', cvnmap.shape[2], 'pixels')  # third dimension is the pixelmap
print('-------------------')

print('========================================')
# reshape the pixels into 2 (100,80) views: XZ and YZ.
print('reshape the pixels into 2 (100,80) views: XZ and YZ.....')


cvnmap_xz = []
cvnmap_yz = []
total_event_counter = 0
file_counter = 0

train_files = int(cvnmap.shape[0])

for file_counter, h5_filename in enumerate(range(train_files)):
    print(f"Processing train cvnmap file {file_counter + 1} of {train_files}")

    print(f"Reshaping all {cvnmap.shape[1]} events into correct pixel map size (100, 80)")
    for event in range(cvnmap.shape[1]):
        reshaped_maps = cvnmap[file_counter][event].reshape(2, 100, 80)
        cvnmap_xz.append(reshaped_maps[0])
        cvnmap_yz.append(reshaped_maps[1])

        total_event_counter += 1

    # might work for the (4,) index
    # reshaped_maps = cvnmap[file_counter].reshape(-1, 2, 100, 80)  # Reshape all events at once
    # Split into XZ and YZ views
    # cvnmap_xz = np.concatenate([cvnmap_xz, reshaped_maps[:, 0]], axis=0)  #add to the end of the array
    # cvnmap_yz = np.concatenate([cvnmap_yz, reshaped_maps[:, 1]], axis=0)
cvnmap_xz = np.array(cvnmap_xz)
cvnmap_yz = np.array(cvnmap_yz)

# Validate results
assert train_files == (file_counter + 1), f"File count mismatch: {file_counter + 1} files processed."
assert cvnmap.shape[1] * cvnmap.shape[0] == total_event_counter, f"Event count mismatch: {total_event_counter} != {cvnmap.shape[1] * cvnmap.shape[0]}."

print('========================================')
# must re-shape the vtx_x_pixelmap array to match the cvnmap_resh_xz array
# I.e. --> remove the first dimension [0], the file.
print('Reshape vtx_x_pixelmap, vtx_y_pixelmap, vtx_z_pixelmap to be of just the events...')
print('Before, vtx_x_pixelmap: ', vtx_x_pixelmap.shape)
vtx_x_pixelmap = vtx_x_pixelmap.reshape(total_events)
vtx_y_pixelmap = vtx_y_pixelmap.reshape(total_events)
vtx_z_pixelmap = vtx_z_pixelmap.reshape(total_events)
print('After, vtx_x_pixelmap: ', vtx_x_pixelmap.shape)

# combine for Nx3 array: [X, Y, Z]
vtx_coords = np.stack((vtx_x_pixelmap, vtx_y_pixelmap, vtx_z_pixelmap), axis=-1)

# Drop the events that are outside the cvnmap!
# Apply to both features & labels.
# dictionary of {keep; np.array, drop: array}
keep_drop_evts = dp.DataCleaning.sort_events_with_vtxs_outside_cvnmaps(vtx_coords)
vtx_coords = vtx_coords[keep_drop_evts['keep']]
cvnmap_xz = cvnmap_xz[keep_drop_evts['keep']]
cvnmap_yz = cvnmap_yz[keep_drop_evts['keep']]
assert cvnmap_xz.shape[0] == cvnmap_yz.shape[0] == vtx_coords.shape[0]

# reduce the memory size for features & labels. Four decimal places is sufficient.
vtx_coords = vtx_coords.astype(np.float16)
cvnmap_xz = cvnmap_xz.astype(np.float16)
cvnmap_yz = cvnmap_yz.astype(np.float16)


# #### Prepare the Training & Test Sets
# split the data into training (+ val) and testing sets
# Creates dictionaries for:
# Train -- for fit(). Val -- for fit(). Test -- for evaluate()
data_train, data_val, data_test = utils.model.Config.create_test_train_val_datasets(cvnmap_xz, cvnmap_yz, vtx_coords)

print('========================================')
utils.model.Hardware.check_gpu_status()
print('========================================')

# ### MultiView Fully Connected Layer Regression CNN Model

print('using XZ and YZ views (two models) for `x`, `y`, and `z` coordinates......')
model_xz = utils.model.Config.create_conv2d_branch_model_single_view()
model_yz = utils.model.Config.create_conv2d_branch_model_single_view()

# create final model, join the XZ and YZ, add dense layers, and concat to 3.
model_regCNN = utils.model.Config.assemble_model_output(model_xz, model_yz)

# compile the concatenated model
utils.model.Config.compile_model(model_regCNN)
# print a summary of the model
print(model_regCNN.summary())

# normalize w MinMaxScaler -- don't fit() the val,test data.
data_train, data_val, data_test = utils.model.Config.transform_data(data_train,
                                                                    data_val,
                                                                    data_test)

dp.print_input_data(data_train, data_test, data_val)

history = utils.model.train_model(model_regCNN,
                                  data_train,
                                  data_val,
                                  args.epochs,
                                  32)

# save the history to a pandas dataframe, will append the evaluate() output to this dataframe
metrics = pd.DataFrame(history.history)

# the default output name
output_name = 'testsize_{}epochs_{}_{}_{}_{}_XYZ'.format(args.epochs, det, horn, flux, date.today())

# save the model
save_model_dir = '/home/k948d562/output/trained-models/'
model_regCNN.save(save_model_dir + 'model_{}.h5'.format(output_name))
print('saved model to: ', save_model_dir + 'model_{}.h5'.format(output_name))
# Items in the model file: <KeysViewHDF5 ['model_weights', 'optimizer_weights']>

# outdir for testsize is different.
save_metric_dir = f'/home/k948d562/output/metrics/small-scale-testing/{output_name}'

# Evaluate the test set
print('METRICS:')
evaluation = utils.model.evaluate_model(model_regCNN,
                                        data_train,
                                        data_test,
                                        save_metric_dir)
print(metrics.head())
metrics.to_csv(save_metric_dir + '/metrics_{}.csv'.format(output_name), index_label='epoch')
print('saved metrics to: ', save_metric_dir + '/metrics_{}.csv'.format(output_name))
# NOTE: evaluation only returns ONE number for each metric , and one for the loss, so just write to txt file.

plot_dir = '/home/k948d562/plots/ml-vertexing-plots/small-scale-testing/'
utils.plot.plot_training_metrics(history, plot_dir, 'train_metrics_' + output_name)
print('Done.')
