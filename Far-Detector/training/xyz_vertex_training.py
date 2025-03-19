#!/usr/bin/env python
# coding: utf-8

# testing of the joint XYZ training together.

# To run this training script:
#  $ $PY37 xyz_vertex_training.py --data_train_path --epochs <200>

import argparse
import os

from datetime import date
import numpy as np
import pandas as pd

# ML Vtx utils
import utils.iomanager as io
import utils.model
import utils.plot
import utils.data_processing as dp
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

print('WARNING: You are doing full training, be sure you have the correct path to train data! ')
print('data_train_path: ', train_path)
datasets, total_events, total_files = io.load_data(train_path, False)

print('========================================')
# Convert all lists in datasets to NumPy arrays with dtype=object (while keeping the arrays within from each file arrays)
datasets = {key: np.array(value, dtype=object) for key, value in datasets.items()}
print('to access FILE, index in array is: (cvnmap.shape[0] = ', datasets['cvnmap'].shape[0], 'files.)')  # first dimension is the file

# trust that they are all numpy.arrays AND have the right shape
for key in datasets:
    print(key)
    dp.Debug(datasets[key]).printout_type()

print('========================================')
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
datasets['firstcellx'] = dp.ConvertFarDetCoords(det, 'x').convert_fd_vtx_to_pixelmap(datasets['vtx.x'], datasets['firstcellx'])
datasets['firstcelly'] = dp.ConvertFarDetCoords(det, 'y').convert_fd_vtx_to_pixelmap(datasets['vtx.y'], datasets['firstcelly'])
datasets['firstplane'] = dp.ConvertFarDetCoords(det, 'z').convert_fd_vtx_to_pixelmap(datasets['vtx.z'], datasets['firstplane'])
# operate in place, and change key name to 'vtx_i_pixelmap'
datasets['vtx_x_pixelmap'] = datasets.pop('firstcellx')
datasets['vtx_y_pixelmap'] = datasets.pop('firstcelly')
datasets['vtx_z_pixelmap'] = datasets.pop('firstplane')

# concatenate/flatten each vertex array
print("Reshape datasets['vtx_x_pixelmap'], datasets['vtx_y_pixelmap'], datasets['vtx_z_pixelmap'] to be of just the events...")
print("Before, datasets['vtx_x_pixelmap']: ", datasets['vtx_x_pixelmap'].shape)
datasets['vtx_x_pixelmap'] = np.concatenate(datasets['vtx_x_pixelmap'], axis=0)
datasets['vtx_y_pixelmap'] = np.concatenate(datasets['vtx_y_pixelmap'], axis=0)
datasets['vtx_z_pixelmap'] = np.concatenate(datasets['vtx_z_pixelmap'], axis=0)
print("After, datasets['vtx_x_pixelmap']: ", datasets['vtx_x_pixelmap'].shape)

# combine for Nx3 array: [X, Y, Z]
vtx_coords = np.stack((datasets['vtx_x_pixelmap'], datasets['vtx_y_pixelmap'], datasets['vtx_z_pixelmap']), axis=-1)
print('Done converting.')


print('========================================')
print('Remove the file index, and reshape the pixels into 2 (100,80) views: XZ and YZ.....')
# this array should be something like: [fileIdx](events_in_file, 16000)
# reshape the pixels into 2 (100,80) views: XZ and YZ.
# NOTE: this works regardless how many events are in each file...
datasets['cvnmap'] = np.concatenate([entry.reshape(-1, 16000) for entry in datasets['cvnmap']], axis=0)  # remove the file index (8,)
datasets['cvnmap'] = datasets['cvnmap'].reshape(total_events, 100, 80, 2)  # divide into the pixels into the map
datasets['cvnmap'] = datasets['cvnmap'].reshape(datasets['cvnmap'].shape[0], datasets['cvnmap'].shape[1], datasets['cvnmap'].shape[2], datasets['cvnmap'].shape[3], 1) # add the color channel
datasets['cvnmap'] = datasets['cvnmap'].astype(np.float16)
# these are views (without copies) to save memory
cvnmap_xz = datasets['cvnmap'][:, :, :, 0].reshape(datasets['cvnmap'].shape[0], 100, 80, 1)  # Extract the XZ view and reshape
cvnmap_yz = datasets['cvnmap'][:, :, :, 1].reshape(datasets['cvnmap'].shape[0], 100, 80, 1)  # Extract the YZ view and reshape

print(datasets['cvnmap'].shape)
print('XZ: ', cvnmap_xz.shape)
print('YZ: ', cvnmap_yz.shape)
print('========================================')

# Drop the events that are outside the cvnmap!
# Apply to both features & labels.
# dictionary of {keep; np.array, drop: array}
keep_drop_evts = dp.DataCleaning.sort_events_with_vtxs_outside_cvnmaps(vtx_coords)
vtx_coords = vtx_coords[keep_drop_evts['keep']]
cvnmap_xz = cvnmap_xz[keep_drop_evts['keep']]
cvnmap_yz = cvnmap_yz[keep_drop_evts['keep']]
assert cvnmap_xz.shape[0] == cvnmap_yz.shape[0] == vtx_coords.shape[0]

cvnmap_xz = cvnmap_xz.astype(np.uint8)
cvnmap_yz = cvnmap_yz.astype(np.uint8)
vtx_coords = vtx_coords.astype(np.float16)


# #### Prepare the Training & Test Sets
# XZ view and YZ view. Train on both views; predict all 3 coordinates.
# Train -- for fit(). Val -- for fit(). Test -- for evaluate()
data_train, data_val, data_test = utils.model.Config.create_test_train_val_datasets(cvnmap_xz, cvnmap_yz, vtx_coords)


print('========================================')
utils.model.Hardware.check_gpu_status()
print('========================================')

# ### MultiView Fully Connected Layer Regression CNN Model

print('using XZ and YZ views to learn `x`, `y`, and `z` coordinates......')
# define separate models for each view
model_xz = utils.model.Config.create_conv2d_branch_model_single_view()
model_yz = utils.model.Config.create_conv2d_branch_model_single_view()

# create final model, join the XZ and YZ, add dense layers, and condense to 3.
model_regCNN = utils.model.Config.assemble_model_output(model_xz, model_yz)

utils.model.Config.compile_model(model_regCNN)
print(model_regCNN.summary())


dp.print_input_data(data_train, data_test, data_val)

# fit() the model.
history = utils.model.train_model(model_regCNN,
                                  data_train,
                                  data_val,
                                  args.epochs,
                                  128)

# save the history to a pandas dataframe, will append the evaluate() output to this dataframe
metrics = pd.DataFrame(history.history)

# the default output name
output_name = f'{args.epochs}epochs_{det}_{horn}_{flux}_{date.today()}_XYZ'

# save the model
save_model_dir = f'/home/{io.USER}/output/trained-models/'
model_regCNN.save(save_model_dir + f'model_{output_name}.h5')
print('saved model to: ', save_model_dir + f'model_{output_name}.h5')
# Items in the model file: <KeysViewHDF5 ['model_weights', 'optimizer_weights']>

save_metric_dir = f'/home/{io.USER}/output/metrics/{output_name}'

# Evaluate the test set
print('METRICS:')
evaluation = utils.model.evaluate_model(model_regCNN,
                                        data_train,
                                        data_test,
                                        save_metric_dir)
print(metrics.head())
metrics.to_csv(save_metric_dir + f'/metrics_{output_name}.csv', index_label='epoch')
print('Saved metrics to: ', save_metric_dir + f'/metrics_{output_name}.csv')


plot_dir = '/home/k948d562/plots/ml-vertexing-plots/training/'
utils.plot.plot_training_metrics(history, plot_dir, 'train_metrics_' + output_name)
print('Done.')
