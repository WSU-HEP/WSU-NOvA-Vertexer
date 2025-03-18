#!/usr/bin/env python
# coding: utf-8
# # `model_predict_coordinates_xyz.py `

# $PY37 model_predict_coordinates_xyz.py --model_file <model_file>  --outdir <$OUTPUT/predictions>

# Output: creates a CSV file of the vertex predictions from the model h5 file via training.
# Note: csv file also contains 'True' and 'Elastic Arms' vertex values.

# TODO: We remove the file index, which is just 1, so squeeze() works.
#  we'll need to fix this if/when we have multiple files to load in...

import utils.data_processing as dp
import utils.iomanager as io

import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os
import time

import utils.model

########### begin main script ###########


# collect the arguments for this macro. the horn and swap options are required.
parser = argparse.ArgumentParser()
parser.add_argument("--model_file", help="the model file to generate predictions", default="", type=str)
parser.add_argument("--outdir", help="the directory to save CSV file predictions", default="", type=str)
args = parser.parse_args()

# convert to the useful case
args.model_file = args.model_file
outdir = os.getcwd() if not args.outdir else args.outdir

training_filename_prefix = args.model_file.split('/')[-1].split('.')[0]  # remove the path, just get the filename.
DETECTOR, HORN, FLUX = io.IOManager.get_det_horn_and_flux_from_string(training_filename_prefix)
DETECTOR = DETECTOR.upper()
HORN = HORN.upper()
FLUX = FLUX.capitalize()
coordinate = 'XYZ'

print('Generating a prediction for this coordinate...')

# Load the SavedModel
print("Loading the model file to generate predictions: ", args.model_file)
model = load_model(args.model_file, compile=False)
print("Model loaded successfully.")

# Load the designated test file. This is file 27 -- has not been pre-processed; all info is in it.
# NOTE: there is only one test file for the FD validation, and it should be within the "test" directory
path_validation = '/home/k948d562/NOvA-shared/FD-Training-Samples/{}-Nominal-{}-{}/test/'.format(
    DETECTOR, HORN, FLUX)


# Read in the info from the validation file -- don't really care about the events & files here.
datasets, _, _ = io.load_data(path_validation, True)

# convert the lists to numpy arrays
print('========================================')
print('Converting lists to numpy arrays...')
# NOTE: need to redefine the 'datasets' np.array() returns a copy, so need to save to a new obj.
datasets = {key: np.array(datasets[key]) for key in datasets}

# These arrays have shape (file (row), events (column), pixels)
print("datasets['cvnmap'].shape: ", datasets['cvnmap'].shape)

# Next, for model prediction, we want to remove the file index completely. Do it now.
print('Flattening the arrays in `datasets`...')
for key in datasets:
    if key == 'cvnmap':
        print(f'Skipping {key}, will NOT flatten().')
        continue
    datasets[key] = datasets[key].flatten()
    print('datasets[{}].shape: '.format(key), datasets[key].shape)


print('========================================')
print('Addressing the bug in the h5 files. Converting firstcellx, firstcelly, firstplane to int type...')
datasets['firstcellx'] = dp.DataCleaning(datasets['firstcellx'], 'x').remove_unsigned_ints(single_file=True)
datasets['firstcelly'] = dp.DataCleaning(datasets['firstcelly'], 'y').remove_unsigned_ints(single_file=True)
datasets['firstplane'] = dp.DataCleaning(datasets['firstplane'], 'z').remove_unsigned_ints(single_file=True)
# Let's now check to make sure we don't have any gigantic numbers, from unsigned int's.
# we already know this is the common number we see if wew do this wrong...so check against it.
for i in [datasets['firstcellx'], datasets['firstcelly']]:
    event = 0
    # assuming only 1 file used:
    if i[event] > 4294967200:  # this a large number, just below the max value of an unsigned int, which should trigger
        print('event: ', event)
    event += 1

vtx_coords = np.stack(
    (
        dp.ConvertFarDetCoords(DETECTOR, 'x').convert_fd_vtx_to_pixelmap(datasets['vtx.x'], datasets['firstcellx']),
        dp.ConvertFarDetCoords(DETECTOR, 'y').convert_fd_vtx_to_pixelmap(datasets['vtx.y'], datasets['firstcelly']),
        dp.ConvertFarDetCoords(DETECTOR, 'z').convert_fd_vtx_to_pixelmap(datasets['vtx.z'], datasets['firstplane']),
    ),
    axis=-1
)
print('Done converting.')

print('Now remove the file index from the cvnmap...')
assert datasets['cvnmap'].shape[0] == 1, "More than 1 file, and the events in each file are likely not the same."
# We assume the index of 1 is the file index! With shape (1, 497067, 16000)
# Will have to come back when we have more than one file to predict...
datasets['cvnmap'] = datasets['cvnmap'].squeeze() # remove the 1, (i.e. file) index

# dictionary of {keep; np.array, drop: array}
keep_drop_evts = dp.DataCleaning.sort_events_with_vtxs_outside_cvnmaps(vtx_coords)
# Now drop events that are outside the map for ALL datasets key-value pairs
print('dropping events outside the 80x100 maps for all datasets keys')
for key in datasets:
    print(key)
    datasets[key] = datasets[key][keep_drop_evts['keep']]

print('========================================')
print('-------------------')
print('cvnmap.shape: ', datasets['cvnmap'].shape)
print('vtx_x.shape: ', datasets['vtx.x'].shape)
print('firstcellx.shape: ', datasets['firstcellx'].shape)
print('-------------------')

# Print the new shape
print("datasets['cvnmap'].shape", datasets['cvnmap'].shape)  # Should be (497067, 16000)
print('The index is now:')
print('cvnmap.shape: ', datasets['cvnmap'].shape)
print('vtx_x.shape: ', datasets['vtx.x'].shape)
print('firstcellx.shape: ', datasets['firstcellx'].shape)
print('-------------------')
# reshape the pixels into 2 (100,80) views: XZ and YZ.
print('reshape the pixels into 2 (100,80) views: XZ and YZ.....')
datasets['cvnmap'] = datasets['cvnmap'].reshape(datasets['cvnmap'].shape[0],
                                                100,
                                                80,
                                                2)  # 2 views, 100x80 pixels onto single axis
print(datasets['cvnmap'].shape)
datasets['cvnmap'] = datasets['cvnmap'].reshape(datasets['cvnmap'].shape[0],
                                                datasets['cvnmap'].shape[1],
                                                datasets['cvnmap'].shape[2],
                                                datasets['cvnmap'].shape[3],
                                                1)  # add the color channel
print(datasets['cvnmap'].shape)
# -1 automatically calculates the size of the dimension
cvnmap_xz = datasets['cvnmap'][:, :, :, 0].reshape(datasets['cvnmap'].shape[0], 100, 80, 1)  # Extract the XZ view and reshape
cvnmap_yz = datasets['cvnmap'][:, :, :, 1].reshape(datasets['cvnmap'].shape[0], 100, 80, 1)  # Extract the YZ view and reshape
print('cvnmap_xz.shape ', cvnmap_xz.shape)
print('cvnmap_yz.shape ', cvnmap_yz.shape)
print('========================================')
assert cvnmap_xz.shape == cvnmap_yz.shape, "Shapes of cvnmap_xz and cvnmap_yz must match."
print('========================================')


print('about to predict() with the model:')
# Prediction of the coordinate in the pixelmap coordinates
start = time.time()
pred_vtx = model.predict([cvnmap_xz, cvnmap_yz])  # need to give the two views bc I have dual-input model
print('Prediction done.')
end = time.time()
print('Time to predict: ', end - start)
print('-------------------')
print('pred_pixelmap.shape ', pred_vtx.shape)
print('-------------------')
print('========================================')

# Convert this prediction BACK into detector coordinates (each column is a coordinate)
print('Converting the prediction back into detector coordinates...')
x = dp.ConvertFarDetCoords(DETECTOR, 'x').convert_pixelmap_to_fd_vtx(pred_vtx[:,0], datasets['firstcellx'])
y = dp.ConvertFarDetCoords(DETECTOR, 'y').convert_pixelmap_to_fd_vtx(pred_vtx[:, 1], datasets['firstcelly'])
z = dp.ConvertFarDetCoords(DETECTOR, 'z').convert_pixelmap_to_fd_vtx(pred_vtx[:, 2], datasets['firstplane'])

# Saving: True, EA Reco, and Model Prediction into CSV file
# NOTE: bc this has shape (1, events) we need the second index.
x_series = pd.Series(datasets['vtx.x'].ravel(), name='True X')     # truth
y_series = pd.Series(datasets['vtx.y'].ravel(), name='True Y')
z_series = pd.Series(datasets['vtx.z'].ravel(), name='True Z')
vtx_x_EA = pd.Series(datasets['vtxEA.x'].ravel(), name='Reco Y')  # Reco EA
vtx_y_EA = pd.Series(datasets['vtxEA.y'].ravel(), name='Reco Y')
vtx_z_EA = pd.Series(datasets['vtxEA.z'].ravel(), name='Reco Z')
x_pred_series = pd.Series(x, name='Model X')                      # model
y_pred_series = pd.Series(y, name='Model Y')
z_pred_series = pd.Series(z, name='Model Z')

df = pd.concat([x_series, y_series, z_series, vtx_x_EA, vtx_y_EA, vtx_z_EA, x_pred_series, y_pred_series, z_pred_series], axis=1)
df.columns = ['True X', 'True Y', 'True Z', 'Reco X', 'Reco Y', 'Reco Z', 'Model Prediction X', 'Model Prediction Y', 'Model Prediction Z']
df.index.name = 'Event'
print(df.columns)

print('df', type(df))
print('The first 10 rows of the CSV are:\n', df.iloc[:10,:])
training_filename_prefix = training_filename_prefix.split('model_')[1]  # take off the 'model_' prefix
fileName = 'model_prediction_{}.csv'.format(training_filename_prefix)
df.to_csv(outdir + '/' + fileName, sep=',')
print(f'Saved to: {outdir}/{fileName}')
