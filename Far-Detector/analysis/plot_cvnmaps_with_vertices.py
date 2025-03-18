#!/usr/bin/env python
# coding: utf-8

# # `plot_cvnmaps_with_vertices.py --pred_file_x <path/to/CSV/file> --pred_file_y <path/to/CSV/file> --pred_file_z <path/to/CSV/file> --verbose false`

# script to plot the 2D pixel maps, the true vertex, Reco E.A. vertex, and Model vertex.
# A qualitative comparison of the performance of each, relative to the true location.
# Produces two sets of plots: 
#     --one of the XZ and YZ pixel maps (with the true vertex location), 
#
# Michael Dolce
# Mar. 2024

# TODO: it would be nice to remove {vtx_x,vtx_y,vtx_z} from this script -- we load this already from the CSVs


import argparse
import os.path
import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import h5py


# Get the detector...
def get_detector(pred_filename_prefix):
    detector = ''
    print('Determining the detector...')
    if 'FD' in pred_filename_prefix:
        detector = 'FD'
    elif 'ND' in pred_filename_prefix:
        detector = 'ND'
    else:
        print('ERROR. I did not find a detector to make predictions for, exiting......')
        exit()
    print('DETECTOR: {}'.format(detector))
    return detector

# Get coordinate...
def get_coordintate(pred_filename_prefix):
    coordinate = ''
    print('Determining the coordinate...')
    if '_X_' in pred_filename_prefix:
        coordinate = 'x'
    elif '_Y_' in pred_filename_prefix:
        coordinate = 'y'
    elif '_Z_' in pred_filename_prefix:
        coordinate = 'z'
    else:
        print('ERROR. I did not find a coordinate to make predictions for, exiting......')
        exit()
    print('COORDINATE: {}'.format(coordinate))
    return coordinate

# Get horn...
def get_horn(pred_filename_prefix):
    horn = ''
    print('Determining the horn...')
    if 'FHC' in pred_filename_prefix:
        horn = 'FHC'
    elif 'RHC' in pred_filename_prefix:
        horn = 'RHC'
    else:
        print('ERROR. I did not find a horn to make predictions for, exiting......')
        exit()
    print('HORN: {}'.format(horn))
    return horn

# Get flux...
def get_flux(pred_filename_prefix):
    flux = ''
    print('Determining the flux...')
    if 'Fluxswap' in pred_filename_prefix:
        flux = 'Fluxswap'
    elif 'Nonswap' in pred_filename_prefix:
        flux = 'Nonswap'
    else:
        print('ERROR. I did not find a flux to make predictions for, exiting......')
        exit()
    print('FLUX: {}'.format(flux))
    return flux


def convert_vtx_x_to_pixelmap(vtx_x_array, firstcellx_array, detStr):
    """
    :param vtx_x_array: `vtx.x` -- x in detector coordinates.
    :param firstcellx_array: `firstcellx` -- first x cell in pixelmap coordinates
    :param detStr: which detector (ND or FD)
    :return: x pixelmap coordinate
    """
    print('Converting x coordinate for {}...'.format(detStr))
    assert (type(vtx_x_array) == np.ndarray), "x_array must be a numpy array"
    if detStr == 'ND':
        return vtx_x_array / 3.99 - firstcellx_array + 48
    elif detStr == 'FD':
        return vtx_x_array / 3.97 - firstcellx_array + 192
    else:
        print('NO DETECTOR. No X coordinate conversion.')
        return None


def convert_vtx_y_to_pixelmap(vtx_y_array, firstcelly_array, detStr):
    """
    :param vtx_y_array: `vtx.y` -- y in detector coordinates.
    :param firstcelly_array: `firstcelly` -- first y cell in pixelmap coordinates
    :param detStr: which detector (ND or FD)
    :return: y pixelmap coordinate
    """
    print('Converting y coordinate for {}...'.format(detStr))
    assert (type(vtx_y_array) == np.ndarray), "y_array must be a numpy array"
    if detStr == 'ND':
        return vtx_y_array / 3.97 - firstcelly_array + 47
    elif detStr == 'FD':
        return vtx_y_array / 3.97 - firstcelly_array + 191
    else:
        print('NO DETECTOR. No Y coordinate conversion.')
        return None


def convert_vtx_z_to_pixelmap(vtx_z_array, firstplane_array, det):
    """
    :param vtx_z_array: `vtx.z` -- z in detector coordinates.
    :param firstplane_array: `firstplane` -- first plane in pixelmap coordinates
    :param det: which detector (ND or FD)
    :return: z pixelmap coordinate
    """
    print('Converting z coordinate for {}...'.format(det))
    assert (type(vtx_z_array) == np.ndarray), "z_array must be a numpy array"
    if det == 'ND':
        return vtx_z_array / 6.61 - firstplane_array
    elif det == 'FD':
        return vtx_z_array / 6.664 - firstplane_array
    else:
        print('NO DETECTOR. No Z coordinate conversion.')
        return None




# # collect the arguments for this macro. the horn and swap options are required.
parser = argparse.ArgumentParser()
parser.add_argument("--pred_file_x", help="the CSV prediction file", default="", type=str)
parser.add_argument("--pred_file_y", help="the CSV prediction file", default="", type=str)
parser.add_argument("--pred_file_z", help="the CSV prediction file", default="", type=str)
parser.add_argument("--verbose", help="add printouts, helpful for debugging", action=argparse.BooleanOptionalAction, type=bool)

args = parser.parse_args()
args.pred_file_x = args.pred_file_x
args.pred_file_y = args.pred_file_y
args.pred_file_z = args.pred_file_z
args.verbose = args.verbose
print('pred_file_x: {}'.format(args.pred_file_x))
print('pred_file_y: {}'.format(args.pred_file_y))
print('pred_file_z: {}'.format(args.pred_file_z))

# make sure the information is correct before we retrieve the information from the files...
pred_filename_prefix_x = args.pred_file_x.split('/')[-1].split('.')[0]  # get the filename only and remove the .csv
pred_filename_prefix_y = args.pred_file_y.split('/')[-1].split('.')[0]  # get the filename only and remove the .csv
pred_filename_prefix_z = args.pred_file_z.split('/')[-1].split('.')[0]  # get the filename only and remove the .csv

COORDINATE_x = get_coordintate(pred_filename_prefix_x)
COORDINATE_y = get_coordintate(pred_filename_prefix_y)
COORDINATE_z = get_coordintate(pred_filename_prefix_z)
DETECTOR_x = get_detector(pred_filename_prefix_x)
DETECTOR_y = get_detector(pred_filename_prefix_y)
DETECTOR_z = get_detector(pred_filename_prefix_z)
HORN_x = get_horn(pred_filename_prefix_x)
HORN_y = get_horn(pred_filename_prefix_y)
HORN_z = get_horn(pred_filename_prefix_z)
FLUX_x = get_flux(pred_filename_prefix_x)
FLUX_y = get_flux(pred_filename_prefix_y)
FLUX_z = get_flux(pred_filename_prefix_z)

assert (COORDINATE_x == 'x' and COORDINATE_y == 'y' and COORDINATE_z == 'z'), "COORDINATE_x, COORDINATE_y, COORDINATE_z must be x, y, z"
assert (DETECTOR_x == DETECTOR_y == DETECTOR_z), "DETECTOR_x, DETECTOR_y, DETECTOR_z must be the same"
assert (HORN_x == HORN_y == HORN_z), "HORN_x, HORN_y, HORN_z must be the same"
assert (FLUX_x == FLUX_y == FLUX_z), "FLUX_x, FLUX_y, FLUX_z must be the same"


# read in the file:
print('='*50)
df_model_x = pd.read_csv(args.pred_file_x)
print('df_model_x.head(): ', df_model_x.head())
print('='*50)
df_model_y = pd.read_csv(args.pred_file_y)
print('df_model_y.head(): ', df_model_y.head())
print('='*50)
df_model_z = pd.read_csv(args.pred_file_z)
print('df_model_z.head(): ', df_model_z)
print('='*50)

model_pred_x = np.array(df_model_x['Model Prediction'])
model_pred_y = np.array(df_model_y['Model Prediction'])
model_pred_z = np.array(df_model_z['Model Prediction'])

reco_EA_x = np.array(df_model_x['Reco X'])
reco_EA_y = np.array(df_model_y['Reco Y'])
reco_EA_z = np.array(df_model_z['Reco Z'])

# create a string to use for the plot name -- all the same, so just use one of them
str_det_horn_flux = '{}_{}_{}'.format(DETECTOR_x, HORN_x, FLUX_x)


# plot directory
plot_dir_prefix = '/Users/michaeldolce/Desktop/ml-vertexing-plots/analysis/plot_cvnmaps_with_vertices/'
plot_dir_model_name = '{}-{}-{}'.format(pred_filename_prefix_x, pred_filename_prefix_y, pred_filename_prefix_z)
plot_dir_local = plot_dir_prefix + plot_dir_model_name

if not os.path.exists(plot_dir_local):
    os.makedirs(plot_dir_local)
    print('created dir: {}'.format(plot_dir_local))
else:
    print('dir already exists: {}'.format(plot_dir_local))


# ## Load the file(s) --> `cvnmap`, `vtx_x`, `vtx_y`, `vtx_z`,  `firstcellx`, `firstcelly`, `firstplane`
# NOTE: `firstcellx`, `firstcelly`, `firstplane` arrays are needed to convert detector <--> pixel coordinates
# This is an essential step to plot the vertex on the pixel map and for the training.

# Conversion from cm (vtx.{x,y,z}) to pixels (cvnmaps) for "standard"-type datasets
# fx == firstcellx, fy == firstcelly, fz == firstplane

# NOTE: we should be using the file 27 here, again.
# NOTE: this file has {x,y,z} in it! --> just need one file for all three coordinates.
local_h5_path = '/Users/michaeldolce/Development/files/h5-files/validation/'
filename = 'trimmed_h5_R20-11-25-prod5.1reco.j_{}-Nominal-{}-{}_27_of_28.h5'.format(DETECTOR_x, HORN_x, FLUX_x)
local_h5_file = local_h5_path + filename
print('local_h5_file about to read in: ', local_h5_file)

# read in the h5 file
# train & test size -- i.e. the number of events in each file
total_events = 0
event_cutoff = False  # restrict number of events to load into memory. Unused atm.

# read in the h5 file
with h5py.File(local_h5_file, 'r') as f:
    events_per_file = len(f['vtx.x'][:])  # count events in each file # can try: sum(len(x) for x in multilist)
    total_events += events_per_file

    # create numpy.arrays of the information
    cvnmap = f['cvnmap'][:]
    vtx_x = f['vtx.x'][:]
    vtx_y = f['vtx.y'][:]
    vtx_z = f['vtx.z'][:]
    firstcellx = f['firstcellx'][:]
    firstcelly = f['firstcelly'][:]
    firstplane = f['firstplane'][:]
    print('events_per_file: ', events_per_file)

print('total_events: ', total_events)

print('fx type: ', type(firstcellx), '. and its elements, ', type(firstcellx[0]))


# ensure all objects are numpy arrays
for i in [cvnmap, vtx_x, vtx_y, vtx_z, firstcellx, firstcelly, firstplane]:
    idx = 0
    assert (type(i) == np.ndarray), "i must be a numpy array"
    if idx == 0:
        print('shape of array', i.shape)
    idx += 1
print('cvnmap type: ', type(cvnmap))
print('Passed. All are numpy arrays.')
print('For reference, the shape of np.arrays......')
print('(events_per_file) ...... aside from cvnmap, which is (events_per_file, 16000) ')

# Special conversion needed for the cells and plane (from Erin E., unclear how this fixes the issue though)
# NOTE: these objects ARE arrays, so this is fairly simple math here
assert (len(firstcellx) == len(firstcelly) == len(firstplane)), "firstcellx, firstcelly, firstplane must be same length"

# convert the cell and plane arrays to integers
# NOTE: for Prod5.1 h5 samples (made from Reco Conveners), the firstcellx, firstcelly arrays are `unsigned int`s.
#       this is incorrect. They need to be `int` type. So Erin E. discovered the solution that we use here: 
#       -- first add 40 to each element in the array
#       -- then convert the array to `int` type
#       -- then subtract 40 from each element in the array
# We do this to `firstplane` as well (cast as int) although not strictly necessary.
# If you do not do this, firstcell numbers can be 4294967200, which is the max value of an unsigned int -- and wrong.

# some debugging, to be sure...
# print('firstcellx[1] before conversion: ', firstcellx[1], type(firstcellx[1]))
print('firstcellx[100] at start: ', firstcellx[100], type(firstcellx[100]))

firstcellx += 40
firstcellx = np.array(firstcellx, dtype='int')
print('firstcellx[100] after conversion + 40 addition: ', firstcellx[100], type(firstcellx[100]))
firstcellx -= 40
print('firstcellx[100] after conversion + 40 subtraction: ', firstcellx[100], type(firstcellx[100]))

firstcelly += 40
firstcelly = np.array(firstcelly, dtype='int')
firstcelly -= 40

firstplane = np.array(firstplane, dtype='int')  # not strictly necessary, Erin doesn't do it...


# Let's now check to make sure we don't have any gigantic numbers.
# we already know this is the common number we see if wew do this wrong...so check against it.
for i in [firstcellx, firstcelly, firstplane]:
    event = 0
    if i[event] > 4294967200:  # this a large number, just below the max value of an unsigned int, which should trigger
        print('i: ', i)
    event += 1


# then create the new array for the pixelmap coordinates of the vertex

# convert the vertex location (detector coordinates) to pixel map coordinates
vtx_x_pixelmap = convert_vtx_x_to_pixelmap(vtx_x, firstcellx, DETECTOR_x)
vtx_y_pixelmap = convert_vtx_y_to_pixelmap(vtx_y, firstcelly, DETECTOR_y)
vtx_z_pixelmap = convert_vtx_z_to_pixelmap(vtx_z, firstplane, DETECTOR_z)

vtx_model_x_pixelmap = convert_vtx_x_to_pixelmap(model_pred_x, firstcellx, DETECTOR_x)
vtx_model_y_pixelmap = convert_vtx_y_to_pixelmap(model_pred_y, firstcelly, DETECTOR_y)
vtx_model_z_pixelmap = convert_vtx_z_to_pixelmap(model_pred_z, firstplane, DETECTOR_z)

vtx_EA_x_pixelmap = convert_vtx_x_to_pixelmap(reco_EA_x, firstcellx, DETECTOR_x)
vtx_EA_y_pixelmap = convert_vtx_y_to_pixelmap(reco_EA_y, firstcelly, DETECTOR_y)
vtx_EA_z_pixelmap = convert_vtx_z_to_pixelmap(reco_EA_z, firstplane, DETECTOR_z)

print('Done converting.')

print('type of vtx_x_pixelmap: ', type(vtx_x_pixelmap))
print('and shape: ', vtx_x_pixelmap.shape)

# print out info for single event to double check
if args.verbose:
    print('Here is an example of the conversion for a single event:')
    print('vtx_x[1] = ', vtx_x[1])
    print('vtx_x_pixelmap[1] = ', vtx_x_pixelmap[1])
    print('firstcellx after conversion: ', firstcellx[1])
    print('vtx_z[1] = ', vtx_z[1])
    print('vtx_z_pixelmap[1] = ', vtx_z_pixelmap[1])





# create plot of the cvnmap
def plot_cvnmap_with_vertices(event_idx=1, plot_vtx=True, plot_Model_vtx=True, plot_EA_vtx=True):
    """
    Create a plot of the cvnmap in the XZ and YZ views.
    :NOTE: add [:-20] sns.heatmap() to cut the end of pixelmap in Z coordinate to get square images of the maps
    :param event_idx: select which event you want to plot.
    :param plot_vtx: plot true vertex
    :param plot_Model_vtx: plot Model vertex
    :param plot_EA_vtx: plot Reco E.A. vertex
    :return: figure
    """
    # print('cvn_map_per_file[event].shape: ', cvnmap_per_file[event_idx].shape)

    print('Plotting:\t Event {}'.format(event_idx))
    true_str, model_str, ea_str = '', '', ''

    # make a new array of just the single event to plot. and make it the correct size.
    assert cvnmap[event_idx].shape == (16000,), "cvnmap[event_idx] must be shape (16000,)"
    cvnmap_to_plot = cvnmap[event_idx].reshape(2, 100, 80)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(21, 7))
    # NOTE: add [:-20] to end of pixelmap for square images
    sns.heatmap(cvnmap_to_plot[0][:], cmap='coolwarm', cbar=False, square=True, xticklabels=10, yticklabels=10,
                ax=axes[0])  # first index of the array is XZ view
    sns.heatmap(cvnmap_to_plot[1][:], cmap='coolwarm', cbar=False, square=True, xticklabels=10, yticklabels=10,
                ax=axes[1])  # second index of the array is YZ view

    if plot_vtx:
        # plot True vertex point
        axes[0].scatter(x=vtx_x_pixelmap[event_idx], y=vtx_z_pixelmap[event_idx], c='magenta', marker='x', s=250)
        axes[1].scatter(x=vtx_y_pixelmap[event_idx], y=vtx_z_pixelmap[event_idx], c='magenta', marker='x', s=250)
        true_str = 'True'

    if plot_Model_vtx:
        # plot Model vertex point
        axes[0].scatter(x=vtx_model_x_pixelmap[event_idx], y=vtx_model_z_pixelmap[event_idx], c='black', marker='x', s=250)
        axes[1].scatter(x=vtx_model_y_pixelmap[event_idx], y=vtx_model_z_pixelmap[event_idx], c='black', marker='x', s=250)
        model_str = 'Model'

    if plot_EA_vtx:
        # plot Reco E.A. vertex point
        axes[0].scatter(x=vtx_EA_x_pixelmap[event_idx], y=vtx_EA_z_pixelmap[event_idx], c='lime', marker='x', s=250)
        axes[1].scatter(x=vtx_EA_y_pixelmap[event_idx], y=vtx_EA_z_pixelmap[event_idx], c='lime', marker='x', s=250)
        ea_str = 'EA'


    # print the vertex location -- down to two decimal places
    x_true = ('%.2f' % vtx_x[event_idx])
    y_true = ('%.2f' % vtx_y[event_idx])
    z_true = ('%.2f' % vtx_z[event_idx])
    x_model = ('%.2f' % model_pred_x[event_idx])
    y_model = ('%.2f' % model_pred_y[event_idx])
    z_model = ('%.2f' % model_pred_z[event_idx])
    x_ea = ('%.2f' % reco_EA_x[event_idx])
    y_ea = ('%.2f' % reco_EA_y[event_idx])
    z_ea = ('%.2f' % reco_EA_z[event_idx])

    # print('CVN Vertex Position [pixels] (x,y,z) = ({},{},{})'.format(x, y, z))
    # plt.text(-70, 50, 'CVN Vertex Position [pixels]:\n(x,y,z) = ({},{},{})\n Event: {}'.format(x, y, z, event_idx),
    #          fontsize=15)
    axes[0].set_title('XZ View', fontsize=25)
    axes[1].set_title('YZ View', fontsize=25)
    plt.suptitle("CVN pixel maps", fontsize=30)
    plt.text(20, -10, 'NOvA Simulation', color='grey', fontsize=26)
    # plt.text(-100, -10, 'NOvA Simulation', color='grey', fontsize=26)
    plt.text(-55, 0, '{} {} {}'.format(DETECTOR_x, HORN_x, FLUX_x), fontsize=15, weight='bold')
    plt.text(-80, 50, 'True: ({}, {}, {}) cm'.format(x_true, y_true, z_true), fontsize=12)
    plt.text(-80, 40, 'Model: ({}, {}, {}) cm'.format(x_model, y_model, z_model ), fontsize=12)
    plt.text(-80, 30, 'E.A.: ({}, {}, {}) cm'.format(x_ea, y_ea, z_ea), fontsize=12)

    axes[0].set_xlabel("Cell", fontsize=25)
    axes[0].set_ylabel("Plane", fontsize=25)
    axes[1].set_xlabel("Cell", fontsize=25)
    axes[1].set_ylabel("Plane", fontsize=25)

    plt.legend(['True Vertex', 'Model Vertex', 'Reco E.A. Vertex'], fontsize=15, loc='lower right')

    # save the plot
    plot_str = ''
    if plot_vtx:
        plot_str += true_str + '_'
    if plot_Model_vtx:
        plot_str += model_str + '_'
    if plot_EA_vtx:
        plot_str += ea_str + '_'

    for ext in ['pdf', 'png']:
        plt.savefig(plot_dir_local + '/cvnmaps_{}_{}_{}_eventidx{}.{}'.format(DETECTOR_x, HORN_x, event_idx, plot_str, ext))

    return 'CVN map produced. Event: {}'.format(event_idx)



# for random_event in [1, 10, 100, 1000, 10000-1]:

# create some random events to plot
from random import randint
rndm_array = [randint(0, total_events) for i in range(3)]
for random_event in rndm_array:
    plot_cvnmap_with_vertices(event_idx=random_event, plot_vtx=True, plot_Model_vtx=True, plot_EA_vtx=True)

plot_cvnmap_with_vertices(event_idx=3076, plot_vtx=True, plot_Model_vtx=True, plot_EA_vtx=True)

