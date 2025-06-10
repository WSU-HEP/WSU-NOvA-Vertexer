#!/usr/bin/env python
#coding: utf-8


# To run this training script:
#  $ $PY37 plot_bad_vertices_distribution.py  --data_train_path  --outdir

# ML Vtx utils
import utils.iomanager as io
import utils.plot
import utils.data_processing as dp

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd
import h5py


########### begin main script ###########

# collect the arguments for this macro. the horn and swap options are required.
parser = argparse.ArgumentParser()
parser.add_argument("--data_train_path", help="path to train data", type=str, required=True)
parser.add_argument("--outdir", help="full path to output directory", default="", type=str)
args = parser.parse_args()
args.outdir = args.outdir

# want the final dir, and extract its strings.
train_path = args.data_train_path
train_path_dir = os.path.basename(os.path.normpath(train_path))
det, horn, flux = io.IOManager.get_det_horn_and_flux_from_string(train_path_dir)

print('WARNING: You are to use the full training dataset, be sure you have the correct path to train data! ')
print('data_train_path: ', train_path)
datasets, total_events, total_files = io.load_data(train_path, False)

print('========================================')
# convert the lists to numpy arrays
datasets = dp.convert_lists_to_nparray(datasets)
print('to access FILE, index in array is: (cvnmap.shape[0] = ', datasets['cvnmap'].shape[0], 'files.)')  # first dimension is the file

#trust that they are all numpy.arrays AND have the right shape
#for key in datasets:
#    print(key)
#    dp.Debug(datasets[key]).printout_type()

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

#output directory
outdir = "/homes/m962g264/wsu_Nova_Vertexer/output/XYZ_outputs/plots/"


#Converting Detector Coordinate to PIxelmap Coordinates
datasets['firstcellx']= dp.ConvertFarDetCoords(det, 'x').convert_fd_vtx_to_pixelmap(datasets['vtx.x'], datasets['firstcellx'])
datasets['firstcelly'] = dp.ConvertFarDetCoords(det, 'y').convert_fd_vtx_to_pixelmap(datasets['vtx.y'], datasets['firstcelly'])
datasets['firstplane']= dp.ConvertFarDetCoords(det, 'z').convert_fd_vtx_to_pixelmap(datasets['vtx.z'], datasets['firstplane'])
print('Convertion Done')

#Modifying a dictionary by renaming keys
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
print('vtx_coords shape:', vtx_coords.shape)


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


# Drop the events that are outside the cvnmap!
# Apply to both features & labels.
# dictionary of {keep; np.array, drop: array}
keep_drop_evts = dp.DataCleaning.sort_events_with_vtxs_outside_cvnmaps(vtx_coords)

#Making a distribution for the droped vtx by extracting the dropped 
dropped_vtx = vtx_coords[keep_drop_evts['drop']]


#Trying to use sns.countplot, the need for the categorization
dropped_labels = (["X"] * dropped_vtx.shape[:, 0] +
                  ["Y"] * dropped_vtx.shape[:, 1] +
                  ["Z"] * dropped_vtx.shape[:, 2])

df_dropped = pd.DataFrame({"Coordinate": dropped_labels})

print('************************************************************')
print(len(keep_drop_evts['drop']), vtx_coords.shape)

total_events = vtx_coords.shape[0] + len(keep_drop_evts['drop'])
drop_percent = (len(keep_drop_evts['drop'])/total_events) * 100
drop_text = f"Dropped Events: {len(keep_drop_evts['drop'])} ({drop_percent:.2f}%)"


#For plots
fig, ax = plt.subplots(figsize=(10, 8))
sns.countplot(data=df_dropped, x="Coordinate", palette=["blue", "green", "yellow"], ax=ax)
ax.set_title("Events Outside The Cvnmap (80 by 100) Distribution by Coordinate")
ax.set_xlabel("Coordinate")
ax.set_ylabel("Events")
ax.text(0.95, 0.95, drop_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))


#Saving plots
fig.savefig(os.path.join(outdir, "Dropped_Count_plots.png"))
fig.savefig(os.path.join(outdir, "Dropped_Count_plots.pdf"))
plt.close(fig)

