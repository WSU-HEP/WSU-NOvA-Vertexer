#!/usr/bin/env python
# coding: utf-8
# # plot_2d_vertex_resolutions.py
# M. Dolce. 
# Feb. 2024
# 
# Makes plots to compare the true and E.A. reconstructed vertex (from a specified coordinate).
# NOTE: this makes resolution plots of over TWO COORDINATES.
# NOTE: order matters for drawing the 2D hist and contour plot.
# --x_axis_file --> CSV model file to plot on the x-axis.
# --y_axis_file --> CSV model file to plot on the y-axis.
# 
# This validation is using `file_27_of_28.h5` as Ashley indicated.
# 
#  $PY37 plot_2d_vertex_resolutions.py --x_axis_file <path/to/CSV/file/x-axis> --y_axis_file <path/to/CSV/file/y-axis>
# 

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


import argparse
import os
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# import seaborn as sns


# collect the arguments for this macro.
parser = argparse.ArgumentParser()
parser.add_argument("--x_axis_file", help="CSV file for x-axis", default="", type=str)
parser.add_argument("--y_axis_file", help="CSV file for y-axis", default="", type=str)
parser.add_argument("--verbose", help="add printouts, helpful for debugging", action=argparse.BooleanOptionalAction, type=bool)
parser.add_argument("--draw_contours", help="draw contours on the 2D hist", action=argparse.BooleanOptionalAction, type=bool)
args = parser.parse_args()

# print the arguments
args.x_axis_file = args.x_axis_file
args.y_axis_file = args.y_axis_file
args.verbose = args.verbose
args.draw_contours = args.draw_contours
print('=' * 100)
print('x-axis file: {}'.format(args.x_axis_file))
print('y-axis file: {}'.format(args.y_axis_file))
print('verbose: {}'.format(args.verbose))
print('draw contours: {}'.format(args.draw_contours))
print('=' * 100)

# filename without the CSV and path.
pred_filename_prefix_xaxis = args.x_axis_file.split('/')[-1].split('.')[0]
pred_filename_prefix_yaxis = args.y_axis_file.split('/')[-1].split('.')[0]


# Get the detector, horn, and flux from the filename.
COORDINATE_xaxis = get_coordintate(pred_filename_prefix_xaxis)
COORDINATE_yaxis = get_coordintate(pred_filename_prefix_yaxis)

DETECTOR_xaxis = get_detector(pred_filename_prefix_xaxis)
DETECTOR_yaxis = get_detector(pred_filename_prefix_yaxis)

HORN_xaxis = get_horn(pred_filename_prefix_xaxis)
HORN_yaxis = get_horn(pred_filename_prefix_yaxis)

FLUX_xaxis = get_flux(pred_filename_prefix_xaxis)
FLUX_yaxis = get_flux(pred_filename_prefix_yaxis)

# check if the detector, horn, and flux are the same for both files.
assert DETECTOR_xaxis == DETECTOR_yaxis, print('DETECTOR_xaxis != DETECTOR_yaxis')
assert HORN_xaxis == HORN_yaxis, print('HORN_xaxis != HORN_yaxis')
assert FLUX_xaxis == FLUX_yaxis, print('FLUX_xaxis != FLUX_yaxis')
# coordinate should be different.
assert COORDINATE_xaxis != COORDINATE_yaxis, print('COORDINATE_xaxis == COORDINATE_yaxis')

# This will be the x-axis on the 2D plot
csvfile_x = pd.read_csv(args.x_axis_file,
                        sep=',',
                        dtype={'': int,
                                 'True {}'.format(COORDINATE_xaxis.upper()): float,
                                 'Reco {}'.format(COORDINATE_xaxis.upper()): float,
                                 'Model Prediction': float},
                        low_memory=False
                        )

# This will be the y-axis on the 2D plot
csvfile_y = pd.read_csv(args.y_axis_file,
                        sep=',',
                        dtype={'': int,
                               'True {}'.format(COORDINATE_yaxis.upper()): float,
                               'Reco {}'.format(COORDINATE_yaxis.upper()): float,
                               'Model Prediction': float},
                        low_memory=False
                        )


# The 'Model Pred {}' is lower case. The rest are upper case.
csvfile_x.columns = ['Row',
                     'True {}'.format(COORDINATE_xaxis.upper()),
                     'Reco {}'.format(COORDINATE_xaxis.upper()),
                     'Model Pred {}'.format(COORDINATE_xaxis)
                     ]
csvfile_y.columns = ['Row',
                     'True {}'.format(COORDINATE_yaxis.upper()),
                     'Reco {}'.format(COORDINATE_yaxis.upper()),
                     'Model Pred {}'.format(COORDINATE_yaxis)
                     ]

df_pred_xaxis = csvfile_x['Model Pred {}'.format(COORDINATE_xaxis)]
df_pred_yaxis = csvfile_y['Model Pred {}'.format(COORDINATE_yaxis)]

assert len(df_pred_xaxis) == len(df_pred_yaxis), print('len(df_pred) != len(df_pred_yaxis)')
if args.verbose:
    print('len(df_pred_xaxis) == len(df_pred_yaxis)')

# define the path to the validation files
validation_path = '/Users/michaeldolce/Development/files/h5-files/validation/'
if args.verbose:
    print(os.listdir(validation_path))


# TODO: will need to change the validation file in future to include Nonswap and FHC/RHC...
# Open the designated validation file
# Only needs the one file for a 2D distribution!
# DETECTOR, HORN, and FLUX are the same for both files.
validation_file = 'trimmed_h5_R20-11-25-prod5.1reco.j_{}-Nominal-{}-{}_27_of_28.h5'.format(DETECTOR_xaxis,
                                                                                           HORN_xaxis,
                                                                                           FLUX_xaxis)

file_h5 = h5py.File(validation_path + validation_file, 'r', )  # open the file:
file_h5.keys()  # see what is in the file
print("Number of events: ", len(file_h5['E']))  # count the number of events in the file

# make sure the number of events is the same in both files
assert len(df_pred_xaxis) == len(df_pred_yaxis) == len(file_h5['E']), print('{} != {}', len(df_pred_yaxis),
                                                                            len(df_pred_xaxis),
                                                                            len(file_h5['E']))
print('=' * 100)
print(df_pred_xaxis.head())
print('=' * 50)
print(df_pred_yaxis.head())
print('=' * 100)



# create dataframe from the h5 validation file

# Create the DataFrame directly from the h5 file
df = pd.DataFrame({'E': file_h5['E'][:],
                   'PDG': file_h5['pdg'][:],
                   'Interaction': file_h5['interaction'][:],
                   'isCC': file_h5['iscc'][:],
                   'Mode': file_h5['mode'][:],
                   'NCid': file_h5['ncid'][:],
                   'FinalState': file_h5['finalstate'][:],
                   'Px': file_h5['p.px'][:],
                   'Py': file_h5['p.py'][:],
                   'Pz': file_h5['p.pz'][:],
                   'vtx.x': file_h5['vtx.x'][:],
                   'vtx.y': file_h5['vtx.y'][:],
                   'vtx.z': file_h5['vtx.z'][:],
                   'vtxEA.x': file_h5['vtxEA.x'][:],
                   'vtxEA.y': file_h5['vtxEA.y'][:],
                   'vtxEA.z': file_h5['vtxEA.z'][:],
                   })


# Add in the Model Prediction.
df = pd.concat([df, df_pred_xaxis, df_pred_yaxis], axis=1)

# print the head of the dataframe
if args.verbose:
    print(df.keys())

# Add the following columns: absolute vertex difference between...
# -- abs(E.A. - True) vertex
# -- abs(Model - True) vertex.

print('Creating "AbsVtxDiff.EA.{}" and "AbsVtxDiff.Model.{}" column'.format(COORDINATE_xaxis, COORDINATE_xaxis))
vtx_diff_temp_xaxis = pd.DataFrame(abs(df['vtx.{}'.format(COORDINATE_xaxis)] - df['vtxEA.{}'.format(COORDINATE_xaxis)]),
                                   columns=['AbsVtxDiff.EA.{}'.format(COORDINATE_xaxis)])
vtx_diff_temp_Model_xaxis = pd.DataFrame(abs(df['vtx.{}'.format(COORDINATE_xaxis)] - df['Model Pred {}'.format(COORDINATE_xaxis)]),
                                         columns=['AbsVtxDiff.Model.{}'.format(COORDINATE_xaxis)])

print('Creating "AbsVtxDiff.EA.{}" and "AbsVtxDiff.Model{}" column'.format(COORDINATE_yaxis, COORDINATE_yaxis))
vtx_diff_temp_yaxis = pd.DataFrame(abs(df['vtx.{}'.format(COORDINATE_yaxis)] - df['vtxEA.{}'.format(COORDINATE_yaxis)]),
                             columns=['AbsVtxDiff.EA.{}'.format(COORDINATE_yaxis)])
vtx_diff_temp_Model_yaxis = pd.DataFrame(abs(df['vtx.{}'.format(COORDINATE_yaxis)] - df['Model Pred {}'.format(COORDINATE_yaxis)]),
                                   columns=['AbsVtxDiff.Model.{}'.format(COORDINATE_yaxis)])

# TODO: maybe add the relative resolution values right here too....? --> (true - pred)/true
df = pd.concat([df, vtx_diff_temp_xaxis, vtx_diff_temp_Model_xaxis, vtx_diff_temp_yaxis, vtx_diff_temp_Model_yaxis], axis=1)

# create a column of the Energy within NOvA's range/relevance -- for convenience.
E_nova = df[df['E'] < 5]
df['E NOvA'] = E_nova['E']
df.describe()

if args.verbose:
    print(df.head())  # check the head of the dataframe
    df.describe()  # describe the dataframe
    print(df.columns)  # print the columns of the dataframe
    # df.info() # get info on the dataframe



# Interaction types.

# NOTE: the information is stored within StandardRecord/SREnums.h
#   /// Neutrino interaction categories
#   enum mode_type_{
#     kUnknownMode               = -1,
#     kQE                        = 0,
#     kRes                       = 1,
#     kDIS                       = 2,
#     kCoh                       = 3,
#     kCohElastic                = 4,
#     kElectronScattering        = 5,
#     kIMDAnnihilation           = 6,
#     kInverseBetaDecay          = 7,
#     kGlashowResonance          = 8,
#     kAMNuGamma                 = 9,
#     kMEC                       = 10,
#     kDiffractive               = 11,
#     kEM                        = 12,
#     kWeakMix                   = 13
#   };

# assert are no kUnknownMode events...
assert (len(df[df['Mode'] == -1]) == 0)  # 'there are Unknown events. Stopping'

mode_names = ['QE', 'Res', 'DIS', 'Coh', 'CohElastic', 'ElectronScattering', 'IMDAnnihilation', 'InverseBetaDecay',
              'GlashowResonance', 'AMNuGamma', 'MEC', 'Diffractive', 'EM', 'WeakMix']

# define the colors for each mode (as close as possible to what they are in NOvA Style)

# colors for the Elastic Arms Reco.
mode_colors = {'QE': 'royalblue',
               'MEC': 'gold',
               'DIS': 'silver',
               'CohElastic': 'green',
               'Res': 'limegreen',
               'Coh': 'lightcoral',
               'ElectronScattering': 'purple',
               'IMDAnnihilation': 'pink',
               'InverseBetaDecay': 'chocolate',
               'GlashowResonance': 'cyan',
               'AMNuGamma': 'magenta',
               'Diffractive': 'dimgray',
               'EM': 'khaki',
               'WeakMix': 'teal'
               }

# colors for the model predictions
# these selected colors are darker shades from the Elastic Arms one.
mode_colors_Model = {'QE': 'navy',
                     'MEC': 'darkgoldenrod',
                     'DIS': 'grey',
                     'CohElastic': 'darkgreen',
                     'Res': 'forestgreen',
                     'Coh': 'firebrick',
                     'ElectronScattering': 'indigo',
                     'IMDAnnihilation': 'palevioletred',
                     'InverseBetaDecay': 'brown',
                     'GlashowResonance': 'cadetblue',
                     'AMNuGamma': 'darkmagenta',
                     'Diffractive': 'black',
                     'EM': 'olive',
                     'WeakMix': 'darkslategrey'
                     }

# list of the dataframes for each mode has both NC and CC.
df_modes = list()
for i in range(0, len(mode_names)):
    if args.verbose:
        print(mode_names[i])
    df_modes.append(df[df['Mode'] == i])

# just as a simple check, print out the head of the QE events.
if args.verbose:
    print('printing out head() for QE events...')
    df_modes[1].head()
    print(df_modes[1].columns)


# define the plot names...

dirname = (pred_filename_prefix_xaxis.split('model_prediction_')[1]
           + '__vs__'
           + pred_filename_prefix_yaxis.split('model_prediction_')[1])

# define path to save some plots (the local dir).
OUTDIR = ('/Users/michaeldolce/Desktop/ml-vertexing-plots/analysis/plot_2d_vertex_resolutions/'
          + dirname + '/')
print('plot outdir is: {}'.format(OUTDIR))

if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)
    print('created dir: {}'.format(OUTDIR))
else:
    print('dir already exists: {}'.format(OUTDIR))

# Again, can use any here, because these are the same for both files.
str_det_horn_flux = '{}_{}_{}'.format(DETECTOR_xaxis, HORN_xaxis, FLUX_xaxis)




# for reco - true vertex. 
bins_resolution = np.arange(-20, 20, 0.1)  # .1 bin per cm.


# All interactions
# ELASTIC ARMS

# plot the resolution of the vertex for each interaction type on single plot.
fig_res_int_EA = plt.figure(figsize=(10, 8))

counts_EA_res, xbins_EA_res, ybins_EA_res, imEA = plt.hist2d(df['vtxEA.{}'.format(COORDINATE_xaxis)] - df['vtx.{}'.format(COORDINATE_xaxis)],
           df['vtxEA.{}'.format(COORDINATE_yaxis)] - df['vtx.{}'.format(COORDINATE_yaxis)],
           bins=(bins_resolution, bins_resolution), cmap='viridis', cmin=1)
plt.colorbar(label='Events')
if args.draw_contours:
    plt.contour(counts_EA_res, extent=[xbins_EA_res.min(), xbins_EA_res.max(), ybins_EA_res.min(), ybins_EA_res.max()], linewidths=1, levels=3, colors='red', linestyle='--')

plt.title('Elastic Arms Vertex Resolution')
plt.xlabel('(Reco - True) Vertex {} [cm]'.format(COORDINATE_xaxis))
plt.ylabel('(Reco - True) Vertex {} [cm]'.format(COORDINATE_yaxis))
plt.text(10, 20, '{}'.format(str_det_horn_flux.replace("_", " ")), fontsize=12)
plt.text(xbins_EA_res.max() * 0.35, ybins_EA_res.max() * 1.1, 'NOvA Simulation', fontsize=26, color='grey')
# plt.legend(loc='upper right')
plt.subplots_adjust(bottom=0.15, left=0.15)
plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')
# plt.show()

for ext in ['pdf', 'png']:
    fig_res_int_EA.savefig(
        OUTDIR + '/' + '/plot_{}_allmodes_Resolution_{}{}_ElasticArms.'.format(
            str_det_horn_flux, COORDINATE_xaxis, COORDINATE_yaxis) + ext, dpi=300)


# Model Prediction
# plot the resolution of the vertex for each interaction type on single plot.
fig_res_int_Model = plt.figure(figsize=(10, 8))

counts_M_res, xbins_M_res, ybins_M_res, im_M_res = plt.hist2d(df['Model Pred {}'.format(COORDINATE_xaxis)] - df['vtx.{}'.format(COORDINATE_xaxis)],
           df['Model Pred {}'.format(COORDINATE_yaxis)] - df['vtx.{}'.format(COORDINATE_yaxis)],
           bins=(bins_resolution, bins_resolution), cmap='viridis', cmin=1)
plt.colorbar(label='Events')
if args.draw_contours:
    plt.contour(counts_M_res, extent=[xbins_M_res.min(), xbins_M_res.max(), ybins_M_res.min(), ybins_M_res.max()], linewidths=1, levels=3, colors='red', linestyle='--')

plt.title('Model Prediction Resolution')
plt.xlabel('(Reco - True) Vertex {} [cm]'.format(COORDINATE_xaxis))
plt.ylabel('(Reco - True) Vertex {} [cm]'.format(COORDINATE_yaxis))
plt.text(10, 20, '{}'.format(str_det_horn_flux.replace("_", " ")), fontsize=12)
plt.text(xbins_M_res.max() * 0.35, ybins_M_res.max() * 1.1, 'NOvA Simulation', fontsize=26, color='grey')
# plt.legend(loc='upper right')
plt.subplots_adjust(bottom=0.15, left=0.15)
plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')
# plt.show()

for ext in ['pdf', 'png']:
    fig_res_int_Model.savefig(
        OUTDIR + '/' + '/plot_{}_allmodes_Resolution_{}{}_ModelPred.'.format(
            str_det_horn_flux, COORDINATE_xaxis, COORDINATE_yaxis) + ext, dpi=300)



# =================================================================================================
# plot the abs(reco - true) vertex difference for both: Elastic Arms and Model Prediction
# Abs(resolution)

# for abs(reco - true) vertex difference for Elastic Arms
bins_abs_resolution = np.arange(0, 9, .1)  # .1 bin per cm.

fig_resolution = plt.figure(figsize=(5, 3))

counts_EA, xbins_EA, ybins_EA, image_EA = plt.hist2d(abs(df['vtxEA.{}'.format(COORDINATE_xaxis)] - df['vtx.{}'.format(COORDINATE_xaxis)]),
                                                     abs(df['vtxEA.{}'.format(COORDINATE_yaxis)] - df['vtx.{}'.format(COORDINATE_yaxis)]),
                                                     bins=(bins_abs_resolution, bins_abs_resolution), cmap='viridis', cmin=1)
plt.colorbar(label='Events')
if args.draw_contours:
    plt.contour(counts_EA, extent=[xbins_EA.min(), xbins_EA.max(), ybins_EA.min(), ybins_EA.max()], linewidths=1, colors='red', linestyle='--')

plt.title('Elastic Arms Resolution')
plt.xlabel('|Reco - True| Vertex {} [cm]'.format(COORDINATE_xaxis))
plt.ylabel('|Reco - True| Vertex {} [cm]'.format(COORDINATE_yaxis))
plt.text(5,5, '{}\nAll Interactions'.format(str_det_horn_flux.replace("_", " "), COORDINATE_xaxis, COORDINATE_yaxis),
         fontsize=8, color='red')
plt.text(xbins_EA.max() * 0.35, ybins_EA.max() * 1.1, 'NOvA Simulation', fontsize=14, color='grey')
plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')
# plt.legend(loc='upper right')
plt.subplots_adjust(bottom=0.15, left=0.15)

# plt.show()
for ext in ['pdf', 'png']:
    fig_resolution.savefig(OUTDIR + '/plot_{}_allmodes_AbsResolution_{}{}_ElasticArms.'.
                           format(str_det_horn_flux, COORDINATE_xaxis, COORDINATE_yaxis) + ext, dpi=300)



# plot 2D abs(reco - true) vertex difference for Model
fig_resolution_Model = plt.figure(figsize=(5, 3))

counts_M, ybins_M, xbins_M, image_M = plt.hist2d(abs(df['Model Pred {}'.format(COORDINATE_xaxis)] - df['vtx.{}'.format(COORDINATE_xaxis)]),
                                                     abs(df['Model Pred {}'.format(COORDINATE_yaxis)] - df['vtx.{}'.format(COORDINATE_yaxis)]),
                                                     bins=(bins_abs_resolution, bins_abs_resolution), cmap='viridis', cmin=1)
plt.colorbar(label='Events')
plt.title('Model Prediction Resolution')
plt.xlabel('|Reco - True| Vertex {} [cm]'.format(COORDINATE_xaxis))
plt.ylabel('|Reco - True| Vertex {} [cm]'.format(COORDINATE_yaxis))
plt.text(5,5, '{}\nAll Interactions'.format(str_det_horn_flux.replace("_", " "), COORDINATE_xaxis, COORDINATE_yaxis),
         fontsize=8, color='red')
plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')
# plt.legend(loc='upper right')

plt.subplots_adjust(bottom=0.15, left=0.15)

if args.draw_contours:
    plt.contour(counts_M.transpose(), extent=[xbins_M.min(), xbins_M.max(), ybins_M.min(), ybins_M.max()], linewidths=1, colors='red')

# plt.show()
for ext in ['pdf', 'png']:
    fig_resolution_Model.savefig(OUTDIR + '/plot_{}_allmodes_AbsResolution_{}{}_ModelPred.'.
                           format(str_det_horn_flux, COORDINATE_xaxis, COORDINATE_yaxis) + ext, dpi=300)

print('Done.')


# TODO: make 2D plots of the %-difference too? : (reco - true)/true
#
# # Relative Resolution
# # plot the (reco - true)/true vertex difference for both: Elastic Arms and Model Prediction
# # for (reco - true)/true vertex difference
# bins_relative_resolution = np.arange(-0.2, 0.2, .01)  # edges at +- 20% of the true value.

# fig_resolution = plt.figure(figsize=(5, 3))
#
# hist_EA_all_relres, bins_EA_all_relres, patches_EA_all_relres = plt.hist(
#     (df['vtxEA.{}'.format(COORDINATE)] - df['vtx.{}'.format(COORDINATE)]) / df['vtx.{}'.format(COORDINATE)],
#     bins=bins_relative_resolution, color='black', alpha=0.5, label='Elastic Arms', hatch='//')
# hist_Model_all_relres, bins_Model_all_relres, patches_Model_all_relres = plt.hist(
#     (df['Model Pred {}'.format(COORDINATE)] - df['vtx.{}'.format(COORDINATE)]) / df['vtx.{}'.format(COORDINATE)],
#     bins=bins_relative_resolution, color='orange', alpha=0.5, label='Model Pred.')
# plt.xlabel('(Reco - True)/True [cm]')
# plt.ylabel('Events')
# plt.text(0.13, hist_EA.max() * 0.7, '{} {} {}\nAll Interactions\n {} coordinate'.format(DETECTOR, HORN, FLUX, COORDINATE),
#          fontsize=8)
# plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')
# plt.legend(loc='upper right')
# plt.subplots_adjust(bottom=0.15, left=0.15)
#
# plt.show()
# for ext in ['pdf', 'png']:
#     fig_resolution.savefig(
#         OUTDIR + '/' + RESOLUTION + '/plot_{}_{}_{}_allmodes_{}_RelResolution.'.format(
#             DETECTOR, HORN, FLUX, COORDINATE) + ext, dpi=300)
#
#
# # Interaction type
# # plot the abs(reco - true) resolution of the vertex for each interaction type
# # for both: Elastic Arms and Model Prediction
#
# # TODO: need to save these interaction type plots to a separate dir...!
#
# for i in range(0, len(mode_names)):
#     # ignore the empty Interaction dataframes
#     if df_modes[i].empty:
#         print('skipping ' + mode_names[i])
#         continue
#     fig_resolution_int = plt.figure(figsize=(5, 3))
#
#     df_mode = df_modes[i]  # need to save the dataframe to a variable
#     hist_EA, bins, patches = plt.hist(df_mode['AbsVtxDiff.EA.{}'.format(COORDINATE)], bins=bins_abs_resolution,
#                                       range=(-50, 50), color=mode_colors[mode_names[i]], alpha=0.5,
#                                       label='Elastic Arms', hatch='//')
#     hist_Model, bins_Model, patches_Model = plt.hist(df_mode['AbsVtxDiff.Model.{}'.format(COORDINATE)],
#                                                      bins=bins_abs_resolution, range=(-50, 50),
#                                                      color=mode_colors_Model[mode_names[i]], alpha=0.5,
#                                                      label='Model Pred.')
#     plt.title('{} Interactions'.format(mode_names[i]))
#     plt.xlabel('|Reco.  - True| Vertex [cm]')
#     plt.ylabel('Events')
#     plt.text(35, hist_EA.max() * 0.6, '{} {} {}\n{} coordinate'.format(DETECTOR, HORN, FLUX, COORDINATE), fontsize=8)
#     plt.legend(loc='upper right')
#     plt.subplots_adjust(bottom=0.15, left=0.15)
#     plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')
#
#     plt.show()
#     for ext in ['pdf', 'png']:
#         fig_resolution_int.savefig(OUTDIR + '/' + RESOLUTION + '/plot_{}_{}_{}_{}_AbsResolution_{}.'.
#                                    format(DETECTOR, HORN, FLUX, COORDINATE, mode_names[i]) + ext, dpi=300)
#
#
#
# # plot the (reco - true) resolution of the vertex for each interaction type
# # for both: Elastic Arms and Model Prediction
#
# for i in range(0, len(mode_names)):
#     # ignore the empty Interaction dataframes
#     if df_modes[i].empty:
#         print('skipping ' + mode_names[i])
#         continue
#
#     fig_res_int = plt.figure(figsize=(5, 3))
#
#     df_mode = df_modes[i]
#     hist_EA, bins_EA, patches_EA = plt.hist(
#         df_mode['vtxEA.{}'.format(COORDINATE)] - df_mode['vtx.{}'.format(COORDINATE)], bins=bins_resolution,
#         range=(-50, 50), color=mode_colors[mode_names[i]], alpha=0.5, label='Elastic Arms', hatch='//')
#     hist_Model, bins_Model, patches_Model = plt.hist(
#         df_mode['Model Pred {}'.format(COORDINATE)] - df_mode['vtx.{}'.format(COORDINATE)], bins=bins_resolution,
#         range=(-50, 50), color=mode_colors_Model[mode_names[i]], alpha=0.5, label='Model Pred.')
#     plt.xlabel('Reco. - True Vertex [cm]')
#     plt.ylabel('Events')
#     plt.title('{} Interactions'.format(mode_names[i]))
#     plt.text(20, hist_EA.max() * 0.55, '{} {} {}\n{} coordinate'.format(DETECTOR, HORN, FLUX, COORDINATE), fontsize=8)
#     plt.legend(loc='upper right')
#     plt.subplots_adjust(bottom=0.15, left=0.15)
#     plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')
#
#     plt.show()
#     for ext in ['pdf', 'png']:
#         fig_res_int.savefig(
#             OUTDIR + '/' + RESOLUTION + '/plot_{}_{}_{}_{}_Resolution_{}.'.format(DETECTOR,
#                                                                                   HORN,
#                                                                                   FLUX,
#                                                                                   COORDINATE,
#                                                                                   mode_names[i]) + ext, dpi=300)
#
#
# # plot the relative resolution (reco - true)/true for each interaction type
# # for both: Elastic Arms and Model Prediction
#
#
# for i in range(0, len(mode_names)):
#     # ignore the empty Interaction dataframes
#     if df_modes[i].empty:
#         print('skipping ' + mode_names[i])
#         continue
#     fig_res_int = plt.figure(figsize=(5, 3))
#
#     df_mode = df_modes[i]
#     hist_EA, bins, patches = plt.hist(
#         (df_mode['vtxEA.{}'.format(COORDINATE)] - df_mode['vtx.{}'.format(COORDINATE)]) / df_mode[
#             'vtx.{}'.format(COORDINATE)], bins=bins_relative_resolution, range=(-50, 50),
#         color=mode_colors[mode_names[i]], alpha=0.5, label='Elastic Arms', hatch='//')
#     hist_Model, bins_Model, patches_Model = plt.hist(
#         (df_mode['Model Pred {}'.format(COORDINATE)] - df_mode['vtx.{}'.format(COORDINATE)]) / df_mode[
#             'vtx.{}'.format(COORDINATE)], bins=bins_relative_resolution, range=(-50, 50),
#         color=mode_colors_Model[mode_names[i]], alpha=0.5, label='Model Pred.')
#     plt.xlabel('(Reco - True)/True Vertex {} [cm]'.format(COORDINATE))
#     plt.ylabel('Events')
#     plt.text(0.1, hist_EA.max() * 0.55, '{} {} {}\n{} coordinate'.format(DETECTOR, HORN, FLUX, COORDINATE), fontsize=8)
#     plt.legend(loc='upper right')
#     plt.subplots_adjust(bottom=0.15, left=0.15)
#     plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')
#
#     plt.show()
#     for ext in ['pdf', 'png']:
#         fig_res_int.savefig(OUTDIR + '/' + RESOLUTION + '/plot_{}_{}_{}_{}_RelResolution_{}.'.
#                             format(DETECTOR, HORN, FLUX, COORDINATE, mode_names[i]) + ext, dpi=300)
#
#
# # TODO: should look into making a box plot somehow....
# # make the width of the box the 10% difference from the mean or something like it?
