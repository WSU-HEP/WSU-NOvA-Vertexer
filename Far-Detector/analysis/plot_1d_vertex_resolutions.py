#!/usr/bin/env python
# coding: utf-8
# # plot_1d_vertex_resolutions.py
# M. Dolce. 
# Feb. 2024
# 
# Makes plots to compare the true and E.A. reconstructed vertex for ONE coordinate.
# NOTE: the abs(resolution) uses the mean from the histogram, not the dataframe.
# NOTE: (reco-true)/true does not report RMS.
# NOTE: the --test_file must be the SAME file used to make the predictions!

#  $ PY37 plot_1d_vertex_resolutions.py --pred_file ... --outdir ... --coordinate
# TODO: draw a line where the mean is....?
# TODO: should look into making a box plot somehow....
#TODO: wrap each of these plot making chunks into a function and then have a main() that calls each...

# TODO: I am not applying the filter I used to do the training! Do I need to do that?


import os
import sys
import argparse
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ML Vtx utils
import utils.data_processing as dp
import utils.iomanager as io
import utils.plot

# collect the arguments for this macro. the horn and swap options are required.
parser = argparse.ArgumentParser()
parser.add_argument("--pred_file", help="full path to CSV file of vertex predictions", default="", type=str)
parser.add_argument("--outdir", help="full path to output directory", default="", type=str)
parser.add_argument("--coordinate", help="the coordinate you want to plot", default="", type=str)
parser.add_argument("--test_file", help="full path to file used for testing/inference",
                    default="/home/k948d562/NOvA-shared/FD-Training-Samples/{}-Nominal-{}-{}/test/trimmed_h5_R20-11-25-prod5.1reco.j_{}-Nominal-{}-{}_27_of_28.h5",
                    type=str)

meg = parser.add_mutually_exclusive_group()
meg.add_argument("--nonswap",  required=False, default=False, action='store_true', help="For 'Combined' only, make predictions with the nonswap inference file")
meg.add_argument("--fluxswap", required=False, default=False, action='store_true', help="For 'Combined' only, make predictions with the fluxswap inference file")
args = parser.parse_args()

args.pred_file = args.pred_file
args.test_file = args.test_file
args.outdir = args.outdir
args.coordinate = args.coordinate
C = args.coordinate.upper()  # less typing

# get the list of Interaction types. Skipping Unknown Mode
int_modes = utils.plot.ModeType.get_known_int_modes()
# colors for each mode (as close as possible to what they are in NOvA Style)
colors = utils.plot.NuModeColors()


print(f'pred_file: {args.pred_file}')
print(f'Making plots for coordinate {C}.....')

# vars to save the column name strings of DataFrame.
true_C = f'True {C}'
reco_C = f'Reco {C}'
model_C = f'Model Pred {C}'
abs_vtx_diff_EA_C = f'AbsVtxDiff.EA.{C}'
abs_vtx_diff_model_C = f'AbsVtxDiff.Model.{C}'

# filename without the CSV and path.
pred_filename_prefix = args.pred_file.split('/')[-1].split('.')[0]  # get the filename only and remove the .csv
DET, HORN, FLUX = io.IOManager.get_det_horn_and_flux_from_string(args.pred_file)

# load the CSV file of the model predictions.
df = dp.ModelPrediction.load_pred_csv_file(args.pred_file)


# If you have 'Combined', determine which file to make predictions on: Fluxswap OR Nonswap.
flavors = {'Nonswap': 'numus', 'Fluxswap': 'nues'}
if FLUX == 'Combined' and args.nonswap == True:
    FLUX = 'Nonswap'
elif FLUX == 'Combined' and args.fluxswap == True:
    FLUX = 'Fluxswap'
else:
    print('ERROR: Unknown flux option to predict. Exiting.')
    sys.exit(1)


# want the mode from the test file (same one used to make the predictions).
test_file = args.test_file.format(DET, HORN, FLUX, DET, HORN, FLUX)
print(f'test_file: {test_file}')
with h5py.File(test_file, mode='r') as f:
    df_mode = pd.DataFrame({'Mode': f['mode'][:]})

# define path to save some plots (the local dir).
vtx_abs_diff_ea_temp, vtx_abs_diff_model_temp = dp.ModelPrediction.create_abs_vtx_diff_columns(df, C)

# new additions to the main DataFrame as new columns
df = pd.concat([df, vtx_abs_diff_ea_temp, vtx_abs_diff_model_temp, df_mode], axis=1)

# list of dataframes for each mode has both NC and CC.
print("Creating 'df_modes'...........")
df_modes = list()
for i in range(len(int_modes)):
    df_modes.append(df[df['Mode'] == i])
# assert are no kUnknownMode events...
assert (len(df[df['Mode'] == -1]) == 0)  # 'there are Unknown events. Stopping'

###################
# TODO: not essential, but in case we want to make distributions from each neutrino flavor

# output directory
print('Output Directory: ', args.outdir)
OUTDIR = utils.plot.make_output_dir(args.outdir, 'resolution', pred_filename_prefix)
str_det_horn = f'{DET}_{HORN}'


print('Creating plots..................')
############# Resolution plots ############
# Histograms. Define the binning here...
# for reco - true vertex. 
bins_resolution = np.arange(-40, 40, 1)  # 1 bin per cm.
# for abs(reco - true) vertex difference
bins_abs_resolution = np.arange(0, 50, 1)  # 1 bin per cm.
# for (reco - true)/true vertex difference
bins_relative_resolution = np.arange(-0.2, 0.2, .01)  # edges at +- 20%

############### Metrics ###############
# directly calculate the mean and RMS from the dataframe.
mean_Model = np.mean(df[model_C] - df[true_C])
rms_Model = np.sqrt(np.mean((df[model_C] - df[true_C]) ** 2))

mean_EA = np.mean(df[reco_C] - df[true_C])
rms_EA = np.sqrt(np.mean((df[reco_C] - df[true_C]) ** 2))

# percent differences. Note, RMS of the residual is not reported.
mean_EA_all_relres = ((df[reco_C] - df[true_C]) / df[true_C]).mean()
mean_float_EA_all_relres = float(mean_EA_all_relres)
mean_Model_all_relres = ((df[model_C] - df[true_C]) / df[true_C]).mean()
mean_float_Model_all_relres = float(mean_Model_all_relres)


# ------------- All interactions separate models -------------
# ELASTIC ARMS, ONLY
# plot the resolution of the vertex for each interaction type on single plot.
fig_res_int_EA = plt.figure(figsize=(10, 8))
for i in range(0, len(int_modes)):
    # ignore the empty Interaction dataframes 
    if df_modes[i].empty:
        continue
    df_mode = df_modes[i]
    hist_EA_all, bins_all, patches_all = plt.hist(
        df_mode[reco_C] - df_mode[true_C],
        bins=bins_resolution,
        range=(-50, 50),
        color=colors.mode_colors_EA[utils.plot.ModeType.name(i)],
        alpha=0.5,
        label=utils.plot.ModeType.name(i))
plt.title('Elastic Arms Vertex Resolution')
plt.xlabel(f'Reco. - True Vertex {C} [cm]')
plt.ylabel('Events')
plt.text(15, 9e3, f'{DET} {HORN}_{flavors[FLUX]}\nElastic Arms\n{C} coordinate', fontsize=12)
plt.text(15, 5e3, f'Mean: {mean_EA:.2f} cm\nRMS: {rms_EA:.2f} cm', fontsize=12)
plt.legend(loc='upper right')
plt.subplots_adjust(bottom=0.15, left=0.15)
plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')
# plt.show()
for ext in ['pdf', 'png']:
    fig_res_int_EA.savefig(
        OUTDIR + f'/plot_{str_det_horn}_{flavors[FLUX]}_allmodes_Resolution_{C}_ElasticArms.' + ext, dpi=300)
plt.close(fig_res_int_EA)

# Model Prediction, ONLY
# plot the resolution of the vertex for each interaction type on single plot.
fig_res_int_Model = plt.figure(figsize=(10, 8))
for i in range(0, len(int_modes)):
    if df_modes[i].empty: continue
    df_mode = df_modes[i]
    hist_Model_all, bins_all, patches_all = plt.hist(
        df_mode[model_C] - df_mode[true_C],
        bins=bins_resolution,
        range=(-50, 50),
        color=colors.mode_colors_Model[utils.plot.ModeType.name(i)],
        alpha=0.5,
        label=utils.plot.ModeType.name(i))
plt.title('Model Prediction Resolution')
plt.xlabel(f'Reco. - True Vertex {C} [cm]')
plt.ylabel('Events')
plt.text(25, 15e3, f'{DET} {HORN}_{flavors[FLUX]}\nModel Prediction\n{C} coordinate', fontsize=12)
plt.text(25, 5e3, f'Mean: {mean_Model:.2f} cm\nRMS: {rms_Model:.2f} cm', fontsize=12)
plt.legend(loc='upper right')
plt.subplots_adjust(bottom=0.15, left=0.15)
plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')
# plt.show()
for ext in ['pdf', 'png']:
    fig_res_int_Model.savefig(
        OUTDIR + f'/plot_{str_det_horn}_{flavors[FLUX]}_allmodes_Resolution_{C}_ModelPred.' + ext, dpi=300)
plt.close(fig_res_int_Model)
# ------------- All interactions separate model end -------------

# ------------ Individual interactions Overlaid, Model + E.A., [Abs, Res, RelRes]  -------------
# Abs(resolution)
# plot the abs(reco - true) vertex difference for both: Elastic Arms and Model Prediction
fig_resolution = plt.figure(figsize=(5, 3))
hist_EA_abs, bins_EA_abs, patches_EA_abs = plt.hist(df[abs_vtx_diff_EA_C],
                                        bins=bins_abs_resolution,
                                        range=(-50, 50),
                                        color='black',
                                        alpha=0.5,
                                        label='Elastic Arms',
                                        hatch='//')
hist_Model_abs, bins_Model_abs, patches_Model_abs = plt.hist(df[abs_vtx_diff_model_C],
                                                 bins=bins_abs_resolution,
                                                 range=(-50, 50),
                                                 color='orange',
                                                 alpha=0.5,
                                                 label='Model Pred.')
plt.xlabel('|Reco - True| Vertex [cm]')
plt.ylabel('Events')
plt.text(30, hist_EA_abs.max() * 0.6, f'{DET} {HORN}_{flavors[FLUX]}\nAll Interactions\n {C} coordinate', fontsize=8)
plt.text(30, hist_EA_abs.max() * 0.45, f'Mean E.A.: {df[abs_vtx_diff_EA_C].mean():.2f} cm\nMean Model: {df[abs_vtx_diff_model_C].mean():.2f} cm', fontsize=8)
plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')
plt.legend(loc='upper right')
plt.subplots_adjust(bottom=0.15, left=0.15)
# plt.show()
for ext in ['pdf', 'png']:
    fig_resolution.savefig(OUTDIR + f'/plot_{str_det_horn}_{flavors[FLUX]}_allmodes_{C}_AbsResolution.' + ext, dpi=300)
plt.close(fig_resolution)

# Resolution
# plot the (reco - true) vertex difference for both: Elastic Arms and Model Prediction
fig_resolution = plt.figure(figsize=(5, 3))

hist_EA_all_res, bins_EA_all_res, patches_EA_all_res = plt.hist(
    df[reco_C] - df[true_C],
    bins=bins_resolution,
    color='black',
    alpha=0.5,
    label='Elastic Arms',
    hatch='//')

hist_Model_all_res, bins_Model_all_res, patches_Model_all_res = plt.hist(
    df[model_C] - df[true_C],
    bins=bins_resolution,
    color='orange',
    alpha=0.5,
    label='Model Pred.')
plt.text(14, 4e4, f'Mean E.A.: {mean_EA:.2f} cm\nRMS E.A.: {rms_EA:.2f} cm', fontsize=8)
plt.text(14, 2e4, f'Mean Model: {mean_Model:.2f} cm\nRMS Model: {rms_Model:.2f} cm', fontsize=8)
plt.xlabel(f'(Reco - True) Vertex {C} [cm]')
plt.ylabel('Events')
plt.text(-40, hist_EA_all_res.max() * 0.75, f'{DET} {HORN}_{flavors[FLUX]}\nAll Interactions\n {C} coordinate', fontsize=8)
plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')
plt.legend(loc='upper right')
plt.subplots_adjust(bottom=0.15, left=0.15)

# plt.show()
for ext in ['pdf', 'png']:
    fig_resolution.savefig(
        OUTDIR + f'/plot_{str_det_horn}_{flavors[FLUX]}_allmodes_{C}_Resolution.' + ext,
        dpi=300)
plt.close(fig_resolution)

# Relative Resolution
# plot the (reco - true)/true vertex difference for both:
# Elastic Arms and Model Prediction
fig_resolution = plt.figure(figsize=(5, 3))
hist_EA_all_relres, bins_EA_all_relres, patches_EA_all_relres = plt.hist(
    (df[reco_C] - df[true_C]) / df[true_C],
    bins=bins_relative_resolution,
    color='black',
    alpha=0.5,
    label='Elastic Arms',
    hatch='//')
hist_Model_all_relres, bins_Model_all_relres, patches_Model_all_relres = plt.hist(
    (df[model_C] - df[true_C]) / df[true_C],
    bins=bins_relative_resolution,
    color='orange',
    alpha=0.5,
    label='Model Pred.')
plt.xlabel(f'(Reco - True)/True {C}')
plt.ylabel('Events')
plt.text(0.05, hist_EA_all_relres.max() * 0.7, f'{DET} {HORN}_{flavors[FLUX]}\nAll Interactions\n {C} coordinate', fontsize=8)
plt.text(0.05,hist_EA_all_relres.max() * 0.5,
         f'Mean E.A.: {mean_float_EA_all_relres:.2f} cm\nMean Model: {mean_float_Model_all_relres:.2f} cm',
         fontsize=8)
plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')
plt.legend(loc='upper right')
plt.subplots_adjust(bottom=0.15, left=0.15)
# plt.show()
for ext in ['pdf', 'png']:
    fig_resolution.savefig(
        OUTDIR + f'/plot_{str_det_horn}_{flavors[FLUX]}_allmodes_{C}_RelResolution.' + ext, dpi=300)
plt.close(fig_resolution)
# ------------ Individual interactions Overlaid, Model + E.A., [Abs, Res, RelRes]  -------------




# ------------ Individual interactions Separate, Model + E.A., [Abs, Res, RelRes]  -------------
# plot the abs(reco - true) resolution of the vertex for EACH INTERACTION TYPE
# for both: Elastic Arms and Model Prediction
for i in range(0, len(int_modes)):
    # ignore the empty Interaction dataframes 
    if df_modes[i].empty:
        continue
    fig_resolution_int = plt.figure(figsize=(5, 3))
    df_mode = df_modes[i]  # need to save the dataframe to a variable
    hist_EA, bins, patches = plt.hist(df_mode[abs_vtx_diff_EA_C],
                                      bins=bins_abs_resolution,
                                      range=(-50, 50),
                                      color=colors.get_color(utils.plot.ModeType.name(i), False),
                                      alpha=0.5,
                                      label='Elastic Arms',
                                      hatch='//')
    hist_Model, bins_Model, patches_Model = plt.hist(df_mode[abs_vtx_diff_model_C],
                                                     bins=bins_abs_resolution,
                                                     range=(-50, 50),
                                                     color=colors.get_color(utils.plot.ModeType.name(i), True),
                                                     alpha=0.5,
                                                     label='Model Pred.')
    # NOTE: no RMS for these plots.
    plt.title(f'{utils.plot.ModeType.name(i)} Interactions')
    plt.xlabel('|Reco.  - True| Vertex [cm]')
    plt.ylabel('Events')
    plt.text(35, hist_EA.max() * 0.6, f'{DET} {HORN}_{flavors[FLUX]}\n{C} coordinate', fontsize=8)
    plt.text(35, hist_EA.max() * 0.45, f'Mean E.A.: {df_mode[abs_vtx_diff_EA_C].mean():.2f} cm\nMean Model: {df_mode[abs_vtx_diff_model_C].mean():.2f} cm', fontsize=8)
    plt.legend(loc='upper right')
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')
    # plt.show()
    for ext in ['pdf', 'png']:
        fig_resolution_int.savefig(OUTDIR + f'/plot_{str_det_horn}_{flavors[FLUX]}_{C}_AbsResolution_{utils.plot.ModeType.name(i)}.' + ext, dpi=300)
    plt.close(fig_resolution_int)

# FOR EACH INTERACTION TYPE
# plot the (reco - true) resolution of the vertex
# for both: Elastic Arms and Model Prediction
for i in range(0, len(int_modes)):
    if df_modes[i].empty:
        continue
    fig_res_int = plt.figure(figsize=(5, 3))
    df_mode = df_modes[i]
    hist_EA, bins_EA, patches_EA = plt.hist(
        df_mode[reco_C] - df_mode[true_C],
        bins=bins_resolution,
        range=(-50, 50),
        color=colors.get_color(utils.plot.ModeType.name(i), False),
        alpha=0.5,
        label='Elastic Arms',
        hatch='//')
    hist_Model, bins_Model, patches_Model = plt.hist(
        df_mode[model_C] - df_mode[true_C],
        bins=bins_resolution,
        range=(-50, 50),
        color=colors.get_color(utils.plot.ModeType.name(i), True),
        alpha=0.5,
        label='Model Pred.')
    # Calculate the mean and RMS individually INSIDE the loop here
    mean_int_EA = np.mean(df_mode[reco_C] - df_mode[true_C])
    mean_int_Model = np.mean(df_mode[model_C] - df_mode[true_C])
    rms_int_EA = np.sqrt(np.mean((df_mode[reco_C] - df_mode[true_C]) ** 2))
    rms_int_Model = np.sqrt(np.mean((df_mode[model_C] - df_mode[true_C]) ** 2))
    plt.xlabel('Reco. - True Vertex [cm]')
    plt.ylabel('Events')
    plt.title(f'{utils.plot.ModeType.name(i)} Interactions')
    plt.text(20, hist_EA.max() * 0.55, f'{DET} {HORN}_{flavors[FLUX]}\n{C} coordinate', fontsize=8)
    plt.text(20, hist_EA.max() * 0.45, f'Mean E.A.: {mean_int_EA:.2f} cm\nMean Model: {mean_int_Model:.2f} cm', fontsize=8)
    plt.text(20, hist_EA.max() * 0.3, f'RMS E.A.: {rms_int_EA:.2f} cm\nRMS Model: {rms_int_Model:.2f} cm', fontsize=8)
    plt.legend(loc='upper right')
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')
    # plt.show()
    for ext in ['pdf', 'png']:
        fig_res_int.savefig(OUTDIR + f'/plot_{str_det_horn}_{flavors[FLUX]}_{C}_Resolution_{utils.plot.ModeType.name(i)}.' + ext, dpi=300)
        plt.close(fig_res_int)

# FOR EACH INTERACTION TYPE
# plot the relative resolution (reco - true)/true
# for both: Elastic Arms and Model Prediction
for i in range(0, len(int_modes)):
    if df_modes[i].empty:
        continue
    fig_res_int = plt.figure(figsize=(5, 3))
    df_mode = df_modes[i]
    hist_EA, bins, patches = plt.hist(
        (df_mode[reco_C] - df_mode[true_C]) / df_mode[
            true_C], bins=bins_relative_resolution, range=(-50, 50),
        color=colors.get_color(utils.plot.ModeType.name(i), False), alpha=0.5, label='Elastic Arms', hatch='//')
    hist_Model, bins_Model, patches_Model = plt.hist(
        (df_mode[model_C] - df_mode[true_C]) / df_mode[
            true_C], bins=bins_relative_resolution, range=(-50, 50),
        color=colors.get_color(utils.plot.ModeType.name(i), True), alpha=0.5, label='Model Pred.')

    # NOTE: we do not report the RMS of a residual here, that doesn't make sense...
    # NOTE: RMS of the residual is not reported. Again, inside the loop.
    mean_EA_int_relres = float(((df_mode[reco_C] - df_mode[true_C]) / df_mode[
        true_C]).mean())

    mean_Model_int_relres = float(((df_mode[model_C] - df_mode[true_C]) / df_mode[
        true_C]).mean())

    plt.xlabel(f'(Reco - True)/True Vertex {C}')
    plt.ylabel('Events')
    plt.title(f'{utils.plot.ModeType.name(i)} Interactions')
    plt.text(0.05, hist_EA.max() * 0.55, f'{DET} {HORN} {flavors[FLUX]}\n{C} coordinate', fontsize=8)
    plt.text(0.05, hist_EA.max() * 0.4, f'Mean E.A.: {mean_EA_int_relres:.2f} cm\nMean Model: {mean_Model_int_relres:.2f} cm', fontsize=8)
    plt.legend(loc='upper right')
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')
    # plt.show()
    for ext in ['pdf', 'png']:
        fig_res_int.savefig(OUTDIR + f'/plot_{str_det_horn}_{flavors[FLUX]}_{C}_RelResolution_{utils.plot.ModeType.name(i)}.' + ext, dpi=300)
    plt.close(fig_res_int)

print('Done!')
