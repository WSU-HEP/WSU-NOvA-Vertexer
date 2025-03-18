#!/usr/bin/env python
# coding: utf-8
# # plot_1d_vertex_resolutions.py
# A. Yahaya 
# Jan. 2025
# 
# Makes plots to compare the true, model pred and Reco (EA) radial distance
# NOTE: the --test_file must be the SAME file used to make the predictions!

#  $ PY37 XYZ_radial_plot.py  --pred_file ... --outdir ... 


# ML Vtx utils
import utils.data_processing as dp
import utils.iomanager as io
import utils.plot

import argparse
import os
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# collect the arguments for this macro. the horn and swap options are required.
parser = argparse.ArgumentParser()
parser.add_argument("--pred_file", help="full path to CSV file of vertex predictions", default="", type=str)
parser.add_argument("--outdir", help="full path to output directory", default="", type=str)
parser.add_argument("--test_file", help="full path to file used for testing/inference",
                    default="/home/k948d562/NOvA-shared/FD-Training-Samples/{}-Nominal-{}-{}/test/trimmed_h5_R20-11-25-prod5.1reco.j_{}-Nominal-{}-{}_27_of_28.h5",
                    type=str)
args = parser.parse_args()

int_modes = utils.plot.ModeType.get_known_int_modes()

colors = utils.plot.NuModeColors()

args.pred_file = args.pred_file
args.test_file = args.test_file
args.outdir = args.outdir


# get the list of Interaction types. Skipping Unknown Mode
int_modes = utils.plot.ModeType.get_known_int_modes()
# colors for each mode (as close as possible to what they are in NOvA Style)
colors = utils.plot.NuModeColors()

# filename without the CSV and path.
pred_filename_prefix = args.pred_file.split('/')[-1].split('.')[0]  # get the filename only and remove the .csv
DET, HORN, FLUX = io.IOManager.get_det_horn_and_flux_from_string(args.pred_file)

# load the CSV file of the model predictions.
df = dp.ModelPrediction.load_pred_csv_file(args.pred_file)

#want the mode from the test file
test_file = args.test_file.format(DET, HORN, FLUX, DET, HORN, FLUX)
print('test_file:{}'.format(test_file))
with h5py.File(test_file, mode='r') as f:
    df_mode = pd.DataFrame({'Mode': f['mode'][:]})

# output directory
print('Output Directory: ', args.outdir)
OUTDIR = utils.plot.make_output_dir(args.outdir, 'radial', pred_filename_prefix)
str_det_horn = '{}_{}'.format(DET, HORN)


df=pd.concat([df, df_mode], axis=1)
df_modes=list()
for i in range(len(int_modes)):
    print('Int Type, Code: ', utils.plot.ModeType.name(i), '', int_modes[i])
    df_modes.append(df[df['Mode'] == i])

assert (len(df[df['Mode'] == -1]) == 0)


# for radial plots
bins_resolution = np.arange(0, 100, 1)  # 1 bin per cm


#radial Calculations
model_radial= np.sqrt(((df['Model Pred X'] - df['True X']) ** 2) + ((df['Model Pred Y'] - df['True Y']) ** 2) + ((df['Model Pred Z'] - df['True Z']) ** 2))
EA_radial = np.sqrt(((df['Reco X'] - df['True X']) ** 2) + ((df['Reco Y'] - df['True Y']) ** 2) + ((df['Reco Z'] - df['True Z']) ** 2))

# plot the (reco - true) vertex difference for both: Elastic Arms and Model Prediction
fig_resolution = plt.figure(figsize=(5, 3))

hist_EA_all_res, bins_EA_all_res, patches_EA_all_res = plt.hist(
    EA_radial,
    bins=bins_resolution,
    color='black',
    alpha=0.5,
    label='Elastic Arms',
    hatch='//')

hist_Model_all_res, bins_Model_all_res, patches_Model_all_res = plt.hist(
    model_radial,
    bins=bins_resolution,
    color='orange',
    alpha=0.5,
    label='Model Pred.')

# Total number of events
total_events = len(df)

# Calculate total events within 10cm and 20cm for Elastic Arms
events_within_10cm_EA = np.sum(EA_radial <= 10)
events_within_20cm_EA = np.sum(EA_radial <= 20)

# Calculate total events within 10cm and 20cm for Model Prediction
events_within_10cm_Model = np.sum(model_radial <= 10)
events_within_20cm_Model = np.sum(model_radial <= 20)

# Calculate percentages
percent_10cm_EA = (events_within_10cm_EA / total_events) * 100
percent_20cm_EA = (events_within_20cm_EA / total_events) * 100
percent_10cm_Model = (events_within_10cm_Model / total_events) * 100
percent_20cm_Model = (events_within_20cm_Model / total_events) * 100


# Add percentages as text annotations
plt.text(50, hist_EA_all_res.max() * 0.55,
        f'Elastic Arms:\n10 cm: {percent_10cm_EA:.2f}%\n20 cm: {percent_20cm_EA:.2f}%',
        fontsize=12, color='black')

plt.text(50, hist_Model_all_res.max() * 0.35,
        f'Model Pred.:\n10 cm: {percent_10cm_Model:.2f}%\n20 cm: {percent_20cm_Model:.2f}%',
        fontsize=12, color='orange')

#labels
plt.xlabel('(Radial Distance) cm')
plt.ylabel('Events')
plt.text(0, hist_EA_all_res.max() * 0.75, '{} {}\nAll Interactions'.format(
    DET, HORN), fontsize=8)
plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')
plt.legend(loc='upper right')
plt.subplots_adjust(bottom=0.15, left=0.15)


# plt.show()
for ext in ['pdf', 'png']:
    fig_resolution.savefig(
        OUTDIR + '/plot_{}_Allmodes_Radial_Distance.'.format(str_det_horn) + ext,
        dpi=300)

#For interaction types
for i in range(0, len(int_modes)):
    if df_modes[i].empty:
        continue
    fig_res_int=plt.figure(figsize=(5, 3))
    df_mode = df_modes[i]

    radial_dist_EA = np.sqrt(((df_mode['Reco X'] - df_mode['True X'])**2) + 
                             ((df_mode['Reco Y'] - df_mode['True Y'])**2) +
                             ((df_mode['Reco Z'] - df_mode['True Z'])**2))
    radial_dist_Model = np.sqrt(((df_mode['Model Pred X'] - df_mode['True X'])**2) +
                                ((df_mode['Model Pred Y'] - df_mode['True Y'])**2) +
                                ((df_mode['Model Pred Z'] - df_mode['True Z'])**2))

    hist_EA, bins_EA, patches_EA = plt.hist(
         radial_dist_EA,
         bins= bins_resolution,
         color=colors.get_color(utils.plot.ModeType.name(i), False),
         alpha=0.5,
         label='Elastic Arms',
         hatch='//')
    hist_Model, bins_Model, patches_Model = plt.hist(
         radial_dist_Model,
         bins=bins_resolution,
         color=colors.get_color(utils.plot.ModeType.name(i), True),
         alpha=0.5,
         label='Model Pred')


#For the  mode of interactions
for i in range(0, len(int_modes)):
    if df_modes[i].empty:
        continue
    fig_res_int = plt.figure(figsize=(5, 3))
    df_mode = df_modes[i]

    radial_dist_EA = np.sqrt(((df_mode['Reco X'] - df_mode['True X']) ** 2) + 
                             ((df_mode['Reco Y'] - df_mode['True Y']) ** 2) + 
                             ((df_mode['Reco Z'] - df_mode['True Z']) ** 2))

    radial_dist_Model = np.sqrt(((df_mode['Model Pred X'] - df_mode['True X']) ** 2) + 
                                ((df_mode['Model Pred Y'] - df_mode['True Y']) ** 2) + 
                                ((df_mode['Model Pred Z'] - df_mode['True Z']) ** 2))



    hist_EA, bins_EA, patches_EA = plt.hist(
        radial_dist_EA,
        bins=bins_resolution,
        range=(-50, 50),
        color=colors.get_color(utils.plot.ModeType.name(i), False),
        alpha=0.5,
        label='Elastic Arms',
        hatch='//')
    hist_Model, bins_Model, patches_Model = plt.hist(
         radial_dist_Model,
        bins=bins_resolution,
        range=(-50, 50),
        color=colors.get_color(utils.plot.ModeType.name(i), True),
        alpha=0.5,
        label='Model Pred.')


    count_10cm_EA = np.sum(radial_dist_EA <= 10)
    count_20cm_EA = np.sum(radial_dist_EA <= 20)
    count_30cm_EA = np.sum(radial_dist_EA <= 30)

    count_10cm_Model = np.sum(radial_dist_Model <= 10)
    count_20cm_Model = np.sum(radial_dist_Model <= 20)
    count_30cm_Model = np.sum(radial_dist_Model <= 30)


    total_event_EA = len(radial_dist_EA)
    total_event_Model= len(radial_dist_Model)

    perc_10cm_EA = (count_10cm_EA/total_event_EA)*100 if total_event_EA > 0 else 0

    perc_10cm_Model = (count_10cm_Model/total_event_Model)*100 if total_event_Model > 0 else 0
    perc_20cm_Model = (count_20cm_Model/total_event_Model)*100 if total_event_Model > 0 else 0
    perc_30cm_Model = (count_30cm_Model/total_event_Model)*100 if total_event_Model > 0 else 0
    perc_20cm_EA = (count_20cm_EA/total_event_EA)*100 if total_event_EA > 0 else 0
    perc_30cm_EA = (count_30cm_EA/total_event_EA)*100 if total_event_EA > 0 else 0

    #Plot labels
    plt.xlabel('Radial Distance [cm]')
    plt.ylabel('Events')
    plt.title('{} Interaction'.format(utils.plot.ModeType.name(i)))
    plt.text(20, hist_EA.max()*0.55, '{} {}'.format(DET, HORN), fontsize=6)
    plt.text(20, hist_EA.max()*0.35, 'Events within 10cm: E.A. ({:.2f}%)\n Model ({:.2f}%)\n'.format(perc_10cm_EA, perc_10cm_Model) + 
            'Events within 20cm: E.A. ({:.2f}%)\n  Model ({:.2f}%)\n'.format(perc_20cm_EA, perc_20cm_Model) +
            'Events within 30cm: E.A. ({:.2f}%)\n Model ({:.2f}%)\n'.format(perc_30cm_EA, perc_30cm_Model))

    plt.xlabel('Radial Distance [cm]')
    plt.ylabel('Events')
    plt.title('{} Interactions'.format(utils.plot.ModeType.name(i)))
    plt.text(20, hist_EA.max() * 0.55, '{} {}'.format(DET, HORN), fontsize=8)
    plt.text(20, hist_EA.max() * 0.45, 'Events ≤10cm: E.A. {} / Model {}\nEvents ≤20cm: E.A. {} / Model {}\nEvents ≤30cm: E.A. {} / Model {}'.format(
        count_10cm_EA, count_10cm_Model, count_20cm_EA, count_20cm_Model, count_30cm_EA, count_30cm_Model), fontsize=8)


    plt.legend(loc='upper right')
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')

    # plt.show()
    for ext in ['pdf', 'png']:
        fig_res_int.savefig(
                           OUTDIR + '/plot_{}_{}_Interaction_Radial_Distance.'.format(str_det_horn, utils.plot.ModeType.name(i)) + ext,
                           dpi=300)
        fig_res_int.savefig(OUTDIR + '/plot_{}_Radial_Distance_{}.'.format(str_det_horn,
                                                                              utils.plot.ModeType.name(i)) + ext, dpi=300)


