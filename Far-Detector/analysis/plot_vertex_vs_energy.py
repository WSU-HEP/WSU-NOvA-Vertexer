#!/usr/bin/env python
# coding: utf-8
# # plot_vertex_vs_energy.py
# M. Dolce. 
# Feb. 2024
# 
# Makes plots to compare the true and E.A. reconstructed vertex (from a specified coordinate)
# as a function of the neutrino energy.
# NOTE: this makes resolution plots of ONE COORDINATE.
# 
# This validation is using `file_27_of_28.h5` as Ashley indicated.
# 
# $ PY37 plot_vertex_vs_energy.py --pred_file <path/to/CSV/predictions/file>  --outdir $PLOTS/...  --coordinate [x, y, z]
# 

import utils.data_processing as dp
import utils.iomanager as io
import utils.plot

import argparse
import h5py
import matplotlib.pyplot as plt
import pandas as pd


# collect the arguments for this macro. the horn and swap options are required.
parser = argparse.ArgumentParser()
parser.add_argument("--pred_file", help="the CSV file of vertex predictions", default="", type=str)
parser.add_argument("--outdir", help="full path to output directory", default="", type=str)
parser.add_argument("--coordinate", help="the coordinate you want to plot", default="", type=str)
parser.add_argument("--test_file", help="full path to file used for testing/inference",
                    default="/home/k948d562/NOvA-shared/FD-Training-Samples/{}-Nominal-{}-{}/test/trimmed_h5_R20-11-25-prod5.1reco.j_{}-Nominal-{}-{}_27_of_28.h5",
                    type=str)
args = parser.parse_args()

C = args.coordinate.upper()

int_modes = utils.plot.ModeType.get_known_int_modes()
colors = utils.plot.NuModeColors()

print('pred_file: {}'.format(args.pred_file))
print(f'Making plots for coordinate: {C}.....')

# filename without the CSV and path.
pred_filename_prefix = args.pred_file.split('/')[-1].split('.')[0]  # get the filename only and remove the .csv
DET, HORN, FLUX = io.IOManager.get_det_horn_and_flux_from_string(args.pred_file)

# Load the CSV file of the model predictions.
df = dp.ModelPrediction.load_pred_csv_file(args.pred_file)

# define the path to the validation files
test_file = args.test_file.format(DET, HORN, FLUX, DET, HORN, FLUX)
print('test_file: {}'.format(test_file))

with h5py.File(test_file, 'r') as f:
    df_test_file = pd.DataFrame({'E': f['E'][:],
                       'PDG': f['pdg'][:],
                       'Interaction': f['interaction'][:],
                       'isCC': f['iscc'][:],
                       'Mode': f['mode'][:],
                       'NCid': f['ncid'][:],
                       'FinalState': f['finalstate'][:],
                       'Px': f['p.px'][:],
                       'Py': f['p.py'][:],
                       'Pz': f['p.pz'][:],
                       })
print("Number of events: ", len(df_test_file['E']))  # count the number of events in the file


# define path to save some plots (the local dir).
OUTDIR = utils.plot.make_output_dir(args.outdir, 'vertex-vs-energy', pred_filename_prefix)

# return two DFs, each the Abs(diff) for E.A and Model.
vtx_abs_diff_ea_temp, vtx_abs_diff_model_temp = dp.ModelPrediction.create_abs_vtx_diff_columns(df, C)

df = pd.concat([df, df_test_file, vtx_abs_diff_ea_temp, vtx_abs_diff_model_temp], axis=1)

# create a column of the Energy within NOvA's range/relevance -- for convenience.
E_nova = df[df['E'] < 5]
df['E NOvA'] = E_nova['E']
df.describe()


# list of dataframes for each mode has both NC and CC.
print("Creating 'df_modes'...........")
df_modes = list()
for i in range(len(int_modes)):
    df_modes.append(df[df['Mode'] == i])
# assert are no kUnknownMode events...
assert (len(df[df['Mode'] == -1]) == 0)  # 'there are Unknown events. Stopping'


# # Vertex vs. Energy plots
# Plot the average "distance" between the E.A. Vertex and the True Vtx.
for i in range(0, len(int_modes)):
    print('creating plots for.......', utils.plot.ModeType.name(i))
    fig_Enu = plt.figure(figsize=(7, 5))
    ax_VtxEnergy = fig_Enu.add_subplot(111)

    ax_VtxEnergy.scatter(df_modes[i]['E'], df_modes[i]['AbsVtxDiff.EA.{}'.format(C)], marker='.', s=1,
                         label='Elastic Arms')
    ax_VtxEnergy.scatter(df_modes[i]['E'], df_modes[i]['AbsVtxDiff.Model.{}'.format(C)], marker='.', s=1,
                         label='Model Pred.')

    plt.title('{} Interactions'.format(utils.plot.ModeType.name(i)))
    plt.xlabel('True Energy [GeV]')
    plt.ylabel('Abs. Vertex {} Difference [cm]'.format(C))
    plt.text(7.5, df_modes[i]['AbsVtxDiff.EA.{}'.format(C)].max() * 0.95,
             'Events {}\n {} Coordinate'.format(len(df_modes[i]), C), fontsize=12)
    plt.legend(loc='upper right')
    # plt.show()

    for ext in ['pdf', 'png']:
        fig_Enu.savefig(OUTDIR + '/plot_{}_{}_{}_abs_diff_EA_and_Model_vtx.{}_vs_E_{}.'.format(
                    DET, HORN, FLUX, C, utils.plot.ModeType.name(i)) + ext, dpi=300)


# ## Vertex vs. Energy plots (NOvA Energy)
# Plot the average "distance" between the E.A. Vertex and the True Vtx.
# make another set of plots that is within the energy range of NOvA 0-5 GeV...
for i in range(0, len(int_modes)):
    fig_nova_E = plt.figure(figsize=(7, 5))
    ax_NOvAVtxEnergy = fig_nova_E.add_subplot(111)

    ax_NOvAVtxEnergy.scatter(df_modes[i]['E NOvA'], df_modes[i]['AbsVtxDiff.EA.{}'.format(C)], s=1, marker='.',
                             label='Elastic Arms')
    ax_NOvAVtxEnergy.scatter(df_modes[i]['E NOvA'], df_modes[i]['AbsVtxDiff.Model.{}'.format(C)], s=1,
                             marker='.', label='Model Pred.')
    plt.title('{} Interactions'.format(utils.plot.ModeType.name(i)))
    plt.xlabel('Energy [GeV]')
    plt.ylabel('Abs. Vertex {} Difference [cm]'.format(C))
    plt.text(2.5, df_modes[i]['AbsVtxDiff.EA.{}'.format(C)].max() * 0.95, 'Events {}'.format(len(df_modes[i])),
             fontsize=8)
    plt.legend(loc='upper right')

    for ext in ['pdf', 'png']:
        fig_nova_E.savefig(
            OUTDIR + '/plot_{}_{}_{}_abs_diff_vtx.{}_vs_E_NOvA_{}.'.format(
                DET, HORN, FLUX, C, utils.plot.ModeType.name(i)) + ext, dpi=300)

print('Done.')

print(f'Plots created in: {OUTDIR}')