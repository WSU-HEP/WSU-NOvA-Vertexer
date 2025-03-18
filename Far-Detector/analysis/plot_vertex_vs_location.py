#!/usr/bin/env python
# coding: utf-8
# # plot_vertex_location.py
# M. Dolce. 
# Feb. 2024
# 
# Makes plots to compare the true and E.A. reconstructed vertex (from a specified coordinate)
# as a function of the vertex location.
# NOTE: this makes resolution plots of ONE COORDINATE.
# 
# This validation is using `file_27_of_28.h5` as Ashley indicated.
# 
#  $ PY37 plot_vertex_vs_location.py --pred_file <path/to/CSV/predictions/file> -- outdir <path> --coordinate <i>

import utils.data_processing as dp
import utils.iomanager as io
import utils.plot

import argparse
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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
print(f'Making plots for coordinate: {C}.....')

print(f'pred_file: {args.pred_file}')

# filename without the CSV and path.
pred_filename_prefix = args.pred_file.split('/')[-1].split('.')[0]  # get the filename only and remove the .csv
DET, HORN, FLUX = io.IOManager.get_det_horn_and_flux_from_string(args.pred_file)

# Load the CSV file of the model predictions.
df = dp.ModelPrediction.load_pred_csv_file(args.pred_file)


# define the path to the validation files
test_file = args.test_file.format(DET, HORN, FLUX, DET, HORN, FLUX)
print('test_file: {}'.format(test_file))


# Create the DataFrame directly from the h5 file
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

# Add the following columns: absolute vertex difference between...
# -- abs(E.A. - True) vertex
# -- abs(Model - True) vertex.
vtx_abs_diff_ea_temp, vtx_deff_temp_model = dp.ModelPrediction.create_abs_vtx_diff_columns(df, C)

# Add in the Model Prediction.
df = pd.concat([df, df_test_file, vtx_abs_diff_ea_temp, vtx_deff_temp_model], axis=1)

# define path to save some plots
OUTDIR = utils.plot.make_output_dir(args.outdir, 'vertex-vs-location', pred_filename_prefix)

# create a column of the Energy within NOvA's range/relevance -- for convenience.
E_nova = df[df['E'] < 5]
df['E NOvA'] = E_nova['E']
df.describe()

# assert are no kUnknownMode events...
assert (len(df[df['Mode'] == -1]) == 0)  # 'there are Unknown events. Stopping'

# list of the dataframes for each mode has both NC and CC.
df_modes = list()
for i in range(0, len(int_modes)):
    df_modes.append(df[df['Mode'] == i])


# Vertex Location plots
# Plot the Reco vs. True vertex for the nue events broken down by interaction type
print('PLOTTING THE TRUE VS. E.A. & Model VERTEX Location FOR EACH INTERACTION TYPE......')
for i in range(0, len(int_modes)):
    int_name = utils.plot.ModeType.name(i)
    print('creating plots for......', int_name)

    fig_reco_true = plt.figure(figsize=(10, 10))
    ax_VtxLocation = fig_reco_true.add_subplot(111)
    ax_VtxLocation.scatter(df_modes[i]['True {}'.format(C)][:],
                           df_modes[i]['Reco {}'.format(C)][:],
                           s=1,
                           label='Elastic Arms')
    ax_VtxLocation.scatter(df_modes[i]['True {}'.format(C)][:],
                           df_modes[i]['Model Pred {}'.format(C)][:],
                           s=1,
                           marker='x',
                           color='orange',
                           label='Model Pred.')
    plt.title('{} Vertex Location'.format(int_name))
    plt.xlabel('True Vtx {} [cm]'.format(C))
    plt.ylabel('Reco. Vtx {} [cm]'.format(C))
    plt.plot(np.arange(-750, 750, 1), np.arange(-750, 750, 1), '--', lw=1.5, color='gray')
    plt.legend(loc='upper left')
    plt.text(300, -500, '{} {} {} \n {} Vertex\n Validation'.format(DET, HORN, FLUX, C), fontsize=12)

    for ext in ['pdf', 'png']:
        fig_reco_true.savefig(
            OUTDIR + '/{}_{}_{}_EA_and_Model_vs_true_vtx_{}.{}.'.format(
                DET, HORN, FLUX, C, int_name) + ext, dpi=300)

print('Done.')
