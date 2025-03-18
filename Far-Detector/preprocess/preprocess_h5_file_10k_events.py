# preprocess h5 file for "test" training: keep only 10k events
# so to run training on an interactive node, ideally.

# which cuts out everything but:
#   -- the vertex
#   -- first cell and plane,
#   -- cvnmap.
# (vtx.x, vtx.y, vtx.z, firstcellx, firstcelly, firstplane, cvnmap).

# this script is submitted to the BeoShock (WSU) cluster
# and done interactively. small scale

# M. Dolce
# Oct. 2023

# To run this script:  $PY37 preprocess_h5_file.py <infile_h5>



import os
import sys
import h5py
import numpy as np

if __name__ == '__main__':
    # Terminal Arguments: input file [1]
    inFilePath = sys.argv[1]
    inPath = inFilePath.split('/')[:-1]
    infile = inFilePath.split('/')[-1]
    # TODO: make the number of events you want to keep an argument
    # print('input path: ', inPath)

    # the number of events we will save to h5 file for "testing" training
    event_limit = 10_000
    event_string = str(int(event_limit/1000)) + 'k_events'

    # this outPath is hard-coded, it's in "my" directory
    outPath = '/home/k948d562/output/wsu-vertexer/preprocess'
    outName = 'preprocessed_{}_{}'.format(event_string, infile)
    print('Processing h5 file: ' + infile)
    print('Saving for training to ' + outPath)

    # Don't recreate the file if it exists
    print('Creating file...{}'.format(os.path.join(outPath, outName)))
    if os.path.exists(os.path.join(outPath, outName)):
        print('File already exists. Don\'t want to overwrite! Exiting...')
        exit(0)

    # Load the h5 file
    # One file at a time to avoid problems with loading a bunch of pixel maps in memory
    print('Opening file.....{}'.format(infile))
    print('Extracting first {} events'.format(event_limit))
    df_x = h5py.File(inFilePath, 'r')['vtx.x'][0:event_limit]
    df_y = h5py.File(inFilePath, 'r')['vtx.y'][0:event_limit]
    df_z = h5py.File(inFilePath, 'r')['vtx.z'][0:event_limit]
    df_cvnmap = h5py.File(inFilePath, 'r')['cvnmap'][0:event_limit]
    df_firstcellx = h5py.File(inFilePath, 'r')['firstcellx'][0:event_limit]
    df_firstcelly = h5py.File(inFilePath, 'r')['firstcelly'][0:event_limit]
    df_firstplane = h5py.File(inFilePath, 'r')['firstplane'][0:event_limit]
    print('loaded the vertices, first cells, first plane and cvnmap dfs. {} events only!'.format(event_limit))

    # Save in an h5 with new dataset keys
    hf = h5py.File(os.path.join(outPath, outName), 'w')

    hf.create_dataset('vtx.x',  data=df_x,                compression='gzip')
    print('added vtx.x')
    hf.create_dataset('vtx.y',  data=df_y,                compression='gzip')
    print('added vtx.y')
    hf.create_dataset('vtx.z',  data=df_z,                compression='gzip')
    print('added vtx.z')
    hf.create_dataset('firstcellx', data=df_firstcellx,   compression='gzip')
    print('added firstcellx')
    hf.create_dataset('firstcelly', data=df_firstcelly,   compression='gzip')
    print('added firstcelly')
    hf.create_dataset('firstplane', data=df_firstplane,   compression='gzip')
    print('added firstplane')

    # save as 'chunks' to save space, since each pixel map is this size.
    hf.create_dataset('cvnmap', data=np.stack(df_cvnmap), chunks=(1, 16000),  compression='gzip')
    print('added cvnmap')

    hf.close()
    print('Output saved to: ', outPath)
    print('File created: ', outName)
