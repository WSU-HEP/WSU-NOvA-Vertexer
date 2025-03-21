#!/bin/bash

# To run this script:
#   source setup_env.sh

export WSUVTX=/homes/$USER/WSU-NOvA-Vertexer/
echo WSUVTX is set: "$WSUVTX"
echo '=========================='

# load the module from the cluster
module load Python/3.11.5-GCCcore-13.2.0 && echo "Success. Python 3.11.5 module loaded." || echo "Failed."
echo '=========================='

# activate the virtual env
source /homes/k948d562/virtual-envs/py3.11-pipTF2.15.0/bin/activate && echo "Success. py3.11-pipTF2 venv sourced." || echo "Failed."
unset PYTHONPATH
export PYTHONPATH=/homes/k948d562/virtual-envs/py3.11-pipTF2.15.0/lib/python3.11/site-packages:$WSUVTX
echo PYTHONPATH is set: "$PYTHONPATH"
echo '=========================='

export LD_LIBRARY_PATH=/homes/k948d562/virtual-envs/py3.11-pipTF2.15.0/lib:$LD_LIBRARY_PATH
echo LD_LIBRARY_PATH is set: "$LD_LIBRARY_PATH"
echo 

echo "+ WSU Vertexing envs established."
