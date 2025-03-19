#!/bin/bash

export WSUVTX=/homes/$USER/WSU-NOvA-Vertexer/
echo WSUVTX is set: $WSUVTX
echo '=========================='

unset PYTHONPATH
export PYTHONPATH=/homes/k948d562/virtual-envs/py3.11-pipTF2.15.0/lib/python3.11/site-packages:$WSUVTX
echo PYTHONPATH is set: $PYTHONPATH
echo '=========================='

export LD_LIBRARY_PATH=/homes/k948d562/virtual-envs/py3.11-pipTF2.15.0/lib:$LD_LIBRARY_PATH
echo LD_LIBRARY_PATH is set: $LD_LIBRARY_PATH
echo 

echo "+ python envs established."
