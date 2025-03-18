#!/bin/bash

export WSUVTX=/homes/$USER/WSU-NOvA-Vertexer/
echo WSUVTX is set: $WSUVTX
echo '=========================='

unset PYTHONPATH
export PYTHONPATH=/homes/k948d562/virtual-envs/poetry1.7.0-test-py3.11-tf2.15/lib/python3.11/site-packages:$WSUVTX
echo PYTHONPATH is set: $PYTHONPATH
echo '=========================='

export LD_LIBRARY_PATH=/homes/k948d562/virtual-envs/poetry1.7.0-test-py3.11-tf2.15/lib:$LD_LIBRARY_PATH
echo LD_LIBRARY_PATH is set: $LD_LIBRARY_PATH
echo 

echo "+ python envs established."
