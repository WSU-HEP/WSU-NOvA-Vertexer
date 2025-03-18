#!/bin/bash
# shellcheck disable=SC2086

# bash script to create a slurm script for Vertexing ML on WSU BeoShock cluster.
# Oct. 2023, M. Dolce
# run this script by:
#   $ . create_slurm_script_training.sh <coordinate> <det> <horn> <flux> <epochs>

#confusion:  from Adam Tygart...don't request multiple nodes if you can't guarantee your code can handle them.
# -- but sbatch config fails with just 1 node.

DATE=$(date +%m-%d-%Y.%H.%M.%S)
echo "current date: " $DATE

# Ensure exactly 5 arguments are provided
if [[ $# -ne 5 ]]; then
    echo "Usage: $0 <COORDINATE> <DET> <HORN> <FLUX> <EPOCHS>"
    echo "Example: $0 xyz FD FHC Fluxswap 100"
    return 0
fi

COORDINATE=$1 # x, y, z, or xyz
DET=$2        # ND or FD
HORN=$3       # FHC or RHC
FLUX=$4       # Nonswap, Fluxswap, combined (both numu+nue)
EPOCHS=$5     # number of epochs to train for

# convert to lowercase, upper, and capitalize
COORDINATE=${COORDINATE,,}
DET=${DET^^}
HORN=${HORN^}
FLUX=${FLUX^}

echo "Coordinate: ${COORDINATE}"
echo "Detector: ${DET}"
echo "Horn: ${HORN}"
echo "Flux: ${FLUX}"
echo "Epochs: ${EPOCHS}"

# TODO: reorganize the trimmed h5 files to single location

outputfile=training_${COORDINATE}_${DET}_${HORN}_${FLUX}_${EPOCHS}Epochs_${DATE}

LOG_OUTDIR="/home/${USER}/output/logs/"

TRAINING_SCRIPT=${COORDINATE}_"vertex_training.py"

DATA_TRAIN_PATH="/home/k948d562/output/training/${DET}-Nominal-${HORN}-${FLUX}/"

slurm_dir="/home/${USER}/slurm-scripts/"
slurm_script="submit_slurm_${outputfile}.sh"

cat > $slurm_dir/submit_slurm_${outputfile}.sh <<EOS
#!/bin/bash

# script automatically generated at ${DATE} by create_slurm_script_training.sh.

## There is one strict rule for guaranteeing Slurm reads all of your options:
## Do not put *any* lines above your resource requests that aren't either:
##    1) blank. (no other characters)
##    2) comments (lines must begin with '#')

#Run this script by [ $ sbatch $slurm_script ]
#======================================================================================================================================
#SBATCH --job-name=${COORDINATE}_${DET}_${HORN}_${FLUX}_${EPOCHS}Epochs
#SBATCH --time=24:00:00

#SBATCH --output ${LOG_OUTDIR}/${outputfile}.out
#SBATCH --error  ${LOG_OUTDIR}/${outputfile}.err

### better for a single "task"
#SBATCH --ntasks=1         # Single task
#SBATCH --cpus-per-task=8  # Allocate 8 CPUs for better parallel processing
#SBATCH --mem=450000M      # RealMemory=514903M for gpu202401
#SBATCH --gres=gpu:2       # Request 2 GPUs

#SBATCH --nodelist=gpu202401  # compatible with TF 2.15.0

###SBATCH --mail-type ALL
###SBATCH --mail-user \${USER}\@wichita.edu
#======================================================================================================================================

echo "user is: \${USER}"

export WSUVTX="/homes/\$USER/WSU-NOvA-Vertexer"
echo "WSUVTX is: \${WSUVTX}"

# load modules
module load Python/3.11.5-GCCcore-13.2.0
source /homes/k948d562/virtual-envs/py3.11-pipTF2.15.0/bin/activate
/homes/k948d562/virtual-envs/py3.11-pipTF2.15.0/bin/python --version

echo "INFO: appending WSUVTX to PYTHONPATH"
unset PYTHONPATH
export PYTHONPATH="/homes/k948d562/virtual-envs/py3.11-pipTF2.15.0/lib/python3.11/site-packages:\${WSUVTX}"
echo "PYTHONPATH is ... \$PYTHONPATH"

export LD_LIBRARY_PATH="/homes/k948d562/virtual-envs/py3.11-pipTF2.15.0/lib:\$LD_LIBRARY_PATH"

echo "/homes/k948d562/virtual-envs/py3.11-pipTF2.15.0/bin/python \${WSUVTX}/Far-Detector/training/${TRAINING_SCRIPT} --data_train_path ${DATA_TRAIN_PATH} --epochs $EPOCHS"
#run python script
/homes/k948d562/virtual-envs/py3.11-pipTF2.15.0/bin/python \${WSUVTX}/Far-Detector/training/${TRAINING_SCRIPT} --data_train_path ${DATA_TRAIN_PATH} --epochs $EPOCHS


# After the job finishes, log resource usage
sleep 120
echo "Job completed. Logging resource usage:"                    >> ${LOG_OUTDIR}/${outputfile}.out
echo "sacct -j \$SLURM_JOB_ID --format=JobID,JobName,MaxRSS,MaxVMSize,NodeList,Elapsed,State "    >> ${LOG_OUTDIR}/${outputfile}.out
sacct -j \$SLURM_JOB_ID --format=JobID,JobName,MaxRSS,MaxVMSize,NodeList,Elapsed,State >> ${LOG_OUTDIR}/${outputfile}.out

EOS

echo "Created: $slurm_dir/$slurm_script "
echo "logs will be written to: $LOG_OUTDIR/${outputfile}"
echo "---------------------"