#!/bin/bash
# shellcheck disable=SC2086

# bash script to create the pre-processing slurm script for Vertexing ML on WSU BeoShock cluster.
# this script will take one h5 file to preprocess...
# NOTE: this preprocessing saves the X,Y, and Z vertex information into single file!
# Oct. 2023, M. Dolce

# run this script by:
#   $ . create_slurm_script_preprocess.sh  <det> <horn> <flux> <file_number>

#confusion:  from Adam Tygart...don't request multiple nodes if you can't guarantee your code can handle them.
# -- but sbatch config fails with just 1 node.

DATE=$(date +%m-%d-%Y.%H.%M.%S)
echo "current date: " $DATE

DET=$1         # ND or FD
HORN=$2        # FHC or RHC
FLUX=$3        # Nonswap, Fluxswap
FILE_NUMBER=$4 # file number to use (options: 0,3,6,9,12,15,18,21)

echo "Detector: $DET"
echo "Horn: $HORN"
echo "Flux: $FLUX"
echo "File number: $FILE_NUMBER"

# The file to be processed. Much of this is hardcoded, so (path/filename) cannot be tampered with!
PREPROCESS_FILE_PATH=/home/k948d562/NOvA-shared/$DET-Training-Samples/$DET-Nominal-$HORN-$FLUX/train/trimmed_h5_R20-11-25-prod5.1reco.j_$DET-Nominal-$HORN-${FLUX}_${FILE_NUMBER}_of_28.h5
echo "Preprocessing file to be created from script: $PREPROCESS_FILE_PATH"
PREPROCESS_FILE=trimmed_h5_R20-11-25-prod5.1reco.j_$DET-Nominal-$HORN-${FLUX}_${FILE_NUMBER}_of_28.h5

outputfile=preprocess_${DET}_${HORN}_${FLUX}_file_${FILE_NUMBER}_date_${DATE}

# the log files go into logs dir
OUTDIR_PREFIX=/home/k948d562/output/wsu-vertexer/preprocess

slurm_dir="/home/k948d562/slurm-scripts/"
slurm_script="submit_slurm_${outputfile}.sh"

cat > ${slurm_dir}/${slurm_script} <<EOF
#!/bin/bash

# script automatically generated at ${DATE} by create_slurm_script_preprocess.sh.

## There is one strict rule for guaranteeing Slurm reads all of your options:
## Do not put *any* lines above your resource requests that aren't either:
##    1) blank. (no other characters)
##    2) comments (lines must begin with '#')

#Run this script by [ $ sbatch $slurm_script ]
#========================================================================================================
#SBATCH --job-name=preprocess_${PREPROCESS_FILE}

# 24 hours seems to be the edge, so add more...
#SBATCH --time=48:00:00

#SBATCH --output ${OUTDIR_PREFIX}/logs/${PREPROCESS_FILE}.out
#SBATCH --error  ${OUTDIR_PREFIX}/logs/${PREPROCESS_FILE}.err

#SBATCH --nodes=2         # 2 if you can't guarantee your code can handle them. your current code cannot.
#SBATCH --mem-per-cpu=40G # memory per CPU core
#========================================================================================================


# load modules
module load Python/3.7.4-GCCcore-8.3.0
module load TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4
source /home/k948d562/virtual-envs/VirtualTensorFlow-Abdul/VirtualTensor/bin/activate
/home/k948d562/virtual-envs/VirtualTensorFlow-Abdul/VirtualTensor/bin/python --version


#run python script
/home/k948d562/virtual-envs/VirtualTensorFlow-Abdul/VirtualTensor/bin/python /home/k948d562/ml-vertexing/wsu-vertexer/preprocess/preprocess_h5_file.py  $PREPROCESS_FILE_PATH
EOF

echo "Slurm script created: ${slurm_dir}/${slurm_script}"
