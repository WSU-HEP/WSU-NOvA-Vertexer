# Far Detector Prod5.1 Vertexing
### Instructions and workflow of the FD Prod5.1 Vertexing.

--- 

_Original Author_: Michael Dolce -- mdolce@fnal.gov

Updated: **Feb. 2025**

--- 

## Workflow
0. [Pre-processing](#pre-processing)
1. [Inital-checks](#initial-checks)
2. [Training](#training)
3. [Prediction](#prediction)
4. [Analysis](#analysis)


### NOTE: changing WSUID paths
The paths in the scripts are set to my WSUID.
You **MUST** change these paths to your WSUID -- `$USER` on BeoShock -- or you will not be able to run the scripts.
Check where they are by doing:
```grep -i "k948d562" * -r```.

**Second NOTE:**
Much of the frequently-used code has been moved to `utils`.

## 0. Pre-processing
As it stands, the code is set up to run on the WSU cluster.
The objective is to slim the file sizeby reducing the other unnecessary NOvA variables. 
The only variables we require are:

- `cvnmap` 
- `vtx.{x,y,z}`
- `firstcell{x,y}`
- `firstplane`

The `cvnmap` and `vtx.{x,y,z}` values are explicitly required for training.
The pixelmaps (`cvnmap`) are our features and the true vertex location is our label.

The `firtcell{x,y}` and `firstplane` are used for the first hits of energy recorded in the detectors. This allows us to translate back and forth from detector space to pixel map space.  

To start, we run a bash script to create the slurm configuration for a WSU Beoshock cluster job(s). In an SSH session on the BeoShock cluster, we use this incantation:
```
. create_slurm_script_preprocess.sh  <det> <horn> <flux> <file_number>
```
where `det` is "FD" for Far Detector, `horn` is the magnetic horn current `FHC` or `RHC` (for "Forward Horn Current" and "Reverse Horn Current"), `flux` is either `nonswap` or `fluxswap` (for muon neutrinos or electron neutrinos, respectively), and `file_number` is the specific file number you want to pre-process (the h5 training files are numbered).


This script will set up the input files and output files as well as other configuration setting for a slurm job. **Be sure to change the paths of these files to your WSUID, and replace the existing one, otherwise you will not be able to proceed** (Beoshock does not allow  access to other users directories).

This script will immediately create a `submit_slurm_<some-descriptions>.sh` script. Now you can submit pre-processing to the WSU grid...

```
sbatch submit_slurm_<some-descriptions>.sh
```
 
This pre-processing can take up to 24 hours (and maybe even more). The expectation (from Oct. 2023) is that the Prod5.1 h5 files from the #reco-conveners should be slimmed from ~800MB --> ~200MB. 

For those that are curious, the `submit_slurm_<some-descriptions>.sh` script is performing this task:

```
/home/k948d562/virtual-envs/VirtualTensorFlow-Abdul/VirtualTensor/bin/python /home/k948d562/ml-vertexing/wsu-vertexer/preprocess/preprocess_h5_file.py  $PREPROCESS_FILE_PATH
```

With the completed files, training can be performed. 

But first, we want to run some initial checks on the files to make sure they are good to go.
The next section...


---

## 1. Initial-checks

At this stage, we want to make sure the information within the pre-processed files are correct and ready to be used.

Because the labels fed into our network are locations -- in detector coordinates -- and the pixel maps (used for the training),
-- the features -- are used to train the network, we need to convert the vertex location into the pixel map space. 
This step confirms everything looks OK.

To run the initial checks, we use two notebooks:
* `validate_preprocessed_h5.ipynb`
* `plot_cvnmaps.ipynb`

The first notebook is very simple, it just checks the size of the `cvnmap` (the pixel maps) and plots the first 20 vertex locations for the X coordinate.

The second notebook is more thorough. It does the following:
1. reads in the pre-processed h5 file as `numpy.ndarray`
2. checks the type and shape of the data (cvnmaps, vtx locations, and cells/plane)
3. creates a new array to convert each vertex coordinate into a pixel map coordinate (it also loads the array in a specific way because the original h5s were saved as `unsigned int` and we need to convert them to `int`).
4. lastly, it makes plots of:
   1. random pixel maps with the true vertex overlaid in yellow 'x'
   2. all the true detector vertex locations.
   3. all the true vertex locations in the 3D pixel map space

Especially for plot **1.**, it is important the vertex location is sensible, otherwise the training will produce nonsense.

Here is an example:

![example of a FD interaction of the pixel map wih the true vertex location overlaid.](https://github.com/mdolce8/WSU-NOvA-Vertexer/blob/main/Prod5.1-FD/initial-checks/cvnmap_example.png?raw=true "CVN Map example")


---


## 2. Training

Next we train a model for **all coordinates simultaneously**.
We have come to call this the "branch model", others in NOvA have called it the "Siamese" model.

Again, we perform the training on the WSU cluster using slurm.
We use `training/create_slurm_script_training.sh` to create the slurm script for training.
Its arguments are:
```
COORDINATE=$1 # xyz
DET=$2        # ND or FD
HORN=$3       # FHC or RHC
FLUX=$4       # Nonswap, Fluxswap
EPOCHS=$5     # number of epochs
```
 which are identical arguments to the pre-processing script. We run the script like this: 

``` . create_slurm_script_training.sh xyz <det> <horn> <flux> <epochs>```

Before running, we must address the file path: 
* **the user must manually copy the `preprocessed_*.h5` files to the `training` directory.**
For example for the FD FHC Fluxswap sample, the preprocessed files must be copied here,
```/home/{WSUID}/output/wsu-vertexer/training/FD-Nominal-FHC-Fluxswap/```
as this is path that the training script will look for the files.
(Note the user will need to change the path to **their** WSUID).

Once this is addressed, we can submit the slurm script that is created from the bash script:

```sbatch submit_slurm_training_${EPOCHS}_${DET}_${HORN}_${FLUX}_${COORDINATE}_${DATE}.sh```

For those curious, what is _really_ running is `xyz_vertex_training.py` ,
where the arguments from the bash script are used into the training script.
This script does many important things:
* loads the pre-processed h5 files,
* correctly formats the data,
* creates the model,
* trains the model,
* and saves the model to an h5 file,
* saves the metrics (the history) to a CSV file.

---

### Resources for training: 

We produce three outputs: X, Y, and Z. The model uses both pixel map images
to "learn" about the vertex location for each coordinate. 
In principle, we are maximizing the information of each coordinate by giving the model the two views for every event.

In our slurm script for submitting the training, we also request available CPUs for better parallel processing. 
However, the bulk of the training is done with one of the two GPU nodes we have available. 
Here are details of their memory allocations: `scontrol show node gpu20??01 | grep RealMemory`:
* gpu20901
  * `RealMemory=384896M`
* gpu202401
  * `RealMemory=514903M`

NOTE: the software stack must be newer than the hardware.

Because we are using significantly more memory loading **both** sets of pixel map images for training,
we use a sizable amount of memory. 
Fortunately our requests (as of now) fit within a single node.
We now use the `gpu202401` node for trainings, as our software stack is compatible with this newer A30 Nvidia hardware.


Our most recent training was done with this node, and a few summary statistics are printed below via `sacct`:

```
sacct -j 928018 --format=JobID,JobName,MaxRSS,MaxVMSize,NodeList,Elapsed,State 
JobID           JobName     MaxRSS  MaxVMSize        NodeList    Elapsed      State 
------------ ---------- ---------- ---------- --------------- ---------- ---------- 
928018       xyz_FD_FH+                             gpu202401   08:53:37  COMPLETED 
928018.batch      batch 169961804K          0       gpu202401   08:53:37  COMPLETED 
928018.exte+     extern          0          0       gpu202401   08:53:37  COMPLETED 
```

We use about ~170 GB of memory usage -- **NOTE** the features (the cvnmaps are `unit8`, currently, reducing our memory usage).

For the current version of TensorFlow & Python (2.15.0 and 3.11.5), user **should** request resources that are in line with this amount of memory. 

**It is important to note** this training reported above is only for 4 files of FD Fluxswap -- adding more files will of course increase memory usage.

---

To check on the status of the job, we can do: ```squeue -u $USER``` or `kstat` to see the status of the job.
Note, that there are `.log` and `.err` files that get made (check the bash script for the exact name) that can be used to debug the job too (very helpful).
The job _should_ take about 4 hours to complete, using the current configuration. 

Once completed, the `.log` file contains useful printout information about the network, loss, and time to train, in addition,
to the names and directories of the saved models, and metrics file.
So, it is important to keep track of the `.log` file for this info.

In addition, once the job completes, it is **highly beneficial** the user use `sacct -j <jobnumber>` to note important metrics from the job, such as:
memory usage, total time, CPUs used, and more. 
```
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,MaxVMSize,NodeList 
```
For more information to report do `sacct --help` to get a list of other quantities to report.

** NOTE: We note that we use a "small scale" python script to train a small subset of the train data to ensure the
plumbing is working properly. This can also be helpful for debugging. It is NOT for production use, but for testing infrastructure.
This script is `xyz_vertex_training_testsize.py`.** 


--- 

# 3. Prediction

Next, we want to use the model file to make predictions on the designated inference data set.

NOTE: 
* the inference data set is different from the training data set.
* In the FD Prod5.1 {FHC,RHC} {nonswap,fluxswap} samples, the inference data set is file 27.

Here is an example of generating the prediction on the designated "file 27" (this script automatically should read in file 27):

```
$PY37 model_predict_coordinates_xyz.py \
--model_file $OUTPUT/trained-models/model_29epochs_FD_FHC_Fluxswap_2024-12-18_XYZ.h5 \
--outdir $OUTPUT/predictions 
```

where `--model_file` is the full path to the h5 file produced from the training.
This will generate a prediction of the coordinate that's name is included in the `model_file` name.
This takes about 5 minutes to run.

---

These predictions will be saved as a CSV file that contain the following information:
```['True X', 'Reco X', 'Model Prediction X', .... Y...., ..... Z .... ]``` ordered and labeled in this fashion.

At this stage, the use can now make numerous comparisons between the model and the standard NOvA "Elastic Arms" algorithm!

Have fun!

--- 

# 4. Analysis

Now we want to analyze the results of the model predictions.

Again, we emphasize that for this particular NOvA production, Prod5.1 for FD, we designate file "27" (and file "24" depending on the case) for the evaluation of the model.


i. `plot_cvnmap_predictions.ipynb`
This macro will plot the pixel map with the true vertex location _and_ the model prediction. This is intended as a qualitative check.
(you can specify a specific event if desired)

ii. `fd_p5p1_nu_specrta.ipynb` This macro plots simple Enu distribution, and also broken down into interaction type (probably better considered an "initial check").

iii. `plot_1d_vertex_resolutions.py` script to plot (model - truth) resolutions, 
as well as %-difference, absolute difference, and also to make the plots broken down by interaction type. Highly useful.

iv. `plot_2d_vertex_resolutions.py` script to plot the 2D resolution for a selected pair of coordinates. 
Better for qualitative interpretation.

v. `plot_vertex_vs_location.py` a helpful script that plots the location of the vertex for both the Elastic Arms and Model 
predictions. A perfect fit would place all points on the diagonal line.

vi. `plot_vertex_vs_energy.py` script to plot Elastic Arms and Model Pred vertices together as a function of neutrino energy,
also a qualitative plot.
