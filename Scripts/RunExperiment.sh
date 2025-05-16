#!/bin/bash

export USER=vichare2
export SCRIPT_START_TIME=$(date +%s)
export HOME=$(pwd)

echo "Starting job for $USER at $SCRIPT_START_TIME"

# Job Identifier for logging - expected to be passed as the first argument else set as year-month-date_hour-minute-second
if [ -z "$1" ]; then
    echo "No job identifier provided. Using current timestamp as identifier."
    export RUN_ID=$(date +%y%m%d_%Hh%Mm)
else
    export RUN_ID=$1
fi

# Environment variables setup
export ENVNAME=vmamba
export ENVDIR=$HOME/$ENVNAME\_env
export PATH=$ENVDIR/bin:$PATH
export OUTPUT_PATH=$HOME/results
export STAGING_PATH=$HOME/staging

ln -s /staging/$USER $STAGING_PATH

# Create the directories
mkdir $OUTPUT_PATH
mkdir $ENVDIR

# Set up the environment
(
    tar -xzf $STAGING_PATH/envs/$ENVNAME.tar.gz -C $ENVDIR
    # tar -xzf $ENVNAME.tar.gz -C $ENVDIR
    source $ENVDIR/bin/activate
    $ENVDIR/bin/conda-unpack
    # conda not in path but packages are unpacked
    echo "Conda environment activated"
) &

# Copy the dataset
(
    tar -xzf $STAGING_PATH/datasets/BraTS19_nnUNet_format.tar.gz -C data/
    # tar -xzf $STAGING_PATH/datasets/UCSF-PDGM-nnUNet_format.tar.gz -C data/
    # tar -xf $STAGING_PATH/datasets/Dataset301_Combined.tar -C data/
    # unzip -q $STAGING_PATH/datasets/brats-2019.zip -d .
    echo "Dataset copied"
) &

# Wait for the environment and dataset to be copied
wait


# Validate the machine environment
echo "Home: $HOME"
echo "Running job on `hostname`"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"
echo "Hello CHTC from Job $1 running on `whoami`@`hostname`"
echo "Architecture: $(uname -m)"
echo "Date: $(date)"
nvidia-smi
cat /etc/os-release

# Addditional setup of the environment
echo "Pip location: $(which pip)"
echo "Python location: $(which python)"
pip install --upgrade pip # Upgrade pip for editable installations
pip --version
pip install -e uumamba

# Additional scripts
bash $HOME/Scripts/runJupyter.sh # Remote JupyterLab
bash $HOME/Scripts/runBore.sh --port 8888 --remote-port 8889 # Remote Bore tunnel
# bash $HOME/Scripts/installNode.sh # Remote Node.js
# source $HOME/.bashrc
# bash $HOME/Scripts/runLocaltunnel.sh --port 8888 --subdomain chtc-jupyter-$USER # Remote Localtunnel


# Run the job
echo "Script execution started at $(date)"

# Restore results directory if continuing training
# tar -xzf $STAGING_PATH/results_backup.tar.gz -C $OUTPUT_PATH
# echo "Results directory restored to `$OUTPUT_PATH`:"
# du -h $OUTPUT_PATH
# echo ""

# Copy converted dataset - Already done
# python uumamba/nnunetv2/dataset_conversion/Dataset111_BraTS19.py -i MICCAI_BraTS_2019_Data_Training -o data/nnUNet_raw/Dataset111_BraTS19
# python uumamba/nnunetv2/dataset_conversion/Dataset112_UCSF_PGDM.py -i data/UCSF-PDGM-Filtered -o data/nnUNet_raw/Dataset112_UCSF_PGDM

# Preprocess the dataset
nnUNetv2_plan_and_preprocess -d 111 -c 3d_fullres --verify_dataset_integrity
# nnUNetv2_train 301 3d_fullres 4 -tr nnUNetTrainerUMambaEnc --c

# Run training with output logged to a file and errors to a separate file
# nnUNetv2_train 310 3d_fullres 4 -tr nnUNetTrainerUMambaEnc --c > logs/$(date +%s).log 2> logs/$(date +%s).err

# If training is completed, move results from results/running to results/RUN_ID
# python finalize_results.py

# DATASET_PATH=MICCAI_BraTS_2019_Data_Training python run.py

# Reference Commands from the README
<<'###'
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -q /staging/vichare2/datasets/ACDC.zip -d data
python data/Dataset027_ACDC.py -i data/ACDC/database -o data/nnUNet_raw/Dataset027_ACDC/
nnUNetv2_plan_and_preprocess -d 027 --verify_dataset_integrity
nnUNetv2_train 027 3d_fullres all -tr nnUNetTrainerUMambaEnc -pretrained_weights pretrain_weight/checkpoint_UU-Mamba.pth
nnUNetv2_train 027 3d_fullres 4 -tr nnUNetTrainerUMambaEnc -pretrained_weights pretrain_weight/checkpoint_UU-Mamba.pth --val
###


# Wait for manual control to end
bash Scripts/waitForManualControl.sh

echo "Script execution ended at $(date)"


# Past command references
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/miniconda.sh
# bash $HOME/miniconda.sh -b -p $HOME/miniconda
# echo $SHELL
# eval "$(/$HOME/miniconda/bin/conda shell.bash hook)"
# conda init
# conda env list

# conda env create -f environment.yml
# conda env list
# conda pack -n vmamba --dest-prefix='$ENVDIR'
# chmod 644 vmamba.tar.gz
# ls -sh vmamba.tar.gz

# echo conda: $(which conda)
# echo python: $(which python)
# echo pip: $(which pip)
# echo nvcc: $(which nvcc)
# nvcc --version
# echo gcc: $(which gcc)
# gcc --version
# echo g++: $(which g++)
# g++ --version


