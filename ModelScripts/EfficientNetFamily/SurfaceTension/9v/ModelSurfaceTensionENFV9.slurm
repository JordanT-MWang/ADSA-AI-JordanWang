#!/bin/bash
#SBATCH --job-name=SurfTensionEFTv9      # Job name
#SBATCH --partition=gpu             # Use GPU partition
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --time=0-12:30:00           # Max runtime (HH:MM:SS)
#SBATCH --cpus-per-task=4           # CPU cores for data loading

#SBATCH --mem=64G ## max amount of memory per node you require

#SBATCH --output=model-%A.out      # STDOUT log
#SBATCH --error=model-%A.err       # STDERR log

## Useful for remote notification
#SBATCH --mail-type=BEGIN,END,FAIL  # Email notifications
#SBATCH --mail-user=jordanw7@hawaii.edu  # Your email

## All options and environment variables found on schedMD site: http://slurm.schedmd.com/sbatch.html

module load lang/Anaconda3/2024.02-1
module load tools/git/2.42.0-GCCcore-13.2.0
module load system/CUDA/12.2.0
# Activate your environment the robust way
eval "$(conda shell.bash hook)"
conda activate ADSAenv

#copy data set locally
HOME_DATA="/home/jordanw7/koa_scratch/ADSA-AI/DataSetCombined"
LOCAL_DATA="$TMPDIR/DataSetCombined"
mkdir -p $LOCAL_DATA
cp -r $HOME_DATA/Edges $LOCAL_DATA/
cp $HOME_DATA/input_params.csv $LOCAL_DATA/
cp $HOME_DATA/output_params.csv $LOCAL_DATA/

eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
git stash
git pull --rebase origin main
git stash pop
python /home/jordanw7/koa_scratch/ADSA-AI/ADSA-AI-JordanWang/ModelScripts/EfficientNetFamily/SurfaceTension/9v/ModelSurfaceTensionENFV9.py --dataset_path $LOCAL_DATA
git add . ':!*.keras'
