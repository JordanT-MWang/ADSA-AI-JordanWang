#!/bin/bash
#SBATCH --job-name=DogCatCNN        # Job name
#SBATCH --partition=gpu             # Use GPU partition
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --time=0-00:30:00           # Max runtime (HH:MM:SS)
#SBATCH --cpus-per-task=2           # CPU cores for data loading

#SBATCH --mem=32G ## max amount of memory per node you require

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


eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
git pull
python ModelSurfaceTension.py
git add -A
git commit -am "Predictions saved for job $SLURM_JOB_ID"
git push origin