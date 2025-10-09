#!/bin/bash
#SBATCH --job-name=DogCatCNN        # Job name
#SBATCH --partition=gpu             # Use GPU partition
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --time=0-02:00:00           # Max runtime (HH:MM:SS)
#SBATCH --cpus-per-task=2           # CPU cores for data loading

#SBATCH --mem=32G ## max amount of memory per node you require

#SBATCH --output=dogcat-%A.out      # STDOUT log
#SBATCH --error=dogcat-%A.err       # STDERR log

## Useful for remote notification
#SBATCH --mail-type=BEGIN,END,FAIL  # Email notifications
#SBATCH --mail-user=jordanw7@hawaii.edu  # Your email

## All options and environment variables found on schedMD site: http://slurm.schedmd.com/sbatch.html

module load lang/Anaconda3/2024.02-1
source activate "ADSAenv"
module load tools/git/2.42.0-GCCcore-13.2.0

python dogcatcnn.py
git add results/*
git commit -m "Predictions saved for job $SLURM_JOB_ID"
git push origin