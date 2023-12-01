#!/bin/bash

#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:00:00
#SBATCH --mem=100000MB
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_output/%j.out

source /home/${USER}/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env01

srun python llama_test.py