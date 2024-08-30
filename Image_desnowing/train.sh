#!/bin/bash

#SBATCH --job-name=desnow
#SBATCH --nodes=1
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

srun --partition=gpuq --nodes=1 --cpus-per-task=8 --ntasks=1 --gpus-per-task=1 /home/prrgpp000/.conda/envs/sde/bin/python main.py