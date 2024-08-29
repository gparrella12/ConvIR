#!/bin/bash

#SBATCH --job-name=convir-derain
#SBATCH --nodes=1
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

srun --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=8 python main.py