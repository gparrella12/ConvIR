#!/bin/bash

#SBATCH --job-name=convir
#SBATCH --nodes=1
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

srun --partition=gpuq --nodes=1 --cpus-per-task=8 --ntasks=1 --gpus-per-task=1 python main.py --data_dir /home/prrgpp000/cpa_enhanced/datasets/reconstructions