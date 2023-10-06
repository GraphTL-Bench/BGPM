#!/bin/bash -l
#SBATCH --job-name=para_tuning
#SBATCH --output=%x.%j.out # %x.%j expands to JobName.JobID
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --partition=lowpriority
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --exclude=node650
#SBATCH --time=2-23:00:00
#SBATCH --constraint=avx512

# Purge any module loaded by default
module purge > /dev/null 2>&1
conda activate pygcl

## cora
#srun python3 run_model.py --task GCL --model BGRL --dataset Cora --config_file config1
srun python3 run_model.py --task GCL --model GBT --dataset Cora --config_file config1
