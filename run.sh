#!/bin/bash
#SBATCH -p oahu                      # Specify the partition
#SBATCH --nodelist=blade2           # Specify the node
#SBATCH --ntasks=24                 # Number of tasks
#SBATCH --mem=100gb                 # Memory allocation
#SBATCH -t 148:00:00  # time (HH:MM:SS)
#SBATCH --output=log_b_max_mean.txt  # Output log file

source ~/miniconda3/etc/profile.d/conda.sh
conda activate my_env
# python main.py synthetic
# python main.py damadics
# python main.py benchmark hpt
# python main.py benchmark default
python main.py benchmark max_mean
 

