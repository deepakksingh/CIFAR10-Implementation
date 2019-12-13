#!/bin/bash
#SBATCH -A dksingh
#SBATCH --reservation=non-deadline-queue
#SBATCH -n 40
#SBATCH --partition=long
#SBATCH --mem-per-cpu=1024
##SBATCH --nodelist=gnode02
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
##SBATCH -C 2080ti


#load necessary modules
module load python/3.6.8 
module load cuda/10.0 
module load cudnn/7-cuda-10.0

#activate anaconda environment
source activate dev
echo "dks conda environment activated"

python main_2.py results_on_1080ti
