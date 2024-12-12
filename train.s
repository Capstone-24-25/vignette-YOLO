#!/bin/bash
#SBATCH -J "trainYOLOx"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH -gres=gpu:1
#SBATCH -o outLog
#SBATCH -e errLog
#SBATCH --mail-user=josephzaki@ucsb.edu
#SBATCH --mail-type ALL

module purge all
module load cuda/11.6
source source /sw/csc/anaconda/anaconda3/bin/activate
conda activate ultralytics-env

cd $SLURM_SUBMIT_DIR
python train11x.py