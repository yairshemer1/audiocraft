#!/bin/sh

#SBATCH --time=7-0
#SBATCH --killable
#SBATCH --requeue
#SBATCH --gres=gpu:2,vmem:16g
#SBATCH --mem=32g
#SBATCH -c4
#SBATCH -o /cs/usr/yair.shemer/AudioLab/audiocraft/slurm_logs/complex_first_try.out

source /cs/labs/adiyoss/yair.shemer/venv/encodec/bin/activate;
dora run -d solver=compression/reconstruct label=complex_first_try wandb.name=complex_first_try
