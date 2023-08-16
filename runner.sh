#!/bin/sh

#SBATCH --time=7-0
#SBATCH --killable
#SBATCH --requeue
#SBATCH --gres=gpu:2,vmem:16g
#SBATCH --mem=32g
#SBATCH -c4
#SBATCH -o /cs/usr/yair.shemer/AudioLab/audiocraft/slurm_logs/baseline_distrib.out

source /cs/labs/adiyoss/yair.shemer/venv/encodec/bin/activate;
dora run -d solver=compression/reconstruct label=baseline_distrib wandb.name=baseline_distrib
