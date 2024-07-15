#!/bin/bash
#SBATCH --job-name=MEG
#SBATCH --account=kanmi.nose
#SBATCH --output=/home/kanmi.nose/log/%j.out  
#SBATCH --error=/home/kanmi.nose/log/%j.err
#SBATCH --time=24:00:00  
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-gpu=6


export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0 # where X is the GPU id of an available GPU

# activate python environment
conda activate dlbasics
# for pyenv/virtualenv instead use
# source /home/username/project/project-venv/bin/activate

cd /gpfs02/work/kanmi.nose/DLBasics2023_colab/dl_lecture_competition_pub/

wandb login 86bdbf7a86b77d8d3fde2881e011e068b751e748

export HYDRA_FULL_ERROR=1

python eval.py use_wandb=False
