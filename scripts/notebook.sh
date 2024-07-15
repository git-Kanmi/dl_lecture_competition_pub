#!/bin/bash
#SBATCH --job-name=notebook
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
conda activate thingsvision
# for pyenv/virtualenv instead use
# source /home/username/project/project-venv/bin/activate

cd /gpfs02/work/kanmi.nose/DLBasics2023_colab/dl_lecture_competition_pub/

export HYDRA_FULL_ERROR=1

jupyter lab --no-browser --ip=0.0.0.0 --port=4869
~

