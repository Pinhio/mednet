#!/bin/bash

#SBATCH --time=14:00:00
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --job-name=mednet
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:1
#SBATCH --output=/home/sc.uni-leipzig.de/jn137pgao/jobfiles/log/%x-%j.out
#SBATCH --error=/home/sc.uni-leipzig.de/jn137pgao/jobfiles/err/%x-%j.err
#SBATCH --mail-type=END


module load PyTorch
python train.py --gpu_id 0 --image_clip_type 't2_sag_black' --model_depth 200 --n_epochs 200 --name 'T2 sag, black, no aug, 200 layers'
