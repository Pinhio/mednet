#!/bin/bash

#SBATCH --time=14:00:00
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --job-name=mednet
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:1

module load PyTorch
python train.py --gpu_id 0 --image_clip_type '2_filtTrue_min_cut_ahe_org_randTrue_20' --pretrain_path 'pretrain/resnet_200.pth' --model_depth 200 --n_epochs 200 --name 'Selective Cut (Dilation 20), 200 layers'
