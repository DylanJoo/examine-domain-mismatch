#!/bin/sh
#SBATCH --job-name=base
#SBATCH --partition gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=15G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate exa-dm_env

# Start the experiment.

# Setup experiments
for dataset in scidocs scifact;do

    backbone=contriever
    exp=facebook/contriever
    # Go
    torchrun --nproc_per_node 2 \
        unsupervised_learning/train_ind_cropping.py \
        --model_name ${exp} \
        --output_dir models/ckpt/${backbone}-baseline/${dataset} \
        --per_device_train_batch_size 32 \
        --temperature 0.1 \
        --pooling mean \
        --chunk_length 256 \
        --num_train_epochs 2 \
        --save_strategy epoch \
        --save_total_limit 2 \
        --warmup_ratio 0.1 \
        --fp16 \
        --report_to wandb --run_name ${dataset}-${backbone}-baseline \
        --train_data_dir /home/dju/datasets/beir/${dataset}/ind_cropping

done
