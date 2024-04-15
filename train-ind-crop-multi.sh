#!/bin/sh
#SBATCH --job-name=multivec
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

# backbone=gte
# ckpt=thenlper/gte-base
backbone=contriever
ckpt=facebook/contriever

# Start the experiment.
for alpha in 1.0;do
for beta in 0.0;do
for gamma in 0.0;do
for delta in 2.0;do
for dataset in trec-covid;do
    exp=alpha.${alpha}-beta.${beta}-gamma.${gamma}-delta.${delta}

    # Go
    torchrun --nproc_per_node 2 \
        unsupervised_learning/train_ind_cropping.py \
        --model_name ${ckpt} \
        --output_dir models/ckpt/${backbone}-${exp}/${dataset} \
        --per_device_train_batch_size 32 \
        --temperature 0.1 \
        --temperature_span 0.5 \
        --pooling mean \
        --chunk_length 256 \
        --late_interaction \
        --num_train_epochs 1 \
        --save_strategy epoch \
        --save_total_limit 2 \
        --warmup_ratio 0.1 \
        --fp16 \
        --norm_doc --norm_query \
        --alpha $alpha --beta $beta --gamma $gamma --delta $delta \
        --report_to wandb --run_name ${dataset}-${exp} \
        --wandb_project debug \
        --train_data_dir /home/dju/datasets/beir/${dataset}/ind_cropping
done
done
done
done
done
