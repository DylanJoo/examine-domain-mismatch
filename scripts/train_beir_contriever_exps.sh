#!/bin/sh
#SBATCH --job-name=ctr
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_titan_v:2
#SBATCH --mem=15G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate exa-dm_env

# backbone=gte
# ckpt=thenlper/gte-base
backbone=contriever
ckpt=facebook/contriever

# Start the experiment.
for dataset in trec-covid scidocs scifact;do
for method in span_average span_key_extract span_select_weird span_extract;do
    exp=${method}

    # Go
    torchrun --nproc_per_node 2 \
        unsupervised_learning/train_ind_cropping.py \
        --model_name ${ckpt} \
        --output_dir models/ckpt/${backbone}-${exp}/${dataset} \
        --per_device_train_batch_size 32 \
        --temperature 0.1 \
        --temperature_span 0.5 \
        --pooling cls \
        --span_pooling ${exp} \
        --chunk_length 256 \
        --span_sent_interaction cont \
        --num_train_epochs 2 \
        --save_strategy epoch \
        --save_total_limit 2 \
        --warmup_ratio 0.1 \
        --fp16 \
        --report_to wandb --run_name ${dataset}-${exp} \
        --train_data_dir /home/dju/datasets/beir/${dataset}/ind_cropping
done
done
