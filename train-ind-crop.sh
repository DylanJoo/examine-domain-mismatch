#!/bin/sh
#SBATCH --job-name=ssldr
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
# cd examine-domain-mismatch

# Setup experiments
MAX_STEPS=5000
MAX_CKPTS=2

# Setting of lexical-enhanced DR
pooling=cls
learn=cont
span_pooling=span_select_weird
exp=ind-cropping-${pooling}-${span_pooling}-${learn}

# Go
torchrun --nproc_per_node 2 \
    unsupervised_learning/train_ind_cropping.py \
    --model_name facebook/contriever \
    --output_dir models/ckpt/contriever-${exp}-trec-covid \
    --per_device_train_batch_size 32 \
    --temperature 0.5 \
    --pooling $pooling \
    --temperature_span 1.0 \
    --distil_from_sentence ${learn} \
    --span_pooling $span_pooling \
    --chunk_length 256 \
    --save_steps 2500 \
    --save_total_limit $MAX_CKPTS \
    --max_steps $MAX_STEPS \
    --warmup_ratio 0.1  \
    --fp16 \
    --report_to wandb --run_name ${exp} \
    --train_data_dir /home/dju/datasets/test_collection/ind_cropping

echo done