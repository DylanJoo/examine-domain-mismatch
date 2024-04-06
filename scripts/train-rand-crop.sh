#!/bin/sh
#SBATCH --job-name=randcrop
#SBATCH --partition gpu
#SBATCH --gres=gpu:4
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate exa-dm_env

# Start the experiment.
cd examine-domain-mismatch

# contriever with default independent cropping
## in random cropping, it uses 2 spans at the same time
## It will forward bidirectionally, so the batch size should be 8
torchrun --nproc_per_node 4 \
    unsupervised_learning/train_rand_cropping.py \
    --model_name facebook/contriever \
    --output_dir models/ckpt/contriever-rand-cropping-trec-covid \
    --per_device_train_batch_size 8 \
    --max_steps 10000 \
    --save_total_limit 1 \
    --fp16 \
    --chunk_length 256 \
    --report_to wandb --run_name random_cropping_wo_mlm \
    --train_data_dir /home/dju/datasets/test_collection/rand_cropping

echo done
