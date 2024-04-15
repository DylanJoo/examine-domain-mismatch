#!/bin/sh
#SBATCH --job-name=ict
#SBATCH --partition gpu
#SBATCH --gres=gpu:4
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate exa-dm_env

# Start the experiment.
cd examine-domain-mismatch

## in ict, it uses the data from random cropping 
## but it only generate the 1 query-context pair per doc, so the batcth size is 16
torchrun --nproc_per_node 4 \
    unsupervised_learning/train_ict.py \
    --model_name facebook/contriever \
    --output_dir models/ckpt/contriever-ict-trec-covid \
    --save_total_limit 1 \
    --per_device_train_batch_size 16 \
    --max_steps 10000 \
    --fp16 \
    --chunk_length 256 \
    --query_in_block_prob 0.1 \
    --report_to wandb --run_name ict \
    --train_data_dir /home/dju/datasets/test_collection/rand_cropping

echo done
