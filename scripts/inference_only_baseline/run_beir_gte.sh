#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=retrieval
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=15G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=%x.%j.debug

# Set-up the environment.
source ${HOME}/.bashrc
conda activate exa-dm_env

# Start the experiment.
index_dir=${HOME}/indexes/beir
data_dir=${HOME}/datasets/beir

# Setting of encoders
cd ${HOME}/examine-domain-mismatch/

# Go
for dataset in trec-covid scidocs scifact;do
    backbone=gte
    model=thenlper/gte-base
    exp=baseline

    echo indexing...
    python3 retrieval/dense_index.py input \
        --corpus ${data_dir}/${dataset}/collection \
        --fields text title \
        --shard-id 0 \
        --shard-num 1 output \
        --embeddings ${index_dir}/${dataset}-${backbone}-${exp}.faiss \
        --to-faiss encoder \
        --encoder-class ${backbone} \
        --encoder ${model} \
        --pooling mean \
        --l2-norm \
        --fields text title \
        --batch 32 \
        --max-length 256 \
        --device cuda

    echo searching...
    python retrieval/dense_search.py \
        --k 100  \
        --index ${index_dir}/${dataset}-${backbone}-${exp}.faiss \
        --encoder_path ${model} \
        --encoder_class ${backbone} \
        --topic ${data_dir}/${dataset}/queries.jsonl \
        --batch_size 64 \
        --pooling mean \
        --l2_norm \
        --device cuda \
        --output runs/${backbone}-${exp}/run.beir.${dataset}.${backbone}.${exp}.txt

    echo -ne "beir-${dataset}.${exp}.${model} | " 
    ~/trec_eval-9.0.7/trec_eval \
        -c -m ndcg_cut.10 -m recall.100 \
        ${data_dir}/${dataset}/qrels.beir-v1.0.0-${dataset}.test.txt \
        runs/${backbone}-${exp}/run.beir.${dataset}.${backbone}.${exp}.txt \
        | cut -f3 | sed ':a; N; $!ba; s/\n/ | /g'
done
