#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=rerank
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_titan_v:1
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

# Go
backbone=gte
for dataset in trec-covid;do
for alpha in 1.0;do
for beta in 1.0;do
for gamma in 1.0;do
    model=alpha.${alpha}-beta.${beta}-gamma.${gamma}
    exp=${model}
    encoder=/home/dju/examine-domain-mismatch/models/ckpt/${backbone}-${exp}/${dataset}/checkpoint-2330

    python retrieval/dense_rerank.py \
        --encoder_path ${encoder} \
        --topic ${data_dir}/${dataset}/queries.jsonl \
        --corpus ${data_dir}/${dataset}/collection/corpus.jsonl \
        --batch_size 32 \
        --input_run runs/bm25/run.beir.${dataset}.bm25-multifield.txt \
        --top_k 1000 \
        --device cuda \
        --pooling mean \
        --output runs/${backbone}-bm25rerank/run.beir.${dataset}.${backbone}.${exp}.txt

    echo -ne "beir-${dataset}.${exp}.${model} | " 
    ~/trec_eval-9.0.7/trec_eval \
        -c -m ndcg_cut.10 -m recall.100 \
        ${data_dir}/${dataset}/qrels.beir-v1.0.0-${dataset}.test.txt \
        runs/${backbone}-bm25rerank/run.beir.${dataset}.${backbone}.${exp}.txt \
        | cut -f3 | sed ':a; N; $!ba; s/\n/ | /g'
done
done
done
done
