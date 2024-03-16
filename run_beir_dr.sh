encoder=facebook/contriever
index_dir=/home/dju/indexes/beir
data_dir=/home/dju/datasets/beir

for dataset in scifact trec-covid scidocs;do

    echo indexing...${dataset}
    python3 encode/dense.py input \
        --corpus ${data_dir}/${dataset}/collection \
        --fields text title \
        --shard-id 0 \
        --shard-num 1 output \
        --embeddings ${index_dir}/${dataset}-contriever.faiss \
        --to-faiss encoder \
        --encoder-class contriever \
        --encoder ${encoder} \
        --fields text title \
        --batch 32 \
        --max-length 256 \
        --device cuda

    echo searching...${dataset}
    python retrieval/dense.py \
        --k 1000  \
        --index ${index_dir}/${dataset}-contriever.faiss \
        --encoder_path ${encoder} \
        --topic ${data_dir}/${dataset}/queries.jsonl \
        --batch_size 64 \
        --device cuda \
        --output runs/run.beir.${dataset}.contriever.txt
done

# Evaluation
for dataset in scifact trec-covid scidocs;do

    echo -ne "beir-${dataset}  | " 
    ~/trec_eval-9.0.7/trec_eval \
        -c -m ndcg_cut.10 -m recall.100 \
        ${data_dir}/${dataset}/qrels.beir-v1.0.0-${dataset}.test.txt \
        runs/run.beir.${dataset}.contriever.txt \
        | cut -f3 | sed ':a; N; $!ba; s/\n/ | /g'
done
