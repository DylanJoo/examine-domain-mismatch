index_dir=/home/dju/indexes/beir
data_dir=/home/dju/datasets/beir

# for exp in ind-cropping-mean-span_select_average-mse- ind-cropping-mean-span_select_average-kl-;do
# for exp in ind-cropping-cls-span_select_average-;do
for ckpt in 2000 4000 6000 8000 10000;do

    exp=ind-cropping-mean-span_select_average-
    encoder=/home/dju/examine-domain-mismatch/models/ckpt/contriever-${exp}trec-covid/checkpoint-${ckpt}
    pooling=mean

    for dataset in trec-covid;do

        # echo indexing...${dataset}...${exp}
        python3 retrieval/dense_index.py input \
            --corpus ${data_dir}/${dataset}/collection \
            --fields text title \
            --shard-id 0 \
            --shard-num 1 output \
            --embeddings ${index_dir}/${dataset}-${exp}contriever.faiss \
            --to-faiss encoder \
            --encoder-class contriever \
            --encoder ${encoder} \
            --pooling ${pooling} \
            --fields text title \
            --batch 32 \
            --max-length 256 \
            --device cuda

        echo searching...${dataset}...${exp}
        python retrieval/dense_search.py \
            --k 1000  \
            --index ${index_dir}/${dataset}-${exp}contriever.faiss \
            --encoder_path ${encoder} \
            --topic ${data_dir}/${dataset}/queries.jsonl \
            --batch_size 64 \
            --pooling ${pooling} \
            --device cuda \
            --output runs/${exp}contriever/run.beir.${dataset}.${exp}contriever.txt
    done

    for dataset in trec-covid;do
        echo -ne "beir-${dataset}.${ckpt}  | " 
        ~/trec_eval-9.0.7/trec_eval \
            -c -m ndcg_cut.10 -m recall.100 \
            ${data_dir}/${dataset}/qrels.beir-v1.0.0-${dataset}.test.txt \
            runs/${exp}contriever/run.beir.${dataset}.${exp}contriever.txt \
            | cut -f3 | sed ':a; N; $!ba; s/\n/ | /g'
    done

done
