# python -m pyserini.index.lucene \
#     --collection BeirFlatCollection \
#     --input /home/dju/datasets/msmarco/collection \
#     --index /home/dju/indexes/msmarco.lucene \
#     --generator DefaultLuceneDocumentGenerator \
#     --threads 4

# python retrieval/bm25.py \
#     --k 1000 --k1 0.82 --b 0.68 \
#     --index /home/dju/indexes/msmarco.lucene \
#     --topic /home/dju/datasets/msmarco/queries.dev-subset.txt \
#     --batch_size 8 \
#     --output runs/run.msmarco-dev-subset.bm25 \

echo 'Indexing msmarco' # done by multi-parallel process
# python encode/dense.py input \
#     --corpus /home/dju/datasets/msmarco/collection \
#     --fields text \
#     --shard-id 0 \
#     --shard-num 1 output \
#     --embeddings /home/dju/indexes/msmarco-contriever.faiss \
#     --to-faiss encoder \
#     --encoder-class contriever \
#     --encoder facebook/contriever \
#     --fields text \
#     --batch 32 \
#     --max-length 256 \
#     --device cuda

python -m pyserini.index.merge_faiss_indexes \
    --prefix /home/dju/indexes/msmarco-contriever.faiss4_ \
    --shard-num 4

echo 'Searching msmarco'
python retrieval/dense.py \
    --k 1000  \
    --index /home/dju/indexes/msmarco-contriever.faiss \
    --encoder_path facebook/contriever \
    --topic /home/dju/datasets/msmarco/queries.dev-subset.txt \
    --batch_size 64 \
    --device cuda \
    --output runs/run.msmarco-dev-subset.contriever.txt

# Evaluation
echo -ne "msmarco-dev-subset | "
~/trec_eval-9.0.7/trec_eval \
    -c -m ndcg_cut.10 -m recall.100 \
    /home/dju/datasets/msmarco/qrels.msmarco-passage.dev-subset.txt \
    runs/run.msmarco-dev-subset.bm25.txt \
    | cut -f3 | sed ':a; N; $!ba; s/\n/ | /g'

~/trec_eval-9.0.7/trec_eval \
    -c -m ndcg_cut.10 -m recall.100 \
    /home/dju/datasets/msmarco/qrels.msmarco-passage.dev-subset.txt \
    runs/run.msmarco-dev-subset.contriever.txt \
    | cut -f3 | sed ':a; N; $!ba; s/\n/ | /g'
