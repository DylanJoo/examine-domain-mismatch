dataset_dir=/home/dju/datasets

# Analyze msmarco and other datasets
# Unigram (the same setting as DAPT)
# python tools/analysis/vocab_overlap.py \
#     --files_path ${dataset_dir}/msmarco/collection/corpus.jsonl \
#     --files_path ${dataset_dir}/beir/scifact/collection/corpus.jsonl \
#     --files_path ${dataset_dir}/beir/scidocs/collection/corpus.jsonl \
#     --files_path ${dataset_dir}/beir/trec-covid/collection/corpus.jsonl \
#     --output_image analysis/analysis-beir.png \
#     --output_text analysis/analysis-beir.txt

# python tools/analysis/vocab_overlap.py \
#     --files_path ${dataset_dir}/msmarco/collection/corpus.jsonl \
#     --files_path ${dataset_dir}/lotte/lifestyle/test/collection/docs00.json \
#     --files_path ${dataset_dir}/lotte/recreation/test/collection/docs00.json \
#     --files_path ${dataset_dir}/lotte/science/test/collection/docs00.json \
#     --files_path ${dataset_dir}/lotte/technology/test/collection/docs00.json \
#     --files_path ${dataset_dir}/lotte/writing/test/collection/docs00.json \
#     --output_image analysis/analysis-lotte.png  \
#     --output_text analysis/analysis-lotte.txt

# Unigram and Bi-gram 
python tools/analysis/vocab_overlap.py \
    --files_path ${dataset_dir}/msmarco/collection/corpus.jsonl \
    --files_path ${dataset_dir}/beir/scifact/collection/corpus.jsonl \
    --files_path ${dataset_dir}/beir/scidocs/collection/corpus.jsonl \
    --files_path ${dataset_dir}/beir/trec-covid/collection/corpus.jsonl \
    --output_image analysis/2gram-analysis-beir.png \
    --output_text analysis/2gram-analysis-beir.txt \
    --min_df 0.01 --max_ngram 2

python tools/analysis/vocab_overlap.py \
    --files_path ${dataset_dir}/msmarco/collection/corpus.jsonl \
    --files_path ${dataset_dir}/lotte/lifestyle/test/collection/docs00.json \
    --files_path ${dataset_dir}/lotte/recreation/test/collection/docs00.json \
    --files_path ${dataset_dir}/lotte/science/test/collection/docs00.json \
    --files_path ${dataset_dir}/lotte/technology/test/collection/docs00.json \
    --files_path ${dataset_dir}/lotte/writing/test/collection/docs00.json \
    --output_image analysis/2gram-analysis-lotte.png  \
    --output_text analysis/2gram-analysis-lotte.txt \
    --min_df 0.01 --max_ngram 2

