for domain in science writing lifestyle recreation technology;do
    for subset in dev test;do

        # Convert tsv to jsonl 
        base_dir=/home/dju/datasets/lotte/${domain}/${subset}
        tsv=${base_dir}/collection.tsv
        data_dir=${base_dir}/collection

        if [ -e ${data_dir}/docs00.json ]
        then
            echo "jsonl collection exists."
        else
            python retrieval/convert_collection_to_jsonl.py \
                --collection-path ${tsv} \
                --output-folder ${data_dir}
            rm ${tsv}
        fi

        # Indexing
        index=/home/dju/indexes/lotte-${domain}-${subset}.lucene
        # python -m pyserini.index.lucene \
        #     --collection JsonCollection \
        #     --input ${data_dir} \
        #     --index ${index} \
        #     --generator DefaultLuceneDocumentGenerator \
        #     --threads 4

        # Search (forum and search)
        python retrieval/bm25.py \
            --k 1000 --k1 0.9 --b 0.4 \
            --index ${index} \
            --topic ${base_dir}/questions.forum.tsv \
            --batch_size 8 \
            --output runs/run.lotte-${domain}-${subset}.forum.bm25.txt

        python retrieval/bm25.py \
            --k 1000 --k1 0.9 --b 0.4 \
            --index ${index} \
            --topic ${base_dir}/questions.search.tsv \
            --batch_size 8 \
            --output runs/run.lotte-${domain}-${subset}.search.bm25.txt
    done
done
