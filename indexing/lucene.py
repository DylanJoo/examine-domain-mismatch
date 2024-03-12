    nd description and metadata(['category', 'template', 'attrs', 'info'])
    # COLLECTION_DIR=/tmp2/trec/pds/data/collection/lucene/full/
    # INDEX=/tmp2/trec/pds/indexes/bm25-full/
    # python -m pyserini.index.lucene \
    #   --collection JsonCollection \
    #   --input $COLLECTION_DIR \
    #   --index $INDEX \
    #   --generator DefaultLuceneDocumentGenerator \
    #   --threads 4 \
    #   --storePositions --storeDocvectors --storeRaw
    #
    # # encode title and description
    # COLLECTION_DIR=/tmp2/trec/pds/data/collection/lucene/simplified/
    # INDEX=/tmp2/trec/pds/indexes/bm25-sim/
    # python -m pyserini.index.lucene \
    #   --collection JsonCollection \
    #   --input $COLLECTION_DIR \
    #   --index $INDEX \
    #   --generator DefaultLuceneDocumentGenerator \
    #   --threads 4 \
    #   --storePositions --storeDocvectors --storeRaw

    # encode title and description
    # this one is same as dense retrieved jsonl
    COLLECTION_DIR=/tmp2/trec/pds/data/collection/lucene/simplified_title/
    INDEX=/tmp2/trec/pds/indexes/bm25-sim-title/
    python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input $COLLECTION_DIR \
    --index $INDEX \
    --generator DefaultLuceneDocumentGenerator \
    --threads 4 \
    --storePositions --storeDocvectors --storeRaw
