import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm 
import collections
from operator import itemgetter

from encoders import ContrieverQueryEncoder, ContrieverDocumentEncoder
from encoders import GTEQueryEncoder, GTEDocumentEncoder
from utils import load_topic, load_corpus, load_results, batch_iterator

def rerank(args, writer):

    if 'contriever' in args.encoder_path:
        qencoder = ContrieverQueryEncoder(
            args.encoder_path, 
            device=args.device,
            pooling=args.pooling,
            l2_norm=args.l2_norm,
            use_span_embedding=True if args.use_span_embedding else False
        )

        dencoder = ContrieverDocumentEncoder(
            args.encoder_path, 
            device=args.device,
            pooling='cls',
            l2_norm=args.l2_norm,
            use_span_embedding=True if args.use_span_embedding else False
        )

    if 'gte' in args.encoder_path:
        qencoder = GTEQueryEncoder(
            args.encoder_path, 
            device=args.device,
            pooling=args.pooling,
            l2_norm=args.l2_norm,
            use_span_embedding=True if args.use_span_embedding else False
        )

        dencoder = GTEDocumentEncoder(
            args.encoder_path, 
            device=args.device,
            pooling='cls',
            l2_norm=args.l2_norm,
            use_span_embedding=True if args.use_span_embedding else False
        )


    topics = load_topic(args.topic)
    qids = list(topics.keys())
    qtexts = list(topics.values())
    corpus_texts = load_corpus(args.corpus)
    results = load_results(args.input_run, topk=args.top_k)

    qvectors = list()

    # pre-encode queries
    for (start, end) in tqdm(
            batch_iterator(range(0, len(qids)), args.batch_size, True),
            total=(len(qids)//args.batch_size)+1
    ):
        qtexts_batch = qtexts[start: end]
        vectors = qencoder.batch_encode(qtexts_batch)
        qvectors.extend(vectors)

    # encode candidates and rerank for each query
    for i, qid in enumerate(tqdm(qids, total=len(qids))):
        qvector = qvectors[i]

        result = results[qid]
        docids = [docid for docid in result]  
        dtexts = [corpus_texts[docid] for docid in result]  

        scores = []
        for batch_dtexts in batch_iterator(dtexts, args.batch_size):
            batch_dvectors = dencoder.encode(batch_dtexts)
            # dot product (elemenet wise multiply then sum)
            batch_scores = np.dot(qvector, batch_dvectors.T).flatten()
            scores.extend(batch_scores)

        # sort by scores
        hits = {docids[idx]: scores[idx] for idx in range(len(scores))} 
        sorted_result =  \
                {k: v for k,v in sorted(hits.items(), key=itemgetter(1), reverse=True)}

        # write
        for i, (docid, score) in enumerate(sorted_result.items()):
            writer.write("{} Q0 {} {} {} dr4rerank\n".format(
                qid, docid, str(i+1), score)
            )  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_path", type=str, default=None)
    parser.add_argument("--topic", type=str, default=None)
    parser.add_argument("--corpus", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_run", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--device", type=str, default='cuda')
    # additiona model setup
    parser.add_argument("--pooling", type=str, default=None)
    parser.add_argument("--l2_norm", action='store_true', default=False)
    parser.add_argument("--use_span_embedding", action='store_true', default=False)
    args = parser.parse_args()

    os.makedirs(args.output.rsplit('/', 1)[0], exist_ok=True)

    writer = open(args.output, 'w')
    rerank(args, writer)
    writer.close()
