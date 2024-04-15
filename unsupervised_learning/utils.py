from tqdm import tqdm
import collections
import json
from datasets import Dataset

def tokenization(example, tokenizer):
    q_inputs = tokenizer.batch_encode_plus(
        example['query'], 
        max_length=64,
        add_special_tokens=True,
        return_tensors='pt'
    )

    c_inputs = tokenizer.batch_encode_plus(
        example['positive'] + example['negative'],
        max_length=256,
        add_special_tokens=True,
        return_tensors='pt'
    )

    return {'q_tokens': q_inputs['input_ids'], 
            'c_tokens': c_inputs['input_ids']}

def convert_qrels_to_dataset(path, queries, corpus_texts, tokenizer):

    positives = collections.defaultdict(list)
    negatives = collections.defaultdict(list)
    all_negative_docids = []

    # load qrels
    with open(path, 'r') as f:
        for line in f:
            qid, _, docid, rel = line.strip().split()
            if int(rel) > 0:
                positives[qid].append(corpus_texts[docid].strip())
            else:
                negatives[qid].append(corpus_texts[docid].strip())

    # convert qrels to samples
    sample_list = []
    for qid in positives:
        query = queries[qid]
        docid = random.sample(positives[qid], 1)[0]
        pos = corpus_texts[docid]
        docid = random.sample(negative[qid], 1)[0]
        neg = corpus_texts[docid]

        sample_list.append({'query': query, 'positive': pos, 'negative':neg})

    dataset = Dataset.from_list(sample_list)

    # customization
    datast = data.map(tokenization, 
        batched=False, fn_kwargs={"tokenizer": tokenizer})

    return dataset


