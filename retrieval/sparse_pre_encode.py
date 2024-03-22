import json
from tqdm import tqdm
import argparse
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import Dataset
from utils import batch_iterator

from encoders import SpladeDocumentEncoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument("--collection_output", type=str)
    parser.add_argument("--model_name_or_dir", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--quantization_factor", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--minimum", type=float, default=0)
    parser.add_argument("--debug", action='store_true', default=False)
    args = parser.parse_args()

    # [model] encoder, tokenizer
    encoder = SpladeDocumentEncoder(
    	args.model_name_or_dir, args.tokenizer_name,
    	device=args.device,
    	pooling='max',
    	minimum=args.minimum,
    	quantization_factor=args.quantization_factor
    )
    encoder.eval()

    # [data] data
    data_list = []
    with open(args.collection, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line.strip())
            data_list.append(item)

            if len(data_list) >= 100 and args.debug:
                break

    dataset = Dataset.from_list(data_list)
    print(dataset)

    # [inference] preparing batch 
    bow_weights = []
    data_iterator = batch_iterator(dataset, args.batch_size, False)
    for batch in tqdm(data_iterator, total=len(dataset)//args.batch_size+1):
        batch_bow_weights = encoder.encode(
	    texts=batch['contents'],
	    max_length=args.max_length
       	)
        assert len(batch['contents']) == len(batch_vectors), \
                'Mismatched amount of examples'

        bow_weights += batch_bow_weights

    # [outout] 
    with open(args.collection_output, 'w') as fout:
	for i, example in enumerate(data_list):
	    example.update({"vector": vectors[i]})
	    fout.write(json.dumps(example, ensure_ascii=False)+'\n')

