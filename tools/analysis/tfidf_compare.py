# This code was cloned from: http://github.com/allenai/dont-stop-pretraining
# Reference: https://arxiv.org/pdf/2004.10964.pdf 
import sys
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import json
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import argparse
from collections import defaultdict

sns.set(context="paper", style="white", font_scale=2.1)

def load_data(data_path, field='text'):
    examples = []
    with tqdm(open(data_path, "r"), desc=f"loading {data_path}") as f:
        for line in f:
            line = line.strip()
            if line:
                if data_path.endswith(".jsonl") or data_path.endswith(".json"):
                    example = json.loads(line)
                else:
                    example = {field: line}
                text = example[field]
                examples.append(text)
    return examples

def build_vectors(file, field, min_df):
    dd = defaultdict(float)
    text = load_data(file, field)
    tfidf_vectorizer = TfidfVectorizer(
            min_df=min_df,
            stop_words="english", 
            ngram_range=(args.min_ngram, args.max_ngram)
    )
    doc_terms_matrix = tfidf_vectorizer.fit_transform(text)
    average_term_vector = doc_terms_matrix.mean(0) * 100
    average_term_vector = np.asarray(average_term_vector).flatten()
    term_list = tfidf_vectorizer.get_feature_names_out()
    vector_dict = {k: v for k, v in zip(term_list, average_term_vector)}
    dd.update(vector_dict)
    return dd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--files_path", nargs="+", action='append')
    parser.add_argument("--output_image", default='dev.png')
    parser.add_argument("--output_text", default='dev.txt')
    parser.add_argument("--min_df", default=3, type=str)
    parser.add_argument("--min_ngram", default=1, type=int)
    parser.add_argument("--max_ngram", default=1, type=int)
    args = parser.parse_args()
    files = [p[0] for p in args.files_path]

    if '.' in args.min_df:
        number = float(args.min_df)
    else:
        number = int(float(args.min_df))

    # Pre-loaded corpora
    all_documents = []
    for path in files:
        if 'lotte' in path:
            all_documents += load_data(path, field='contents')
        else:
            all_documents += load_data(path)

    # Construct a tf-idf vectorizer with all doucments
    count_vectorizer = CountVectorizer(
            min_df=number,
            stop_words="english", 
            ngram_range=(args.min_ngram, args.max_ngram),
            binary=True # only extract the appearance
    )
    count_vectorizer.fit(tqdm(all_documents))
    all_vocabulary = list(count_vectorizer.vocabulary_.keys())
    del all_documents

    # Build tfidf vectors for the datasets
    vectors_ = {}
    for path in files:
        kwargs = dict(file=path, field='text', min_df=number)
        if 'scidocs' in path:
            vectors_['SD'] = build_vectors(**kwargs)
        elif 'scifact' in path:
            vectors_['SF'] = build_vectors(**kwargs)
        elif 'trec-covid' in path:
            vectors_['TC'] = build_vectors(**kwargs)
        elif 'msmarco' in path:
            vectors_['MS'] = build_vectors(**kwargs)
        elif 'lotte' in path:
            kwargs['field'] = 'contents'
            if 'lifestyle' in path:
                vectors_['lotte-li'] = build_vectors(**kwargs)
            elif 'recreation' in path:
                vectors_['lotte-re'] = build_vectors(**kwargs)
            elif 'science' in path:
                vectors_['lotte-sc'] = build_vectors(**kwargs)
            elif 'writing' in path:
                vectors_['lotte-wr'] = build_vectors(**kwargs)
            elif 'technology' in path:
                vectors_['lotte-te'] = build_vectors(**kwargs)

    if len(vectors_.keys()) <= 1:
        raise ValueError('At least two collections required.')

    # reorganize the array with shared vocabulary
    vectors = defaultdict(list)
    for dataset_key, vector_dict in vectors_.items():
        for i, vocab in enumerate(all_vocabulary):
            vectors[dataset_key].append(vector_dict[vocab])

    vectors_matrix = np.array([vectors[k] for k in vectors.keys()]) 
    data = np.corrcoef(vectors_matrix)

    with open(args.output_text, "w") as f:
        for row in data:
            f.write(str(row.tolist())+'\n')
    
    # printing
    labels = list(vectors.keys())
    ax = sns.heatmap(data, 
            cmap="Blues", xticklabels=labels, annot=True, 
            fmt=".1f", cbar=False, yticklabels=labels
    )
    #
    plt.yticks(rotation=0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(args.output_image, dpi=300)
    print('done')
