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

def build_vectors(file, field, tdidf_vectorizer):
    # vectorizer, documents, min_df, ngram_range):
    text = load_data(file, field)
    doc_terms_matrix = tfidf_vectorizer.fit_transform(text)
    doc_terms_matrix = doc_terms_matrix.toarray()
    average_term_vector = doc_terms_matrix.mean(0) # average across documents
    return average_term_vector * 100 # rescale

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--files_path", nargs="+", action='append')
    parser.add_argument("--output_image", default='dev.png')
    parser.add_argument("--output_text", default='dev.txt')
    parser.add_argument("--min_df", default=3, type=str)
    parser.add_argument("--max_ngram", default=1, type=int)
    args = parser.parse_args()
    files = [p[0] for p in args.files_path]
    print(files)

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
            ngram_range=(1, args.max_ngram),
            binary=True # only extract the appearance
    )
    count_vectorizer.fit(tqdm(all_documents))
    all_vocabulary = set(count_vectorizer.vocabulary_.keys())
    del all_documents

    tfidf_vectorizer = TfidfVectorizer(
            min_df=number,
            stop_words="english", 
            vocabulary=all_vocabulary,
            ngram_range=(1, args.max_ngram)
    )
    del count_vectorizer

    # Build tfidf vectors for the datasets
    vectors = {}
    for path in files:
        if 'scidocs' in path:
            vectors['SD'] = build_vectors(path, 'text', tfidf_vectorizer)
        elif 'scifact' in path:
            vectors['SF'] = build_vectors(path, 'text', tfidf_vectorizer)
        elif 'trec-covid' in path:
            vectors['TC'] = build_vectors(path, 'text', tfidf_vectorizer)
        elif 'msmarco' in path:
            vectors['MS'] = build_vectors(path, 'text', tfidf_vectorizer)
        elif 'lotte' in path:

            if 'lifestyle' in path:
                vectors['lotte-li'] = build_vectors(path, 'contents', tfidf_vectorizer)
            elif 'recreation' in path:
                vectors['lotte-re'] = build_vectors(path, 'contents', tfidf_vectorizer)
            elif 'science' in path:
                vectors['lotte-sc'] = build_vectors(path, 'contents', tfidf_vectorizer)
            elif 'writing' in path:
                vectors['lotte-wr'] = build_vectors(path, 'contents', tfidf_vectorizer)
            elif 'technology' in path:
                vectors['lotte-te'] = build_vectors(path, 'contents', tfidf_vectorizer)

    if len(vectors.keys()) <= 1:
        raise ValueError('At least two collections required.')

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
