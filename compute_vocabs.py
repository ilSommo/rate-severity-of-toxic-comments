import os
import argparse

import pandas as pd
from tqdm import tqdm

from rate_severity_of_toxic_comments.embedding import check_OOV_terms, count_OOV_frequency, load_embedding_model
from rate_severity_of_toxic_comments.preprocessing import apply_preprocessing_pipelines
from rate_severity_of_toxic_comments.tokenizer import NaiveTokenizer, load_vocabulary
from rate_severity_of_toxic_comments.vocabulary import build_vocabulary


def store_vocab(df, cols, vocab_file, pipelines):
    df = df.copy()
    pipelines.sort()
    vocab_to_load = vocab_file[:-4]

    for pipeline in pipelines:
        vocab_to_load += '_' + pipeline
    vocab_to_load += '.txt'

    print(f'New vocab file path {vocab_to_load}')

    if len(pipelines) == 0 or os.path.isfile(vocab_to_load):
        vocab = load_vocabulary(vocab_to_load)
        if len(vocab) > 0:
            return vocab

    if len(pipelines) > 0:
        sentences_in_cols = [v for col in cols for v in df[col].values]
        num_sentences = len(sentences_in_cols)
        print(f"Dataset comments to preprocess: {num_sentences}")
        print(f"Pipelines to apply: {pipelines}")

        for col in cols:
            for i in tqdm(df.index, total=len(df)):
                df.at[i, col], bad_words_count, count = apply_preprocessing_pipelines(df.at[i, col], pipelines)

    tokenizer = NaiveTokenizer()
    vocab, tokenizer = build_vocabulary(df, cols, tokenizer, save_path=vocab_to_load)
    return vocab

def print_oov_stats(vocab, df, cols, embedding_model):
    oov = check_OOV_terms(embedding_model, vocab)
    count_OOV_frequency(df, cols, oov)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file')
    parser.add_argument('--dataset_file')
    parser.add_argument('--cols')
    parser.add_argument('--embedding_size', default=300)
    args = parser.parse_args()

    embedding_model = load_embedding_model({
        "embedding_type": "glove",
        "embedding_dimension": args.embedding_size
    })
    vocab_file = args.vocab_file
    df = pd.read_csv(args.dataset_file)
    cols = args.cols.split(",")
    
    combinations = [
        [],
        ["LOWER", "PUNCTUATION", "WHITESPACES"],
        ["LOWER", "PUNCTUATION", "WHITESPACES", "NUMBERS"],
        ["LOWER", "PUNCTUATION", "WHITESPACES", "TRIPLE"],
        ["LOWER", "PUNCTUATION", "WHITESPACES", "TRIPLE", "NUMBERS"],
    ]

    for subset in combinations:
        vocab = store_vocab(df, cols, vocab_file, subset)
        print_oov_stats(vocab, df, cols, embedding_model)