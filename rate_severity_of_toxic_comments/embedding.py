from collections import Counter
import collections
from tqdm import tqdm

import numpy as np
import gensim
import gensim.downloader as gloader


def load_embedding_model(config) -> gensim.models.keyedvectors.KeyedVectors:
    """
    Loads a pre-trained word embedding model via gensim library.

    :param model_type: name of the word embedding model to load.
    :param embedding_dimension: size of the embedding space to consider

    :return
        - pre-trained word embedding model (gensim KeyedVectors object)
    """
    model_type, embedding_dimension = config["embedding_type"], config["embedding_dimension"]

    download_path = ""

    # Find the correct embedding model name
    if model_type.strip().lower() == 'word2vec':
        download_path = "word2vec-google-news-300"
    elif model_type.strip().lower() == 'glove':
        download_path = "glove-wiki-gigaword-{}".format(embedding_dimension)
    elif model_type.strip().lower() == 'fasttext':
        download_path = "fasttext-wiki-news-subwords-300"
    else:
        raise AttributeError(
            "Unsupported embedding model type! Available ones: word2vec, glove, fasttext")

    # Check download
    try:
        emb_model = gloader.load(download_path)
    except ValueError as e:
        print("Invalid embedding model name! Check the embedding dimension:")
        print("Word2Vec: 300")
        print("Glove: 50, 100, 200, 300")
        raise e

    return emb_model


def build_embedding_matrix(embedding_model: gensim.models.keyedvectors.KeyedVectors, embedding_dim, tokenizer) -> np.ndarray:
    """
    Builds the embedding matrix of a specific dataset given a pre-trained word embedding model

    :param embedding_model: pre-trained word embedding model (gensim wrapper)
    :param word_to_idx: vocabulary map (word -> index) (dict)
    :param vocab_size: size of the vocabulary
    :param oov_terms: list of OOV terms (list)

    :return
        - embedding matrix that assigns a high dimensional vector to each word in the dataset specific vocabulary (shape |V| x d)
    """
    embedding_dimension, vocab = embedding_dim, tokenizer.get_vocab()
    embedding_matrix = np.zeros(
        (len(vocab), embedding_dimension), dtype=np.float32)

    for idx, word in tqdm(enumerate(vocab.items())):
        if idx == 0:
            # Zeros vector for padding token
            embedding_vector = np.zeros(embedding_dimension)
        else:
            try:
                # TODO: Why 2 vectors?
                embedding_vector = embedding_model[word][0]
            except (KeyError, TypeError):
                embedding_vector = np.random.uniform(
                    low=-0.05, high=0.05, size=embedding_dimension)

        embedding_matrix[idx] = embedding_vector

    return embedding_matrix


def check_OOV_terms(embedding_model: gensim.models.keyedvectors.KeyedVectors, vocab):
    """
    Checks differences between pre-trained embedding model vocabulary
    and dataset specific vocabulary in order to highlight out-of-vocabulary terms.

    :param embedding_model: pre-trained word embedding model (gensim wrapper)
    :param word_listing: dataset specific vocabulary (list)

    :return
        - list of OOV terms
    """

    embedding_vocabulary = set(embedding_model.index_to_key)
    vocab_set = set(vocab.keys())
    oov = vocab_set.difference(embedding_vocabulary)
    known_words = vocab_set.intersection(embedding_vocabulary)
    oov = list(oov)
    print('OOV words: ', len(oov))
    print('OOV examples: ', oov[:10])
    print('Known words embedding:', len(known_words))
    print('Pretrained embeddings size: ', len(list(embedding_vocabulary)))
    print(f'{(len(known_words) / len(list(embedding_vocabulary)) * 100):.2f}% Embedding used')
    return oov
