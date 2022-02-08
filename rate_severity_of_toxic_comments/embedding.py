__version__ = '1.0.0-rc.1'
__author__ = 'Lorenzo Menghini, Martino Pulici, Alessandro Stockman, Luca Zucchini'


import gensim
import gensim.downloader as gloader
import numpy as np
from tqdm import tqdm


AVAILABLE_EMBEDDINGS = [('word2vec', 300), ('glove', 50), ('glove', 100),
                        ('glove', 200), ('glove', 300), ('fasttext', 300)]


def build_embedding_matrix(
        embedding_model: gensim.models.keyedvectors.KeyedVectors,
        embedding_dim,
        vocab) -> np.ndarray:
    """
    Builds the embedding matrix.

    Parameters
    ----------
    embedding_model : gensim.models.keyedvectors.KeyedVectors
        Embedding model.
    embedding_dim : int
        Embedding dimension.
    vocab : dict
        Vocabulary.

    Returns
    -------
    embedding_matrix : numpy.ndarray
        Embedding matrix.

    """
    embedding_matrix = np.zeros(
        (len(vocab), embedding_dim), dtype=np.float32)
    print(f'Building embedding matrix')
    for idx, (word, word_idx) in tqdm(
            enumerate(vocab.items()), total=len(vocab)):
        if idx == 0:
            embedding_vector = np.zeros(embedding_dim)
        else:
            try:
                embedding_vector = embedding_model[word]
            except (KeyError, TypeError):
                embedding_vector = np.random.uniform(
                    low=-0.05, high=0.05, size=embedding_dim)
        embedding_matrix[idx] = embedding_vector
    return embedding_matrix


def check_OOV_terms(
        embedding_model: gensim.models.keyedvectors.KeyedVectors,
        vocab):
    """
    Highlights out-of-vocabulary terms.

    Parameters
    ----------
    embedding_model : gensim.models.keyedvectors.KeyedVectors
        Embedding model.
    vocab : dict
        Vocabulary.

    Returns
    -------
    oov : list
        List of out-of-vocabulary terms.

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


def count_OOV_frequency(df, cols, oov):
    """
    Counts out-of-vocabulary frequency.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Dataset.
    cols : list
        List of column names.
    oov : list
        List of out-of-vocabulary terms.

    """
    counts = {word: df[col].str.count(word).sum()
              for word in oov for col in cols}
    sorted_counts = {
        k: v for idx,
        (k,
         v) in enumerate(
            sorted(
                counts.items(),
                key=lambda item: item[1],
                reverse=True)) if idx < 10}
    print(f'Top 10 OOV occurrencies\n{sorted_counts}')


def load_embedding_model(
        model_params) -> gensim.models.keyedvectors.KeyedVectors:
    """
    Loads a pre-trained word embedding model.

    Parameters
    ----------
    model_params : dict
        Model parameters.

    Returns
    -------
    emb_model : gensim.models.keyedvectors.KeyedVectors
        Embedding model.

    """
    model_type, embedding_dimension = model_params['embedding_type'], model_params['embedding_dimension']
    download_path = ''
    if model_type.strip().lower() == 'word2vec':
        download_path = 'word2vec-google-news-300'
    elif model_type.strip().lower() == 'glove':
        download_path = 'glove-wiki-gigaword-{}'.format(embedding_dimension)
    elif model_type.strip().lower() == 'fasttext':
        download_path = 'fasttext-wiki-news-subwords-300'
    else:
        raise AttributeError(
            'Unsupported embedding model type! Available ones: word2vec, glove, fasttext')
    try:
        print('Loading embedding model')
        emb_model = gloader.load(download_path)
        print('Embedding model loaded')
    except ValueError as e:
        print('Invalid embedding model name! Check the embedding dimension:')
        print('Word2Vec: 300')
        print('Glove: 50, 100, 200, 300')
        raise e
    return emb_model
