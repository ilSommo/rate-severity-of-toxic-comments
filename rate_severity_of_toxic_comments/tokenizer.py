__version__ = '1.0.0-rc.1'
__author__ = 'Lorenzo Menghini, Martino Pulici, Alessandro Stockman, Luca Zucchini'


import collections

import torch
from transformers import BasicTokenizer, PreTrainedTokenizer

from rate_severity_of_toxic_comments.embedding import build_embedding_matrix, check_OOV_terms, count_OOV_frequency, load_embedding_model
from rate_severity_of_toxic_comments.vocabulary import build_vocabulary, load_vocabulary


class NaiveTokenizer(PreTrainedTokenizer):
    """
    Class containing a naive tokenizer.

    Attributes
    ----------
    basic_tokenizer : transformers.models.bert.tokenization_bert.BasicTokenizer
        Basic tokenizer.

    Methods
    -------
    __init__(self, do_lower_case=True, never_split=None, unk_token='[UNK]', pad_token='[PAD]', **kwargs)
        Initializes the tokenizer.
    do_lower_case(self)
        Returns the lowercasing flag.
    vocab_size(self)
        Returns the size of the vocabulary.
    set_vocab(self, vocab)
        Sets the vocabulary.
    get_vocab(self)
        Gets the vocabulary.
    _tokenize(self, text)
        Returns the tokenized text.
    _convert_token_to_id(self, token)
        Converts a token in an id.
    _convert_id_to_token(self, index)
        Converts an index in a token.
    convert_tokens_to_string(self, tokens)
        Converts a sequence of tokens in a single string.

    """

    def __init__(
            self,
            do_lower_case=True,
            never_split=None,
            unk_token='[UNK]',
            pad_token='[PAD]',
            **kwargs):
        """
        Initializes the model.

        Parameters
        ----------
        do_lower_case : bool, default True
            Lowercasing flag.
        never_split : list, optional
            List of tokens which will never be split during tokenization.
        unk_token : str
            Unkown token string.
        pad_token : str
            Pad token string.
        **kwargs : dict, optional
            Extra arguments.

        """
        super().__init__(
            do_lower_case=do_lower_case,
            do_basic_tokenize=True,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=None,
            pad_token=pad_token,
            cls_token=None,
            mask_token=None,
            tokenize_chinese_chars=True,
            strip_accents=None,
            **kwargs
        )
        self.basic_tokenizer = BasicTokenizer(
            do_lower_case=do_lower_case,
            never_split=never_split
        )

    @property
    def do_lower_case(self):
        """
        Returns the lowercasing flag.

        Returns
        -------
        lower_case : bool
            Lowercasing flag.

        """
        lower_case = self.basic_tokenizer.do_lower_case
        return lower_case

    @property
    def vocab_size(self):
        """
        Returns the size of the vocabulary.

        Returns
        -------
        vocab_size : int
            Size of the vocabulary.

        """
        vocab_size = len(self.vocab)
        return vocab_size

    def set_vocab(self, vocab):
        """
        Sets the vocabulary.

        Parameters
        ----------
        vocab : dict
            Vocabulary.

        """
        self.vocab = vocab
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])

    def get_vocab(self):
        """
        Gets the vocabulary.

        Returns
        ----------
        vocab : dict
            Vocabulary.

        """
        vocab = dict(self.vocab, **self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        """
        Returns the tokenized text.

        Parameters
        ----------
        text : str
            Text to tokenize.

        Returns
        ----------
        tokens : list
            List of tokens.

        """
        tokens = self.basic_tokenizer.tokenize(
            text, never_split=self.all_special_tokens)
        return tokens

    def _convert_token_to_id(self, token):
        """
        Converts a token in an id.

        Parameters
        ----------
        token : str
            Token to convert.

        Returns
        ----------
        id : torch.Tensor
            Id tensor.

        """
        id = torch.tensor(
            self.vocab.get(
                token, self.vocab.get(
                    self.unk_token)))
        return id

    def _convert_id_to_token(self, index):
        """
        Converts an index in a token.

        Parameters
        ----------
        index : int
            Index to convert.

        Returns
        ----------
        token : str
            Token.

        """
        token = self.ids_to_tokens.get(index, self.unk_token)
        return token

    def convert_tokens_to_string(self, tokens):
        """
        Converts a sequence of tokens in a single string.

        Parameters
        ----------
        tokens : list
            List of tokens.

        Returns
        ----------
        out_string : str
            Output string.

        """
        out_string = ' '.join(tokens).replace(' ##', '').strip()
        return out_string


def create_recurrent_model_tokenizer(config, df, verbose=False):
    """
    Creates a recurrent model tokenizer.

    Parameters
    ----------
    config : dict
        Configuration parameters.
    df : pandas.core.frame.DataFrame
        Dataset.
    verbose : bool, default False
        Verbosity flag.

    Returns
    -------
    tokenizer : rate_severity_of_toxic_comments.tokenizer.NaiveTokenizer
        Tokenizer.
    embedding_matrix : numpy.ndarray
        Embedding matrix.

    """
    vocab_file_path = config['recurrent']['vocab_file']
    dataframe_cols = config['training']['dataset']['cols']
    embedding_dim = config['recurrent']['embedding_dimension']
    tokenizer = NaiveTokenizer()
    vocab = load_vocabulary(vocab_file_path)
    if len(vocab) == 0:
        vocab, tokenizer = build_vocabulary(
            df, dataframe_cols, tokenizer, save_path=vocab_file_path)
    tokenizer.set_vocab(vocab)
    embedding_model = load_embedding_model(config['recurrent'])
    if verbose:
        oov = check_OOV_terms(embedding_model, vocab)
        count_OOV_frequency(df, dataframe_cols, oov)
    embedding_matrix = build_embedding_matrix(
        embedding_model, embedding_dim, vocab)
    return tokenizer, embedding_matrix
