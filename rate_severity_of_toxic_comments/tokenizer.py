import os
import collections
from typing import Optional
import pandas as pd
import torch
from transformers import BasicTokenizer, PreTrainedTokenizer
from rate_severity_of_toxic_comments.embedding import build_embedding_matrix, load_embedding_model, check_OOV_terms, count_OOV_frequency
from rate_severity_of_toxic_comments.vocabulary import load_vocabulary, build_vocabulary


def create_recurrent_model_tokenizer(config, df, verbose=False):

    vocab_file_path = config["recurrent"]["vocab_file"]
    dataframe_cols = config["training"]["dataset"]["cols"]
    embedding_dim = config["recurrent"]["embedding_dimension"]
    tokenizer = NaiveTokenizer()
    vocab = load_vocabulary(vocab_file_path)
    if len(vocab) == 0:
        vocab, tokenizer = build_vocabulary(df,
                                            dataframe_cols, tokenizer, save_path=vocab_file_path)
    tokenizer.set_vocab(vocab)

    embedding_model = load_embedding_model(config["recurrent"])

    if verbose:
        oov = check_OOV_terms(embedding_model, vocab)
        # count_OOV_frequency(df, dataframe_cols, oov)

    embedding_matrix = build_embedding_matrix(
        embedding_model, embedding_dim, vocab)

    return tokenizer, embedding_matrix


class NaiveTokenizer(PreTrainedTokenizer):
    r"""
    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
    """

    def __init__(
        self,
        do_lower_case=True,
        never_split=None,
        unk_token="[UNK]",
        pad_token="[PAD]",
        **kwargs
    ):
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
            **kwargs,
        )
        self.basic_tokenizer = BasicTokenizer(
            do_lower_case=do_lower_case,
            never_split=never_split,
        )

    @property
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        return len(self.vocab)

    def set_vocab(self, vocab):
        self.vocab = vocab
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        return self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return torch.tensor(self.vocab.get(token, self.vocab.get(self.unk_token)))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string
