import os
import collections
from typing import Optional
import pandas as pd
import torch
from transformers import BasicTokenizer, PreTrainedTokenizer
from rate_severity_of_toxic_comments.preprocessing import apply_preprocessing_pipelines
from rate_severity_of_toxic_comments.embedding import build_embedding_matrix, check_OOV_terms, load_embedding_model
from rate_severity_of_toxic_comments.vocabulary import load_vocabulary, build_vocabulary_and_tokenize


def create_recurrent_model_tokenizer(config):
    vocab_file_path = config["vocab_file"]
    dataframe_path = config["training_set"]["path"]
    dataframe_cols = config["training_set"]["cols"]
    embedding_dim = config["embedding_dimension"]
    tokenizer = NaiveTokenizer()
    vocab = load_vocabulary(vocab_file_path)
    df = pd.read_csv(dataframe_path)
    if len(vocab) == 0:
        vocab, tokenizer = build_vocabulary_and_tokenize(df,
                                                         dataframe_cols, tokenizer, save_path=vocab_file_path)
    else:
        tokenizer.set_vocab(vocab)
        tokenizer.tokenize_comments(df, dataframe_cols)

    embedding_model = load_embedding_model(config)
    check_OOV_terms(embedding_model, vocab)
    embedding_matrix = build_embedding_matrix(
        embedding_model, embedding_dim, tokenizer)
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
        vocab_file=None,
        preprocessing_pipelines=[],
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

        if vocab_file:
            self.set_vocab(load_vocabulary(vocab_file))

        self.preprocessing_pipelines = preprocessing_pipelines
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
        processed_text, bad_words_counter, preprocessing_metric = apply_preprocessing_pipelines(
            text, self.preprocessing_pipelines)
        return self.basic_tokenizer.tokenize(processed_text, never_split=self.all_special_tokens)

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

    def tokenize_comments(self, df, csv_cols):
        # If vocab is empty, populate it with training sets
        sentences_in_cols = [v for col in csv_cols for v in df[col].values]
        num_sentences = len(sentences_in_cols)
        percentage_printed = 0.0
        print(f" Tokenizing on datasets columns {csv_cols}")
        for index, sentence in enumerate(sentences_in_cols):
            percentage = round(index / num_sentences, 2)
            if percentage == 0.25 and percentage_printed == 0.0:
                print(f"25% tokenization done")
                percentage_printed = 0.25
            elif percentage == 0.50 and percentage_printed == 0.25:
                print(f"50% tokenization done")
                percentage_printed = 0.50
            elif percentage == 0.75 and percentage_printed == 0.5:
                print(f"75% tokenization done")
                percentage_printed = 0.75
            self._tokenize(sentence)
