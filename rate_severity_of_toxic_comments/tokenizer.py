import os
import collections
from typing import Optional

import torch
import torchtext
from transformers import BasicTokenizer, PreTrainedTokenizer
from rate_severity_of_toxic_comments.preprocessing import apply_preprocessing_pipelines


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


def build_vocab(df, cols, tokenizer: PreTrainedTokenizer, min_freq=1, save_path=None):
    """
    Returns a Vocab object containing all the tokens appearing in the `cols` columns of the dataframe `df`
    """
    # vocab = collections.OrderedDict()
    counter = collections.Counter()

    # Append to vocab special tokens expected by the tokenizer
    for token in tokenizer.special_tokens_map.values():
        counter[token] = min_freq + 1

    for sentence in [v for col in cols for v in df[col].values]:
        # for token in tokenizer._tokenize(sentence:
        #     vocab[token] = index
        counter.update(tokenizer._tokenize(sentence))

    v = torchtext.vocab.vocab(counter, min_freq=min_freq).get_stoi()
    if save_path:
        save_vocabulary(save_path, v)
    return v, counter


def save_vocabulary(save_path: str, vocab: collections.OrderedDict) -> tuple[str]:
    index = 0
    with open(save_path, "w", encoding="utf-8") as writer:
        for token, token_index in sorted(vocab.items(), key=lambda kv: kv[1]):
            if index != token_index:
                # logger.warning(
                #     f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                #     " Please check that the vocabulary is not corrupted!"
                # )
                index = token_index
            writer.write(token + "\n")
            index += 1


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
            self.vocab = load_vocab(vocab_file)
            self.ids_to_tokens = collections.OrderedDict(
                [(ids, tok) for tok, ids in self.vocab.items()])

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
        processed_text = apply_preprocessing_pipelines(
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

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple[str]:
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix +
                                 "-" if filename_prefix else "") + "naive-vocab"
            )
        else:
            vocab_file = (filename_prefix +
                          "-" if filename_prefix else "") + save_directory
        save_vocabulary(vocab_file, self.vocab)
        return (vocab_file,)
