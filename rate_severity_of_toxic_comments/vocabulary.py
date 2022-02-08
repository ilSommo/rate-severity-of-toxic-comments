import collections
import torchtext
import os

from tqdm import tqdm


def get_preprocess_filenames(pipelines, vocab_file, dataset_file=None) -> str:
    """Gets the appropriate vocabulary file path to load based on the preprocessing pipeline to apply"""

    pipelines.sort()
    if pipelines is None or len(pipelines) == 0:
        return None

    dataset_to_load = dataset_file[:-4] if dataset_file else ""
    vocab_to_load = vocab_file[:-4]

    for pipeline in pipelines:
        dataset_to_load += '_' + pipeline
        vocab_to_load += '_' + pipeline

    dataset_to_load += '.csv'
    vocab_to_load += '.txt'

    if dataset_file:
        return vocab_to_load, dataset_to_load
    else:
        return vocab_to_load


def load_vocabulary(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab_dict = collections.OrderedDict()
    print(f"Loading vocabulary from {vocab_file}")
    if not os.path.isfile(vocab_file):
        print(f"Vocabulary file not found")
        open(vocab_file, 'a+').close()

    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()

    for index, token in tqdm(enumerate(tokens), total=len(tokens)):
        token = token.rstrip("\n")
        vocab_dict[token] = index + 1

    vocab = torchtext.vocab.vocab(vocab_dict, min_freq=1).get_stoi()
    print(f"Loaded vocabulary")

    return vocab


def build_vocabulary(df, cols, tokenizer, min_freq=1, save_path=None):
    """
    Returns a Vocab object containing all the tokens appearing in the `cols` columns of the dataframe `df`
    """
    # vocab = collections.OrderedDict()
    counter = collections.Counter()

    # Append to vocab special tokens expected by the tokenizer
    for token in tokenizer.special_tokens_map.values():
        counter[token] = min_freq + 1

    sentences_in_cols = [v for col in cols for v in df[col].values]

    num_sentences = len(sentences_in_cols)

    print(f"Comments count: {num_sentences}")

    print(f"Creating vocabulary from comments...")
    percentage_printed = 0.0
    for index, sentence in enumerate(sentences_in_cols):
        percentage = round(index / num_sentences, 2)
        if percentage == 0.25 and percentage_printed == 0.0:
            print(f"25% vocabulary created")
            percentage_printed = 0.25
        elif percentage == 0.50 and percentage_printed == 0.25:
            print(f"50% vocabulary created")
            percentage_printed = 0.50
        elif percentage == 0.75 and percentage_printed == 0.5:
            print(f"75% vocabulary created")
            percentage_printed = 0.75
        counter.update(tokenizer._tokenize(sentence))

    v = torchtext.vocab.vocab(counter, min_freq=min_freq).get_stoi()
    print(f"Vocabulary created ({len(v)} words)")
    if save_path:
        save_vocabulary(save_path, v)
    return v, tokenizer


def save_vocabulary(save_path, vocab):
    """
    Save the [vocab] to the given file path 
    """
    index = 0
    print(f"Saving vocabulary to path: {save_path}")
    with open(save_path, "w", encoding="utf-8") as writer:
        for token, token_index in sorted(vocab.items(), key=lambda kv: kv[1]):
            if index != token_index:
                print(
                    f"Saving vocabulary to {save_path}: vocabulary indices are not consecutive."
                    " Please check that the vocabulary is not corrupted!"
                )
                index = token_index
            writer.write(token + "\n")
            index += 1
