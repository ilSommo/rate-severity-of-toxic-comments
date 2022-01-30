import re
import string

AVAILABLE_PREPROCESSING_PIPELINES = [
    'LOWER',
    'PUNCTUATION',
    'WHITESPACES',
    'COUNT_BAD_WORDS',
    'REPLACE_BAD_WORDS',
    'NUMBERS',
    'TRIPLE'
]

BAD_WORDS_FILE = 'res/bad_words.txt'
MASTER_WORD = 'shit'


def apply_preprocessing_pipelines(text, pipelines):
    counter = 0
    metric = 0
    for pipeline in pipelines:
        text, counter, metric = apply_preprocessing_pipeline(
            text, counter, metric, pipeline)
    return text, counter, metric


def apply_preprocessing_pipeline(text, counter, metric, pipeline):
    if(pipeline in AVAILABLE_PREPROCESSING_PIPELINES):
        additional_metric = 0
        if pipeline == 'LOWER':
            text, counter = _apply_lower_pipeline(text)
        elif pipeline == 'PUNCTUATION':
            text, counter = _apply_punctuation_pipeline(text)
        elif pipeline == 'WHITESPACES':
            text, counter = _apply_whitespaces_pipeline(text)
        elif pipeline == 'NUMBERS':
            text, counter = _apply_numbers_pipeline(text)
        elif pipeline == 'TRIPLE':
            text, counter = _apply_triple_pipeline(text)
        elif pipeline == 'REPLACE_BAD_WORDS':
            bad_words = import_bad_words(BAD_WORDS_FILE)
            text, counter = _apply_bad_word_pipeline(text)
        elif pipeline == 'COUNT_BAD_WORDS':
            bad_words = import_bad_words(BAD_WORDS_FILE)
            counter += count_bad_words(text, bad_words)
    return text, counter, metric + additional_metric


def _apply_lower_pipeline(text):
    additional_metric = len(re.findall('[A-Z]', text))
    text = text.lower()
    return text, additional_metric


def _apply_punctuation_pipeline(text):
    additional_metric = len(re.findall(
        '[%s]' % re.escape(string.punctuation), text))
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    return text, additional_metric


def _apply_whitespaces_pipeline(text):
    return text, 0


def _apply_numbers_pipeline(text):
    additional_metric = len(re.findall('[0-9]', text))
    text = re.sub('0', 'o', text)
    text = re.sub('1', 'i', text)
    text = re.sub('2', 'z', text)
    text = re.sub('3', 'e', text)
    text = re.sub('4', 'a', text)
    text = re.sub('5', 's', text)
    text = re.sub('7', 't', text)
    text = re.sub('8', 'ate', text)
    return text, additional_metric


def _apply_triple_pipeline(text):
    additional_metric = len(re.findall('(.)\\1{2,}', text))
    additional_metric += len(re.findall('s{2,}\\b', text))
    text = re.sub('(.)\\1{2,}', '\\1\\1', text)
    text = re.sub('s{2,}\\b', 's', text)
    return text, additional_metric


def _apply_bad_word_pipeline(text, bad_words):
    additional_metric = 0
    for word in bad_words:
        additional_metric += len(re.findall(word, text))
        text = re.sub(word, MASTER_WORD, text)
    return text, additional_metric


def import_bad_words(file):
    bad_words = []
    with open(file, 'r') as f:
        for line in f:
            word = line[:-1]
            word = word.replace('*', '\\*')
            word = word.replace('(', '\\(')
            word = word.replace(')', '\\)')
            bad_words.append(word)
    return bad_words


def count_bad_words(text, bad_words):
    counter = 0
    for word in bad_words:
        if word in text:
            counter += 1
    return counter


def preprocess_dataframe(df, cols, pipelines: list):
    if pipelines is None or len(pipelines) == 0:
        return df
    sentences_in_cols = [v for col in cols for v in df[col].values]
    num_sentences = len(sentences_in_cols)
    print(f"Dataset comments to preprocess:\t{num_sentences}")
    counter = 0
    for col in cols:
        for i in df.index:
            df.at[i, col], count, __ = apply_preprocessing_pipelines(
                df.at[i, col], pipelines)
            counter += count

    print(f"Dataframe preprocessed in {counter} occurrenrcies")
    return df
