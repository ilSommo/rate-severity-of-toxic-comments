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
    bad_words_counter = 0
    metric = 0
    bad_words = import_bad_words(BAD_WORDS_FILE)
    for pipeline in pipelines:
        text, bad_words_counter, metric = apply_preprocessing_pipeline(
            text, bad_words_counter, metric, pipeline, bad_words)
    return text, bad_words_counter, metric


def apply_preprocessing_pipeline(text, bad_words_counter, metric, pipeline, bad_words):
    if(pipeline in AVAILABLE_PREPROCESSING_PIPELINES):
        additional_metric = 0
        if pipeline == 'LOWER':
            text, additional_metric = _apply_lower_pipeline(text)
        elif pipeline == 'PUNCTUATION':
            text, additional_metric = _apply_punctuation_pipeline(text)
        elif pipeline == 'WHITESPACES':
            text, additional_metric = _apply_whitespaces_pipeline(text)
        elif pipeline == 'NUMBERS':
            text, additional_metric = _apply_numbers_pipeline(text)
        elif pipeline == 'TRIPLE':
            text, additional_metric = _apply_triple_pipeline(text)
        elif pipeline == 'REPLACE_BAD_WORDS':
            text, additional_metric = _apply_bad_word_pipeline(text, bad_words)
        elif pipeline == 'COUNT_BAD_WORDS':
            bad_words_counter += count_bad_words(text, bad_words)
    return text, bad_words_counter, metric + additional_metric


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
