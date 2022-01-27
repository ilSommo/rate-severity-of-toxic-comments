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
    for pipeline in pipelines:
        text, counter = apply_preprocessing_pipeline(text, pipeline)
    return text, counter


def apply_preprocessing_pipeline(text, pipeline):
    if(pipeline in AVAILABLE_PREPROCESSING_PIPELINES):
        if pipeline == 'LOWER':
            return apply_lower_pipeline(text)
        elif pipeline == 'PUNCTUATION':
            return apply_punctuation_pipeline(text)
        elif pipeline == 'WHITESPACES':
            return apply_whitespaces_pipeline(text)
        elif pipeline == 'COUNT_BAD_WORDS':
            bad_words = import_bad_words(BAD_WORDS_FILE)
            return text, count_bad_words(text, bad_words)
        elif pipeline == 'REPLACE_BAD_WORDS':
            bad_words = import_bad_words(BAD_WORDS_FILE)
            return apply_bad_word_pipeline(text), count_bad_words(text, MASTER_WORD)
        elif pipeline == 'NUMBERS':
            return apply_numbers_pipeline(text)
        elif pipeline == 'TRIPLE':
            return apply_triple_pipeline(text)


def apply_punctuation_pipeline(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    return text


def apply_lower_pipeline(text):
    text = text.lower()
    return text


def apply_whitespaces_pipeline(text):
    return text

def apply_numbers_pipeline(text):
    text = re.sub('0', 'o', text)
    text = re.sub('1', 'i', text)
    text = re.sub('2', 'z', text)
    text = re.sub('3', 'e', text)
    text = re.sub('4', 'a', text)
    text = re.sub('5', 's', text)
    text = re.sub('7', 't', text)
    text = re.sub('8', 'ate', text)
    return text

def apply_triple_pipeline(text):
    text = re.sub('(.)\\1{2,}', '\\1\\1', text)
    text = re.sub('s{2,}\\b', 's', text)
    return text

def apply_bad_word_pipeline(text, bad_words):
    for word in bad_words:
        text=re.sub(word, MASTER_WORD, text)
    return text

def import_bad_words(file):
    bad_words = []
    with open(file, 'r') as f:
        for line in f:
            word = line[:-1]
            word = word.replace('*', '\\*')
            word = word.replace('(', '\(')
            word = word.replace(')', '\)')
            bad_words.append(word)
    return bad_words

def count_bad_words(text, bad_words):
    counter = 0
    for word in bad_words:
        if word in text:
            counter += 1
    return counter