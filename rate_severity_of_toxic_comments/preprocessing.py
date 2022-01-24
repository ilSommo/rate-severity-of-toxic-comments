import re
import string

AVAILABLE_PREPROCESSING_PIPELINES = [
    'LOWER',
    'PUNCTUATION',
    'WHITESPACES',
]


def apply_preprocessing_pipelines(text, pipelines):
    for pipeline in pipelines:
        text = apply_preprocessing_pipeline(text, pipeline)
    return text


def apply_preprocessing_pipeline(text, pipeline):
    if(pipeline in AVAILABLE_PREPROCESSING_PIPELINES):
        if pipeline == 'LOWER':
            return _apply_lower_pipeline(text)
        elif pipeline == 'PUNCTUATION':
            return _apply_punctuation_pipeline(text)
        elif pipeline == 'WHITESPACES':
            return _apply_whitespaces_pipeline(text)


def _apply_punctuation_pipeline(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    return text


def _apply_lower_pipeline(text):
    text = text.lower()
    return text


def _apply_whitespaces_pipeline(text):
    return text
