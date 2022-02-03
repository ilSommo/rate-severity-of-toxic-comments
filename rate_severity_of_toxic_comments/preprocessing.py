import re
import string


AVAILABLE_PREPROCESSING_PIPELINES = [
    'LOWER',
    'PUNCTUATION',
    'WHITESPACES',
    'NUMBERS',
    'TRIPLE'
]


def apply_preprocessing_pipelines(text, pipelines):
    metric = 0
    for pipeline in pipelines:
        text, metric = apply_preprocessing_pipeline(text, metric, pipeline)
    return text, metric / len(pipelines)


def apply_preprocessing_pipeline(text, metric, pipeline):
    additional_metric = 0
    if(pipeline in AVAILABLE_PREPROCESSING_PIPELINES):
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
    return text, metric + additional_metric


def _apply_lower_pipeline(text):
    additional_metric = len(re.findall('[A-Z]', text)) / len(text)
    text = text.lower()
    return text, additional_metric


def _apply_punctuation_pipeline(text):
    additional_metric = len(re.findall('[%s]' % re.escape(string.punctuation), text)) / len(text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    return text, additional_metric


def _apply_whitespaces_pipeline(text):
    additional_metric = len(re.findall('  +', text)) / len(re.findall(' +', text))
    text = re.sub('  +', ' ', text)
    return text, additional_metric


def _apply_numbers_pipeline(text):
    additional_metric = len(re.findall('[0-9]', text)) / len(text)
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
    additional_metric_0 = len(re.findall('(.)\\1{2,}', text)) / len(re.findall('[\w]+', text))
    additional_metric_1 = len(re.findall('s{2,}\\b', text)) / len(re.findall('[\w]+', text))
    text = re.sub('(.)\\1{2,}', '\\1\\1', text)
    text = re.sub('s{2,}\\b', 's', text)
    return text, min((additional_metric_0 + additional_metric_1) / 2, 1.0)
