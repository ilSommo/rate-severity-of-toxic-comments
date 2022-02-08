__version__ = '1.0.0-rc.1'
__author__ = 'Lorenzo Menghini, Martino Pulici, Alessandro Stockman, Luca Zucchini'


import string

import re


AVAILABLE_PREPROCESSING_PIPELINES = [
    'LOWER',
    'PUNCTUATION',
    'WHITESPACES',
    'NUMBERS',
    'TRIPLE'
]


def apply_preprocessing_pipelines(text, pipelines):
    """
    Applies the preprocesssing pipelines to a text.

    Parameters
    ----------
    text : str
        Text to preprocess.
    pipelines : list
        Pipelines to execute.

    Returns
    -------
    text : str
        Preprocessed text.
    metric : float
        Preprocessing amount metric.

    """
    metric = 0
    for pipeline in pipelines:
        text, metric = apply_preprocessing_pipeline(text, metric, pipeline)
    metric = metric / len(pipelines)
    return text, metric


def apply_preprocessing_pipeline(text, metric, pipeline):
    """
    Applies a preprocesssing pipeline to a text.

    Parameters
    ----------
    text : str
        Text to preprocess.
    metric : float
        Preprocessing amount metric
    pipeline : str
        Pipeline to execute.

    Returns
    -------
    text : str
        Preprocessed text.
    metric : float
        Preprocessing amount metric.

    """
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
    metric += additional_metric
    return text, metric


def _apply_lower_pipeline(text):
    """
    Applies a the lowercasing pipeline to a text.

    Parameters
    ----------
    text : str
        Text to preprocess.

    Returns
    -------
    text : str
        Preprocessed text.
    additional_metric : float
        Preprocessing amount metric.

    """
    additional_metric = len(re.findall('[A-Z]', text)) / max(len(text), 1)
    text = text.lower()
    return text, additional_metric


def _apply_punctuation_pipeline(text):
    """
    Applies a the punctuation pipeline to a text.

    Parameters
    ----------
    text : str
        Text to preprocess.

    Returns
    -------
    text : str
        Preprocessed text.
    additional_metric : float
        Preprocessing amount metric.

    """
    additional_metric = len(re.findall('[%s]' % re.escape(
        string.punctuation), text)) / max(len(text), 1)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    return text, additional_metric


def _apply_whitespaces_pipeline(text):
    """
    Applies a the whitespaces removal pipeline to a text.

    Parameters
    ----------
    text : str
        Text to preprocess.

    Returns
    -------
    text : str
        Preprocessed text.
    additional_metric : float
        Preprocessing amount metric.

    """
    additional_metric = len(re.findall('  +', text)) / \
        max(len(re.findall(' +', text)), 1)
    text = re.sub('  +', ' ', text)
    return text, additional_metric


def _apply_numbers_pipeline(text):
    """
    Applies a the number substitution pipeline to a text.

    Parameters
    ----------
    text : str
        Text to preprocess.

    Returns
    -------
    text : str
        Preprocessed text.
    additional_metric : float
        Preprocessing amount metric.

    """
    additional_metric = len(re.findall('[0-9]', text)) / max(len(text), 1)
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
    """
    Applies a the triple letter removal pipeline to a text.

    Parameters
    ----------
    text : str
        Text to preprocess.

    Returns
    -------
    text : str
        Preprocessed text.
    additional_metric : float
        Preprocessing amount metric.

    """
    additional_metric_0 = len(re.findall(
        '(.)\\1{2,}', text)) / max(len(re.findall('[\\w]+', text)), 1)
    additional_metric_1 = len(re.findall(
        's{2,}\\b', text)) / max(len(re.findall('[\\w]+', text)), 1)
    text = re.sub('(.)\\1{2,}', '\\1\\1', text)
    text = re.sub('s{2,}\\b', 's', text)
    additional_metric = min(
        (additional_metric_0 + additional_metric_1) / 2, 1.0)
    return text, additional_metric
