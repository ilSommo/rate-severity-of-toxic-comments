from pandas import DataFrame
import re
import string

AVAILABLE_PREPROCESSING_PIPELINES = [
    'LOWER',
    'PUNCTUATION',
    'WHITESPACES',
]


def apply_preprocessing_pipelines(dataframe, pipelines):
    for pipeline in pipelines:
        dataframe = apply_preprocessing_pipeline(dataframe, pipeline)
    return dataframe


def apply_preprocessing_pipeline(dataframe, pipeline) -> DataFrame:
    if(pipeline in AVAILABLE_PREPROCESSING_PIPELINES):
        if pipeline == 'LOWER':
            return _apply_lower_pipeline(dataframe)
        elif pipeline == 'PUNCTUATION':
            return _apply_punctuation_pipeline(dataframe)
        elif pipeline == 'WHITESPACES':
            return _apply_whitespaces_pipeline(dataframe)


def _apply_punctuation_pipeline(dataframe: DataFrame):
    dataframe["comment_text"] = dataframe["comment_text"].apply(
        lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))
    return dataframe


def _apply_lower_pipeline(dataframe: DataFrame):
    dataframe["comment_text"] = dataframe["comment_text"].apply(
        lambda val: val.lower())
    return dataframe


def _apply_whitespaces_pipeline(dataframe):
    return dataframe
