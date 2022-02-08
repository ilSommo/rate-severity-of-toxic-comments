__version__ = '0.1.0'
__author__ = 'Lorenzo Menghini, Martino Pulici, Alessandro Stockman, Luca Zucchini'


import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torch.utils.data.dataloader import DataLoader

from rate_severity_of_toxic_comments.preprocessing import apply_preprocessing_pipelines


AVAILABLE_DATASET_TYPES = ['pairwise', 'scored', 'binarized']


class BinarizedDataset(Dataset):
    """
    Class containing a binarized dataset.

    Attributes
    ----------
    df : pandas.core.frame.DataFrame
        DataFrame containing the dataset.
    max_len : int
        Maximum length of the dataset.
    tokenizer : transformers.tokenization_utils.PreTrainedTokenizer
        Text tokenizer.
    text : list
        List of comments.
    metric : list
        List of preprocessing metrics of comments.
    target : list
        List of targets.

    Methods
    -------
    __init__(self, df, tokenizer, max_length)
        Initializes the dataset.
    __len__(self)
        Returns the length of the dataset.
    __getitem__(self, index)
        Returns a dataset item.

    """

    def __init__(self, df, tokenizer, max_length):
        """
        Initializes the dataset.

        Parameters
        ----------
        df : df
            DataFrame for the dataset.
        tokenizer : transformers.tokenization_utils.PreTrainedTokenizer
            Text tokenizer for the dataset.
        max_length : int
            Maximum length of the dataset.

        """
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.text = df['Sentence'].values
        self.metric = df['Sentence_metric'].values
        self.target = df['Percentage of toxicity binarized'].values

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns
        -------
        length : int
            Length of the dataset.

        """
        length = len(self.df)
        return length

    def __getitem__(self, index):
        """
        Returns a dataset item.

        Parameters
        ----------
        index : int
            Index of the item to return.

        Returns
        -------
        item : dict
            Item to return.

        """
        text = self.text[index]
        if self.max_len:
            inputs_text = self.tokenizer(text,truncation=True,add_special_tokens=True,max_length=self.max_len,padding='max_length')
        else:
            inputs_text = self.tokenizer(text,truncation=True,add_special_tokens=True,padding='longest')
        text_ids = inputs_text['input_ids']
        text_mask = inputs_text['attention_mask']
        text_metric = self.metric[index]
        target = self.target[index]
        item= {
            'text_ids': torch.tensor(text_ids, dtype=torch.long),
            'text_mask': torch.tensor(text_mask, dtype=torch.long),
            'text_metric': torch.tensor(text_metric, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.bool)}
        return item


class PairwiseDataset(Dataset):
    """
    Class containing a pairwise dataset.

    Attributes
    ----------
    df : pandas.core.frame.DataFrame
        DataFrame containing the dataset.
    max_len : int
        Maximum length of the dataset.
    tokenizer : transformers.tokenization_utils.PreTrainedTokenizer
        Text tokenizer.
    more_toxic : list
        List of more toxic comments.
    more_toxic_metric : list
        List of preprocessing metrics of more toxic comments.
    less_toxic : list
        List of less toxic comments.
    less_toxic_metric : list
        List of preprocessing metrics of less toxic comments.

    Methods
    -------
    __init__(self, df, tokenizer, max_length)
        Initializes the dataset.
    __len__(self)
        Returns the length of the dataset.
    __getitem__(self, index)
        Returns a dataset item.

    """

    def __init__(self, df, tokenizer, max_length):
        """
        Initializes the dataset.

        Parameters
        ----------
        df : df
            DataFrame for the dataset.
        tokenizer : transformers.tokenization_utils.PreTrainedTokenizer
            Text tokenizer for the dataset.
        max_length : int
            Maximum length of the dataset.

        """
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.more_toxic = df['more_toxic'].values
        self.more_toxic_metric = df['more_toxic_metric'].values
        self.less_toxic = df['less_toxic'].values
        self.less_toxic_metric = df['less_toxic_metric'].values

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns
        -------
        length : int
            Length of the dataset.

        """
        length = len(self.df)
        return length

    def __getitem__(self, index):
        """
        Returns a dataset item.

        Parameters
        ----------
        index : int
            Index of the item to return.

        Returns
        -------
        item : dict
            Item to return.

        """
        more_toxic = self.more_toxic[index]
        less_toxic = self.less_toxic[index]
        if self.max_len:
            inputs_more_toxic = self.tokenizer(more_toxic,truncation=True,add_special_tokens=True,max_length=self.max_len,padding='max_length')
            inputs_less_toxic = self.tokenizer(less_toxic,truncation=True,add_special_tokens=True,max_length=self.max_len,padding='max_length')
        else:
            inputs_more_toxic = self.tokenizer(more_toxic,truncation=True,add_special_tokens=True,padding='longest')
            inputs_less_toxic = self.tokenizer(less_toxic,truncation=True,add_special_tokens=True,padding='longest')
        more_toxic_ids = inputs_more_toxic['input_ids']
        more_toxic_mask = inputs_more_toxic['attention_mask']
        more_toxic_metric = self.more_toxic_metric[index]
        less_toxic_ids = inputs_less_toxic['input_ids']
        less_toxic_mask = inputs_less_toxic['attention_mask']
        less_toxic_metric = self.less_toxic_metric[index]
        target = 1
        item =  {
            'more_toxic_ids': torch.tensor(more_toxic_ids, dtype=torch.long),
            'more_toxic_mask': torch.tensor(more_toxic_mask, dtype=torch.long),
            'more_toxic_metric': torch.tensor(more_toxic_metric, dtype=torch.float32),
            'less_toxic_ids': torch.tensor(less_toxic_ids, dtype=torch.long),
            'less_toxic_mask': torch.tensor(less_toxic_mask, dtype=torch.long),
            'less_toxic_metric': torch.tensor(less_toxic_metric, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.long)
        }
        return item
    

class ScoredDataset(Dataset):
    """
    Class containing a binarized dataset.

    Attributes
    ----------
    df : pandas.core.frame.DataFrame
        DataFrame containing the dataset.
    max_len : int
        Maximum length of the dataset.
    tokenizer : transformers.tokenization_utils.PreTrainedTokenizer
        Text tokenizer.
    text : list
        List of comments.
    sample_weight : list
        List of sample weights.
    preprocessing_metric : list
        List of preprocessing metrics of comments.
    target : list
        List of targets.

    Methods
    -------
    __init__(self, df, tokenizer, max_length)
        Initializes the dataset.
    __len__(self)
        Returns the length of the dataset.
    __getitem__(self, index)
        Returns a dataset item.

    """

    def __init__(self, df, tokenizer, max_length):
        """
        Initializes the dataset.

        Parameters
        ----------
        df : df
            DataFrame for the dataset.
        tokenizer : transformers.tokenization_utils.PreTrainedTokenizer
            Text tokenizer for the dataset.
        max_length : int
            Maximum length of the dataset.

        """
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.text = df['comment_text'].values
        self.sample_weight = df['sample_weight'].values
        self.preprocessing_metric = df['comment_text_metric'].values
        self.target = df['target'].values

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns
        -------
        length : int
            Length of the dataset.

        """
        length = len(self.df)
        return length

    def __getitem__(self, index):
        """
        Returns a dataset item.

        Parameters
        ----------
        index : int
            Index of the item to return.

        Returns
        -------
        item : dict
            Item to return.

        """
        text = self.text[index]
        if self.max_len:
            inputs = self.tokenizer(
                text,
                truncation=True,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length'
            )
        else:
            inputs = self.tokenizer(
                text,
                truncation=True,
                add_special_tokens=True,
                padding='longest'
            )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        preprocessing_metric = self.preprocessing_metric[index]
        target = self.target[index]
        item= {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.float32),
            'preprocessing_metric': torch.tensor(preprocessing_metric, dtype=torch.float32)
        }
        return item


def build_dataloaders(datasets, batch_sizes):
    """
    Builds the dataloader.

    Parameters
    ----------
    datasets : list
        List of datasets.
    batch_sizes : list
        List of batch sizes.

    Returns
    -------
    data_loaders : list
        List of dataloaders.

    """
    data_loaders = []
    for ds, batch_size in zip(datasets, batch_sizes):
        try:
            data_loaders.append(DataLoader(ds, batch_size=batch_size, num_workers=2, sampler=WeightedRandomSampler(ds.sample_weight, len(ds))))
        except:
            data_loaders.append(DataLoader(ds, batch_size=batch_size, num_workers=2, shuffle=False, pin_memory=True))
    return data_loaders


def build_dataset(df, dataset_params, model_params, tokenizer):
    """
    Builds the dataset.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        DataFrame containing the dataset.
    dataset_params : dict
        Dataset parameters.
    model_params : dict
        Model parameters.
    tokenizer : transformers.tokenization_utils.PreTrainedTokenizer
        Text tokenizer.

    Returns
    -------
    dataset : torch.utils.data.dataset.Dataset
        Dataset to return.

    """
    if dataset_params['type'] == 'pairwise':
        dataset =  PairwiseDataset(df, tokenizer=tokenizer, max_length=model_params['max_length'])
    elif dataset_params['type'] == 'scored':
        dataset =  ScoredDataset(df, tokenizer=tokenizer, max_length=model_params['max_length'])
    elif dataset_params['type'] == 'binarized':
        dataset =  BinarizedDataset(df, tokenizer=tokenizer, max_length=model_params['max_length'])
    return dataset


def get_sample_weights(df, target_col_name, bins=100):
    """
    Returns the sample weights.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        DataFrame containing the dataset.
    target_col_name : str
        Name of the target column.
    bins : int, default 100
        Number of bins.

    Returns
    -------
    binned_targets : pandas.core.series.Series
        Sample weights.

    """        
    binned_targets = pd.cut(df[target_col_name], bins)
    sample_weights = binned_targets.map(1 / binned_targets.value_counts())
    return sample_weights


def load_dataframe(run_mode, dataset_params, model_params):
    """
    Loads the dataframe.

    Parameters
    ----------
    run_mode : str
        Run mode.
    dataset_params : dict
        Dataset parameters.
    model_params : dict
        Model parameters.

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Loaded dataframe.

    """
    base_train_file_path = dataset_params['path']
    cols = dataset_params['cols']
    data_frame_to_load = base_train_file_path
    pipelines = []
    if run_mode == 'recurrent':
        pipelines = model_params['preprocessing']
        pipelines.sort()
        vocab_file = model_params['vocab_file']
        if pipelines is None or len(pipelines) == 0:
            print(f'Loaded base dataframe from {base_train_file_path}\n')
            df = pd.read_csv(base_train_file_path)
            for col in cols:
                df[col+'_metric'] = 0

            if dataset_params['weighted_sampling']:
                df['sample_weight'] = get_sample_weights(df, dataset_params['target_col'])
            return df
        data_frame_to_load = base_train_file_path[:-4]
        vocab_to_load = vocab_file[:-4]
        for pipeline in pipelines:
            data_frame_to_load += '_' + pipeline
            vocab_to_load += '_' + pipeline
        data_frame_to_load += '.csv'
        vocab_to_load += '.txt'
        print(f'Trying to load dataframe from {data_frame_to_load}')
        print(f'New vocab file path {vocab_to_load}')
        model_params['vocab_file'] = vocab_to_load
    else:
        df = pd.read_csv(data_frame_to_load)
        if dataset_params['weighted_sampling']:
            df['sample_weight'] = get_sample_weights(df, dataset_params['target_col'])
            for col in cols:
                df[col+'_metric'] = 0
        return df

    if os.path.exists(data_frame_to_load):
        print(f'Loading preprocessed dataframe from {data_frame_to_load}\n')
        df = pd.read_csv(data_frame_to_load)
        if dataset_params['weighted_sampling']:
            df['sample_weight'] = get_sample_weights(df, dataset_params['target_col'])
        return df
    else:
        df = pd.read_csv(base_train_file_path)
    sentences_in_cols = [v for col in cols for v in df[col].values]
    num_sentences = len(sentences_in_cols)
    print(f'Dataset comments to preprocess: {num_sentences}')
    print(f'Pipelines to apply: {pipelines}')
    for col in cols:
        for i in df.index:
            if i == int(num_sentences / 4):
                print(f'25% comments preprocessed')
            elif i == int(num_sentences / 2):
                print(f'50% comments preprocessed')
            elif i == int(num_sentences / 1.5):
                print(f'75% comments preprocessed')
            df.at[i, col], df.at[i, col+'_metric'] = apply_preprocessing_pipelines(df.at[i, col], pipelines)
    print(f'Dataframe preprocessed\n')
    df.to_csv(data_frame_to_load)
    if dataset_params['weighted_sampling']:
        df['sample_weight'] = get_sample_weights(df, dataset_params['target_col'])
    return df


def split_dataset(dataframe: pd.DataFrame, target_col_name, seed):
    """
    Splits the dataset.

    Parameters
    ----------
    dataframe : pandas.core.frame.DataFrame
        DataFrame containing the dataset.
    target_col_name : str
        Name of the target column.
    seed : int
        Seed for random state.

    Returns
    -------
    splitting : list
        List of dataset splittings.

    """    
    dataframe['label'] = dataframe[target_col_name] * 10
    unique, counts = np.unique(np.floor(dataframe['label']), return_counts=True)
    print(dict(zip(unique, counts)))
    splitting = train_test_split(dataframe, stratify=np.floor(dataframe['label']), random_state=seed)
    return splitting
