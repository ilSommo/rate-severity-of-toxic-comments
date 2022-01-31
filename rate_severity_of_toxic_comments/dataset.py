import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import pandas as pd
import numpy as np
from rate_severity_of_toxic_comments.preprocessing import apply_preprocessing_pipelines
from sklearn.model_selection import train_test_split
import os


class PairwiseDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.more_toxic = df['more_toxic'].values
        self.less_toxic = df['less_toxic'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        more_toxic = self.more_toxic[index]
        less_toxic = self.less_toxic[index]
        inputs_more_toxic = self.tokenizer(
            more_toxic,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length'
        )
        inputs_less_toxic = self.tokenizer(
            less_toxic,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length'
        )
        target = 1

        more_toxic_ids = inputs_more_toxic['input_ids']
        more_toxic_mask = inputs_more_toxic['attention_mask']

        less_toxic_ids = inputs_less_toxic['input_ids']
        less_toxic_mask = inputs_less_toxic['attention_mask']

        return {
            'more_toxic_ids': torch.tensor(more_toxic_ids, dtype=torch.long),
            'more_toxic_mask': torch.tensor(more_toxic_mask, dtype=torch.long),
            'less_toxic_ids': torch.tensor(less_toxic_ids, dtype=torch.long),
            'less_toxic_mask': torch.tensor(less_toxic_mask, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long)
        }


class WeightedDataset(Dataset):
    def __init__(self, df, weights, tokenizer, max_length):
        self.df = df
        self.weights = weights
        self.text = df["comment_text"].values
        self.target = df["target"].values
        self.tokenizer = tokenizer
        self.max_len = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.text[index]
        inputs = self.tokenizer(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length'
        )
        target = self.target[index]

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.float32),
        }


def build_datasets(dfs, config, dataset_types):
    dts = []
    for df, ds_type in zip(dfs, dataset_types):
        if ds_type == "pairwise":
            dts.append(PairwiseDataset(
                df, tokenizer=config["tokenizer"], max_length=config["max_length"]))
        elif ds_type == "weighted":
            dts.append(WeightedDataset(
                df, [], tokenizer=config["tokenizer"], max_length=config["max_length"]))
    return dts


def build_dataloaders(datasets, batch_sizes):
    return [DataLoader(ds, batch_size=batch_size, num_workers=2, shuffle=False, pin_memory=True)
            for ds, batch_size in zip(datasets, batch_sizes)]


def split_dataset(dataframe: pd.DataFrame, seed):

    dataframe["label"] = dataframe["target"] * 10

    unique, counts = np.unique(
        np.floor(dataframe["label"]), return_counts=True)
    print(dict(zip(unique, counts)))
    return train_test_split(dataframe, stratify=np.floor(dataframe["label"]), random_state=seed)


def load_dataframe(config):
    pipelines = config["preprocessing"]
    pipelines.sort()
    base_train_file_path = config["training_set"]["path"]
    vocab_file = config["vocab_file"]
    if pipelines is None or len(pipelines) == 0:
        print(f'Loaded base dataframe from {base_train_file_path}')
        return pd.read_csv(base_train_file_path)

    data_frame_to_load = base_train_file_path[:-3]
    vocab_to_load = vocab_file[:-3]

    for pipeline in pipelines:
        data_frame_to_load += '_' + pipeline
        vocab_to_load += '_' + pipeline
    data_frame_to_load += '.csv'
    vocab_to_load += '.txt'

    print(f'Trying to load dataframe from {data_frame_to_load}')
    print(f'New vocab file path {vocab_to_load}')
    config["vocab_file"] = vocab_to_load

    if os.path.exists(data_frame_to_load):
        print(f'Loading preprocessed dataframe from {data_frame_to_load}')
        df = pd.read_csv(data_frame_to_load)
        return df
    else:
        df = pd.read_csv(base_train_file_path)

    cols = config["training_set"]["cols"]
    sentences_in_cols = [v for col in cols for v in df[col].values]
    num_sentences = len(sentences_in_cols)
    print(f"Dataset comments to preprocess: {num_sentences}")
    print(f"Pipelines to apply: {pipelines}")

    counter = 0
    for col in cols:
        for i in df.index:
            if i == int(num_sentences / 4):
                print(f"25% comments preprocessed")
            elif i == int(num_sentences / 2):
                print(f"50% comments preprocessed")
            elif i == int(num_sentences / 1.5):
                print(f"75% comments preprocessed")
            df.at[i, col], bad_words_count, count = apply_preprocessing_pipelines(
                df.at[i, col], pipelines)
            counter += count

    print(f"Dataframe preprocessed in {counter} occurrencies")
    df.to_csv(data_frame_to_load)
    return df
