import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


class JigsawDataset(Dataset):
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
        inputs_more_toxic = self.tokenizer.encode_plus(
            more_toxic,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length'
        )
        inputs_less_toxic = self.tokenizer.encode_plus(
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


def build_datasets(dfs, tokenizer, max_length):
    return [JigsawDataset(df, tokenizer=tokenizer, max_length=max_length) for df in dfs]


def build_dataloaders(datasets, batch_sizes):
    return [DataLoader(ds, batch_size=batch_size, num_workers=2, shuffle=False, pin_memory=True)
            for ds, batch_size in zip(datasets, batch_sizes)]
