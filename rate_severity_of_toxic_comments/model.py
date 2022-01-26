from unicodedata import bidirectional
import torch
from torch import nn
from transformers import AutoModel

OUTPUT_CLASSES = 1


class PretrainedModel(nn.Module):
    def __init__(self, model_name, dropout, output_features):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        if dropout != None:
            self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(output_features, OUTPUT_CLASSES)

    def forward(self, ids, mask):
        out = self.model(input_ids=ids, attention_mask=mask,
                         output_hidden_states=False)
        out = self.drop(out.last_hidden_state)
        outputs = self.fc(out)
        return outputs


class RecurrentModel(nn.Module):
    def __init__(self, embedding_matrix, dropout, hidden_dim, architecture):
        super().__init__()
        _, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix), freeze=True)
        if architecture == 'LSTM':
            self.recurrent = nn.LSTM(embedding_dim, hidden_dim,
                                     batch_first=True, dropout=dropout)
        elif architecture == 'GRU':
            self.recurrent = nn.GRU(embedding_dim, hidden_dim,
                                    batch_first=True, dropout=dropout)
        elif architecture == 'BiDi':
            self.recurrent = nn.LSTM(embedding_dim, hidden_dim,
                                     batch_first=True, dropout=dropout, bidirectional=True)
        else:
            self.recurrent = nn.LSTM(embedding_dim, hidden_dim,
                                     batch_first=True, dropout=dropout)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, OUTPUT_CLASSES)

    def forward(self, ids, mask):
        embedded = self.embedding(ids)
        lengths = torch.count_nonzero(mask, dim=1)
        embedded = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.recurrent(embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        x = torch.mean(output, dim=-2)
        x = self.relu(x)
        return self.fc(x).squeeze()


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, OUTPUT_CLASSES)

    def forward(self, ids, mask):
        return self.fc(ids.to(torch.float32).mean(dim=0).unsqueeze(1))


def create_model(config):
    if config["run_mode"] == "test":
        return DummyModel()
    if config["run_mode"] == "recurrent":
        hidden_dim = config["output_features"]
        drop = config['dropout']
        architecture = config['architecture']
        return RecurrentModel(config["embedding_matrix"], drop, hidden_dim, architecture)
    elif config["run_mode"] == "pretrained":
        drop = config['dropout']
        output_features = config["output_features"]
        return PretrainedModel(config["model_name"], drop, output_features)
