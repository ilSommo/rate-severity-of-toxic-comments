import torch
from torch.nn import LSTM, GRU, Embedding, Module, Dropout, ReLU, Linear, Sigmoid
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel

OUTPUT_CLASSES = 1
AVAILABLE_ARCHITECTURES = ["LSTM", "GRU", "BiDi"]

class PretrainedModel(Module):
    def __init__(self, model_name, dropout, output_features):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.sig = Sigmoid()
        self.fc = Linear(output_features, OUTPUT_CLASSES)

    def forward(self, ids, mask, preprocessing_metric):
        out = self.model(input_ids=ids, attention_mask=mask,
                         output_hidden_states=False)
        cls_token = out.last_hidden_state[:, 0, :]
        outputs = self.fc(cls_token)

        return self.sig(outputs).squeeze()


class RecurrentModel(Module):
    def __init__(self, embedding_matrix, dropout, hidden_dim, architecture, preprocessing_metric):
        super().__init__()
        self.preprocessing_metric = preprocessing_metric
        _, embedding_dim = embedding_matrix.shape
        self.embedding = Embedding.from_pretrained(
            torch.tensor(embedding_matrix))
        if architecture == 'LSTM':
            self.recurrent = LSTM(embedding_dim, hidden_dim,
                                  batch_first=True)
        elif architecture == 'GRU':
            self.recurrent = GRU(embedding_dim, hidden_dim,
                                 batch_first=True)
        elif architecture == 'BiDi':
            self.recurrent = LSTM(embedding_dim, hidden_dim,
                                  batch_first=True, bidirectional=True)
            hidden_dim = hidden_dim * 2

        self.drop = Dropout(p=dropout)
        self.relu = ReLU()
        self.sig = Sigmoid()
        self.fc = Linear(hidden_dim+int(preprocessing_metric), OUTPUT_CLASSES)

    def forward(self, ids, mask, preprocessing_metric):
        embedded = self.embedding(ids)
        lengths = torch.count_nonzero(mask, dim=1)
        batch_lengths = lengths.to('cpu')
        embedded = pack_padded_sequence(
            embedded, batch_lengths, batch_first=True, enforce_sorted=False)
        rec_out, _ = self.recurrent(embedded)
        rec_out = pad_packed_sequence(
            rec_out, batch_first=True)
        drop_out = self.drop(rec_out[0])
        x = torch.mean(drop_out, dim=-2)
        if self.preprocessing_metric:
            x = torch.cat((x, preprocessing_metric[:,None]),dim=1)
        x = self.relu(x)
        x = self.fc(x)
        return self.sig(x).squeeze()


class DummyModel(Module):
    def __init__(self):
        super().__init__()
        self.fc = Linear(1, OUTPUT_CLASSES)

    def forward(self, ids, mask):
        return self.fc(ids.to(torch.float32).mean(dim=0).unsqueeze(1))


def create_model(run_mode, train_params, model_params, support_bag):
    if run_mode == "debug":
        return DummyModel()
    if run_mode == "recurrent":
        return RecurrentModel(support_bag["embedding_matrix"], train_params['dropout'], model_params['hidden_dim'], model_params['architecture'], model_params['preprocessing_metric'])
    elif run_mode == "pretrained":
        return PretrainedModel(model_params["model_name"], train_params['dropout'], model_params["output_features"])
