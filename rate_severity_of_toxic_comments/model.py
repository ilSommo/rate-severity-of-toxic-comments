import torch
from torch import nn
from transformers import AutoModel

OUTPUT_CLASSES = 1

class PretrainedModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.2)
        #TODO: Is pooler.dense.out_features guaranteed to exist?
        self.fc = nn.Linear(self.model.pooler.dense.out_features, OUTPUT_CLASSES) 
        
    def forward(self, ids, mask):        
        out = self.model(input_ids=ids,attention_mask=mask,
                         output_hidden_states=False)
        out = self.drop(out.last_hidden_state)
        outputs = self.fc(out)
        return outputs

class RecurrentModel(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim):
        super().__init__()
        _, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix), freeze=True)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, OUTPUT_CLASSES)

    def forward(self, ids, mask):
        embedded = self.embedding(ids)
        lengths = torch.count_nonzero(mask, dim=1)
        embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        x = torch.mean(output, dim=-2)
        x = self.relu(x)
        return self.fc(x)

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
        hidden_dim = 128 # config["hidden_dim"]
        return RecurrentModel(config["embedding_matrix"], hidden_dim)
    elif config["run_mode"] == "pretrained":
        return PretrainedModel(config["model_name"])
