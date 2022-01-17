import torch
from torch import nn

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
        out = self.drop(out[1])
        outputs = self.fc(out)
        return outputs

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, OUTPUT_CLASSES)
        
    def forward(self, ids, mask):
        return self.fc(ids.type(torch.float).mean(dim=0).unsqueeze(1))

def create_model(config):
    if config["run_mode"] == "test":
        return DummyModel()
    elif config["run_mode"] == "pretrained":
        return PretrainedModel(config["model_name"])