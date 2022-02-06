import argparse
import json
import os

import pandas as pd
import torch

from rate_severity_of_toxic_comments.model import create_model
from rate_severity_of_toxic_comments.utilities import validate_config, process_config

DEFAULT_CONFIG_FILE_PATH = "config/default.json"
LOCAL_CONFIG_FILE_PATH = "config/local.json"
BEST_MODELS_FILE_PATH = "config/best_models.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file')
    args = parser.parse_args()

    default = open(DEFAULT_CONFIG_FILE_PATH)
    CONFIG = json.load(default)

    if os.path.exists(LOCAL_CONFIG_FILE_PATH):
        with open(LOCAL_CONFIG_FILE_PATH) as local:
            CONFIG.update(json.load(local))

    validate_config(CONFIG)
    support_bag = process_config(pd.DataFrame(), CONFIG)

    run_mode = CONFIG["options"]["run_mode"]

    device = torch.device("cuda" if torch.cuda.is_available() and CONFIG["options"]["use_gpu"] else "cpu")

    model = create_model(run_mode, CONFIG["training"], CONFIG[run_mode], support_bag)
    model.load_state_dict(torch.load(args.model_file))
    model.to(device)

    query = True
    while query:
        query = input("Type comment:")

        inputs = support_bag['tokenizer'](
            query,
            truncation=True,
            add_special_tokens=True,
            max_length=128,
            padding='max_length'
        )
            
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        score = model(ids.unsqueeze(dim=0), mask.unsqueeze(dim=0), torch.tensor([0]))
        print("Score:", score.item())
