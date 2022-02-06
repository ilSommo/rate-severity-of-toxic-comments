import argparse
import json
import os
from lightgbm import train

import pandas as pd
import torch
from torch import nn

from rate_severity_of_toxic_comments.dataset import build_dataloaders, build_dataset, load_dataframe
from rate_severity_of_toxic_comments.model import create_model
from rate_severity_of_toxic_comments.training import test_loop
from rate_severity_of_toxic_comments.utilities import process_config, validate_config

DEFAULT_CONFIG_FILE_PATH = "config/default.json"
LOCAL_CONFIG_FILE_PATH = "config/local.json"
BEST_MODELS_FILE_PATH = "config/best_models.json"

if __name__ == "__main__":
    args = {"batch_size": 32}

    default = open(DEFAULT_CONFIG_FILE_PATH)
    CONFIG = json.load(default)

    if os.path.exists(LOCAL_CONFIG_FILE_PATH):
        with open(LOCAL_CONFIG_FILE_PATH) as local:
            CONFIG.update(json.load(local))

    validate_config(CONFIG)

    models_file = open(BEST_MODELS_FILE_PATH)
    models = json.load(models_file)

    run_mode = CONFIG["options"]["run_mode"]
    eval_dataset_params = CONFIG["evaluation"]["dataset"]
    model_params = CONFIG[run_mode]
    
    df_test = load_dataframe(run_mode, eval_dataset_params, model_params=model_params)
    support_bag = process_config(df_test, CONFIG)

    test_data = build_dataset(df_test, CONFIG["evaluation"]["dataset"], model_params, support_bag["tokenizer"])

    batch_size = args["batch_size"]
    test_dl, = build_dataloaders([test_data], [batch_size])

    if eval_dataset_params["type"] == "scored":
        loss_fn = nn.MSELoss()
    elif eval_dataset_params["type"] == "pairwise":
        loss_fn = nn.MarginRankingLoss(margin=eval_dataset_params["loss_margin"])

    device = torch.device("cuda" if torch.cuda.is_available()
                        and CONFIG["options"]["use_gpu"] else "cpu")

    for model_details in models:
        #TODO: Use model_details["params"] instead of model_params
        model = create_model(run_mode, CONFIG["training"], model_params, support_bag)
        model.load_state_dict(torch.load(model_details["path"]))
        model.to(device)
        
        metrics = test_loop(test_dl, model, loss_fn, device, log_interval=1000, dataset_type=eval_dataset_params["type"], use_wandb=False)
        print(model_details["description"], metrics)
