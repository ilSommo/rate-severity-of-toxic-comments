import argparse
import json
import os

import pandas as pd
import torch

from rate_severity_of_toxic_comments.dataset import build_dataset, load_dataframe, split_dataset
from rate_severity_of_toxic_comments.training import run_training
from rate_severity_of_toxic_comments.utilities import process_config, validate_config

DEFAULT_CONFIG_FILE_PATH = "config/default.json"
LOCAL_CONFIG_FILE_PATH = "config/local.json"

if __name__ == "__main__":
    default = open(DEFAULT_CONFIG_FILE_PATH)
    CONFIG = json.load(default)

    if os.path.exists(LOCAL_CONFIG_FILE_PATH):
        with open(LOCAL_CONFIG_FILE_PATH) as local:
            CONFIG.update(json.load(local))

    validate_config(CONFIG)

    run_mode = CONFIG["options"]["run_mode"]
    df = load_dataframe(run_mode, CONFIG["training"], CONFIG[run_mode])

    support_bag = process_config(df, CONFIG)

    df_train, df_valid = split_dataset(df, CONFIG['options']['seed'])

    training_data = build_dataset(df_train, CONFIG["training"]["dataset"], 
            CONFIG[run_mode], support_bag["tokenizer"])
    val_data = build_dataset(df_valid, CONFIG["training"]["dataset"], 
            CONFIG[run_mode], support_bag["tokenizer"])

    model, loss_history = run_training(run_mode, training_data, val_data, 
            CONFIG["training"], CONFIG[run_mode], support_bag, CONFIG["options"]["seed"], 
            CONFIG["options"]["wandb"], CONFIG["options"]["use_gpu"], 
            verbose=True, log_interval=100)
