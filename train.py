import argparse
import json
import os

import pandas as pd
import torch

from rate_severity_of_toxic_comments.dataset import build_datasets
from rate_severity_of_toxic_comments.training import run_training
from rate_severity_of_toxic_comments.utilities import process_config


DEFAULT_CONFIG_FILE_PATH = "config/default.json"
LOCAL_CONFIG_FILE_PATH = "config/local.json"
TRAIN_TEST_SPLIT = 0.7

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    default = open(DEFAULT_CONFIG_FILE_PATH)
    CONFIG = json.load(default)

    if os.path.exists(LOCAL_CONFIG_FILE_PATH):
        with open(LOCAL_CONFIG_FILE_PATH) as local:
            CONFIG.update(json.load(local))

    CONFIG = process_config(CONFIG)

    df = pd.read_csv(args.dataset)
    data_size = len(df.index)
    data, = build_datasets([df], CONFIG, ["weighted"])
    train_size = int(data_size * TRAIN_TEST_SPLIT)
    training_data, val_data = torch.utils.data.random_split(data, [train_size, data_size - train_size])
    stats = run_training(training_data, val_data, log_interval=10, config=CONFIG, verbose=args.verbose)