import argparse
import json
import os

import pandas as pd
import torch

from rate_severity_of_toxic_comments.dataset import build_datasets, load_dataframe
from rate_severity_of_toxic_comments.training import run_training
from rate_severity_of_toxic_comments.utilities import process_config, split_dataset


DEFAULT_CONFIG_FILE_PATH = "config/default.json"
LOCAL_CONFIG_FILE_PATH = "config/local.json"
TRAIN_TEST_SPLIT = 0.7

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    default = open(DEFAULT_CONFIG_FILE_PATH)
    CONFIG = json.load(default)

    if os.path.exists(LOCAL_CONFIG_FILE_PATH):
        with open(LOCAL_CONFIG_FILE_PATH) as local:
            CONFIG.update(json.load(local))

    if CONFIG["run_mode"] == "pretrained":
        CONFIG["preprocessing"] = []

    df = load_dataframe(CONFIG)

    CONFIG = process_config(df, CONFIG)

    df = pd.read_csv(CONFIG["training_set"]["path"])
    df_train, df_valid = split_dataset(df, CONFIG['seed'])

    training_data, val_data = build_datasets([df_train, df_valid], CONFIG, [
                                             CONFIG["training_set"]["type"], CONFIG["training_set"]["type"]])
    stats = run_training(training_data, val_data,
                         log_interval=10, config=CONFIG, verbose=args.verbose)
