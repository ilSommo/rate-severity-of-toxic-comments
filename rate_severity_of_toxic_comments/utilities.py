__version__ = '0.1.0'
__author__ = 'Lorenzo Menghini, Martino Pulici, Alessandro Stockman, Luca Zucchini'

import math
import random
import os
import time
from typing import OrderedDict

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, pipeline

from verstack.stratified_continuous_split import scsplit
from sklearn.model_selection import train_test_split
from rate_severity_of_toxic_comments.embedding import build_embedding_matrix, load_embedding_model
from rate_severity_of_toxic_comments.preprocessing import AVAILABLE_PREPROCESSING_PIPELINES
from rate_severity_of_toxic_comments.tokenizer import NaiveTokenizer, build_vocab

_bad_words = []
_times = OrderedDict()
_time_deltas = OrderedDict()
_enable_time_tracking = None
_time_log_filename = None

def obfuscator(text):
    global _bad_words
    if len(_bad_words) == 0:
        with open("res/bad_words.txt") as file:
            bad_words = [l.rstrip() for l in file.readlines() if len(l) > 2]
            bad_words = [l for l in bad_words if len(l) > 2]

        _bad_words = list(sorted(bad_words, key=len, reverse=True))

    for word in _bad_words:
        visible = min(len(word) // 3, 3)
        censorship = word[0:visible] + \
            ((len(word) - visible * 2) * "*") + word[-visible:]
        text = text.replace(word, censorship)
    return text


def fix_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


def process_config(config):
    global _enable_time_tracking

    if not all([p not in AVAILABLE_PREPROCESSING_PIPELINES for p in config["preprocessing"]]):
        raise ValueError()

    try:
        config['output_features']
    except:
        raise ValueError()
    # TODO Add validation for other values

    """ 
    
    CHECKS CORRCT CONFIG FILE AND ELABORATION OF DATA DEPENDING ON CONFIGURATION
    
    """
    #TODO: Remove unused configs so that they are not uploaded to wandb

    if "time_tracking" in config:
        _enable_time_tracking = config["time_tracking"]

    if config["run_mode"] == "pretrained":
        config["tokenizer"] = AutoTokenizer.from_pretrained(
            config['model_name'])
    elif config['run_mode'] == 'recurrent':
        # Creates vocab file if it doens't exist
        if not os.path.isfile(config["vocab_file"]):
            open(config["vocab_file"], 'a').close()
        config["tokenizer"] = NaiveTokenizer(config["vocab_file"])

        # If vocab is empty, populate it with training sets
        if len(config["tokenizer"].get_vocab()) == 0:
            df = pd.read_csv(config["training_set"]["path"])
            vocab, _ = build_vocab(df, config["training_set"]["cols"], config["tokenizer"], save_path=config["vocab_file"])
            print(type(vocab))
            config["tokenizer"].set_vocab(vocab)
        embedding_model = load_embedding_model(config)
        embedding_matrix = build_embedding_matrix(embedding_model, config)
        config["embedding_matrix"] = embedding_matrix
    else:
        config["tokenizer"] = NaiveTokenizer(config["vocab_file"])
    return config


def split_dataset(dataframe: pd.DataFrame, seed):
    dataframe["label"] = dataframe["target"] * 10
    unique, counts = np.unique(np.floor(dataframe["label"]), return_counts=True)
    print(dict(zip(unique, counts)))
    return train_test_split(dataframe, stratify=np.floor(dataframe["label"]), random_state=seed)


def track_time(label, continuous_update=True):
    global _time_log_filename, _enable_time_tracking, _times, _time_deltas

    if _time_log_filename is None:
       _time_log_filename = "times-"+time.strftime("%Y%m%d-%H%M%S")+".txt"
    
    if _enable_time_tracking and label not in _time_deltas:
        if label not in _times:
            _times[label] = time.time()
        else:
            _time_deltas[label] = time.time() - _times[label]
            if continuous_update:
                with open(os.path.join("res", "logs", _time_log_filename), "a+") as log_file:
                    log_file.write(label + ": " + str(_time_deltas[label]) + "\n")
