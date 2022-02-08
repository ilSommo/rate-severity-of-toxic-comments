__version__ = '0.1.0'
__author__ = 'Lorenzo Menghini, Martino Pulici, Alessandro Stockman, Luca Zucchini'

import random
import json
import os

import numpy as np
import torch
from transformers import AutoTokenizer

from rate_severity_of_toxic_comments.preprocessing import AVAILABLE_PREPROCESSING_PIPELINES
from rate_severity_of_toxic_comments.model import AVAILABLE_ARCHITECTURES
from rate_severity_of_toxic_comments.embedding import AVAILABLE_EMBEDDINGS
from rate_severity_of_toxic_comments.dataset import AVAILABLE_DATASET_TYPES
from rate_severity_of_toxic_comments.tokenizer import NaiveTokenizer, create_recurrent_model_tokenizer

_bad_words = []
AVAILABLE_MODES = ["recurrent", "pretrained", "debug"]


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
    """Initialize seed value"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def validate_config(config):
    """
    Validates the config file and raise a ValueError if some value is missing or not supported\n
    - Checks for the available preprocessing pipeline to apply\n
    - Checks for run mode\n
    - Checks for dataset type\n
    """
    # Check for value correctness
    if config["options"]["run_mode"] not in AVAILABLE_MODES:
        raise ValueError("Invalid configuration! Run Mode not supported")
    elif config["recurrent"]["architecture"] not in AVAILABLE_ARCHITECTURES:
        raise ValueError(
            "Invalid configuration! Recurrent architecture not supported")
    elif not all([p in AVAILABLE_PREPROCESSING_PIPELINES for p in config["recurrent"]["preprocessing"]]):
        raise ValueError(
            "Invalid configuration! Preprocessing pipeline not supported")
    elif (config["recurrent"]["embedding_type"], config["recurrent"]["embedding_dimension"]) not in AVAILABLE_EMBEDDINGS:
        raise ValueError(
            "Invalid configuration! Embedding type and dimension not supported")
    elif config["training"]["dataset"]["type"] not in AVAILABLE_DATASET_TYPES:
        raise ValueError("Invalid configuration! Dataset type not supported")
    elif config["evaluation"]["dataset"]["type"] not in AVAILABLE_DATASET_TYPES:
        raise ValueError("Invalid configuration! Dataset type not supported")

    # Check if mandatory attributes are present
    if not all(item in config["options"].keys() for item in
               ["run_mode", "use_gpu", "wandb"]):
        raise ValueError(
            "Invalid configuration! Value missing under 'options'")
    elif not all(item in config["pretrained"].keys() for item in
                 ["model_name", "output_features"]):
        raise ValueError(
            "Invalid configuration! Value missing under 'pretrained'")
    elif not all(item in config["recurrent"].keys() for item in
                 ["architecture", "preprocessing", "vocab_file", "embedding_type", "embedding_dimension", "hidden_dim"]):
        raise ValueError(
            "Invalid configuration! Value missing under 'recurrent'")
    elif not all(item in config["training"].keys() for item in
                 ["epochs", "train_batch_size", "valid_batch_size", "learning_rate", "dataset"]):
        raise ValueError(
            "Invalid configuration! Value missing under 'training'")
    elif not all(item in config["evaluation"].keys() for item in
                 ["dataset"]):
        raise ValueError(
            "Invalid configuration! Value missing under 'evaluation'")
    elif config["training"]["dataset"]["type"] == "pairwise" and not "loss_margin" in config["training"]["dataset"]:
        raise ValueError(
            "Pairwise dataset requires a margin attribute!")
    elif config["evaluation"]["dataset"]["type"] == "pairwise" and not "loss_margin" in config["evaluation"]["dataset"]:
        raise ValueError(
            "Pairwise dataset requires a margin attribute!")


def process_config(df, config):
    """
    Given the config file depending on the config["options"]["run_mode"] add entries to the config dictionary, in particular\n
    - "pretrained" -> Loads the AutoTokenizer depending on the pretrained model name\n
    - "recurrent" -> Loads the NaiveTokenizer and the embedding matrix generated using pretrained embeddings\n
    - "debug" -> For debug purposes\n
    """
    support_bag = {}
    if config["options"]["run_mode"] == "pretrained":
        support_bag["tokenizer"] = AutoTokenizer.from_pretrained(
            config["pretrained"]["model_name"])

    elif config["options"]["run_mode"] == "recurrent":
        tokenizer, embedding_matrix = create_recurrent_model_tokenizer(
            config, df)
        support_bag["tokenizer"] = tokenizer
        support_bag["embedding_matrix"] = embedding_matrix

    elif config["options"]["run_mode"] == "debug":
        support_bag["tokenizer"] = NaiveTokenizer()
        support_bag["tokenizer"].set_vocab({"[UNK]": 0, "[PAD]": 1})

    # If requested, fixes random seed
    if config["options"]["seed"]:
        fix_random_seed(config["options"]["seed"])

    return support_bag

def parse_config(default_filepath, local_filepath=None):
    default = open(default_filepath)
    CONFIG = json.load(default)

    if os.path.exists(local_filepath):
        with open(local_filepath) as local:
            import collections.abc

            def deep_update(d, u):
                for k, v in u.items():
                    if isinstance(v, collections.abc.Mapping):
                        d[k] = deep_update(d.get(k, {}), v)
                    else:
                        d[k] = v
                return d

            CONFIG = deep_update(CONFIG, json.load(local))

    validate_config(CONFIG)
    return CONFIG