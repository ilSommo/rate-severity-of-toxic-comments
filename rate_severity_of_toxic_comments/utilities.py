__version__ = '0.1.0'
__author__ = 'Lorenzo Menghini, Martino Pulici, Alessandro Stockman, Luca Zucchini'

import random

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
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
        #torch.backends.cudnn.enabled = False
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True)


def validate_config(config):
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
