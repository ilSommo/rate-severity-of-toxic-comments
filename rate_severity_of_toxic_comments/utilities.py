__version__ = '0.1.0'
__author__ = 'Lorenzo Menghini, Martino Pulici, Alessandro Stockman, Luca Zucchini'

import random

import numpy as np
import torch
from transformers import AutoTokenizer, pipeline

from rate_severity_of_toxic_comments.embedding import build_embedding_matrix, load_embedding_model
from rate_severity_of_toxic_comments.preprocessing import AVAILABLE_PREPROCESSING_PIPELINES
from rate_severity_of_toxic_comments.tokenizer import NaiveTokenizer

_bad_words = []


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
    if not all([p not in AVAILABLE_PREPROCESSING_PIPELINES for p in config["preprocessing"]]):
        raise ValueError()
    # TODO Add validation for other values

    """ 
    
    CHECKS CORRCT CONFIG FILE AND ELABORATION OF DATA DEPENDING ON CONFIGURATION
    
    """

    if config["run_mode"] == "pretrained":
        config["tokenizer"] = AutoTokenizer.from_pretrained(
            config['model_name'])
    else:
        config["tokenizer"] = NaiveTokenizer(config["vocab_file"])
        embedding_model = load_embedding_model(config)
        embedding_matrix = build_embedding_matrix(embedding_model, config)
        config["embedding_matrix"] = embedding_matrix
    return config
