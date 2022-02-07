import os
import json
import argparse
from xml.dom import NotFoundErr
import requests
from interactive import BEST_MODELS_FILE_PATH

from rate_severity_of_toxic_comments.vocabulary import get_preprocess_filenames


DEFAULT_CONFIG_FILE_PATH = "config/default.json"
LOCAL_CONFIG_FILE_PATH = "config/local.json"
VOCAB_CONFIG_FILE_PATH = "config/vocabs.json"
BEST_MODELS_FILE_PATH = "config/best_models.json"

def download(file_path, download_url):
    if not os.path.isfile(file_path):
        if download_url is None:
            raise NotFoundErr("Download url for " + file_path + " file is null")
        headers = {"User-Agent": "Wget/1.12 (cygwin)"}
        req = requests.get(download_url, headers=headers)
        url_content = req.content
        csv_file = open(file_path, "wb")
        csv_file.write(url_content)
        csv_file.close()
        print("File downloaded")
    else:
        print("File already on file system")

if __name__ == "__main__":
    default = open(DEFAULT_CONFIG_FILE_PATH)
    CONFIG = json.load(default)

    if os.path.exists(LOCAL_CONFIG_FILE_PATH):
        with open(LOCAL_CONFIG_FILE_PATH) as local:
            CONFIG.update(json.load(local))
    
    vocabs_file = open(VOCAB_CONFIG_FILE_PATH)
    vocabs = json.load(vocabs_file)
    
    models_file = open(BEST_MODELS_FILE_PATH)
    models = json.load(models_file)

    print("Downloading training set file")
    download(CONFIG["training"]["dataset"]["path"], CONFIG["training"]["dataset"]["download"])
    print("Downloading test set file")
    download(CONFIG["evaluation"]["dataset"]["path"], CONFIG["evaluation"]["dataset"]["download"])

    print("Downloading vocab file")
    preprocess_vocab = get_preprocess_filenames(CONFIG["recurrent"]["preprocessing"], CONFIG["recurrent"]["vocab_file"])
    download(preprocess_vocab, vocabs[preprocess_vocab])

    print("Downloading final models")
    for model in models:
        print("Downloading", model["description"])
        download(model["path"], model["download"])
    print("Finished")



    
