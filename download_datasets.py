import os
import json
import argparse
import requests

BASE_DIR = os.path.join("res", "data")

def download(file_path, download_url):
    if not os.path.isfile(file_path):
        req = requests.get(download_url)
        url_content = req.content
        csv_file = open(file_path, "wb")
        csv_file.write(url_content)
        csv_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--verbose", action="store_true")

    DEFAULT_CONFIG_FILE_PATH = "config/default.json"
    LOCAL_CONFIG_FILE_PATH = "config/local.json"

    default = open(DEFAULT_CONFIG_FILE_PATH)
    CONFIG = json.load(default)

    if os.path.exists(LOCAL_CONFIG_FILE_PATH):
        with open(LOCAL_CONFIG_FILE_PATH) as local:
            CONFIG.update(json.load(local))

    download(CONFIG["training_set"]["path"], CONFIG["training_set"]["download"])
    download(CONFIG["test_set"]["path"], CONFIG["test_set"]["download"])


    
