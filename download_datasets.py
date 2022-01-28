import os
import json
import requests

def download(file_path, download_url):
    if not os.path.isfile(file_path):
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
    DEFAULT_CONFIG_FILE_PATH = "config/default.json"
    LOCAL_CONFIG_FILE_PATH = "config/local.json"

    default = open(DEFAULT_CONFIG_FILE_PATH)
    CONFIG = json.load(default)

    if os.path.exists(LOCAL_CONFIG_FILE_PATH):
        with open(LOCAL_CONFIG_FILE_PATH) as local:
            CONFIG.update(json.load(local))

    print("Downloading training set file")
    download(CONFIG["training_set"]["path"], CONFIG["training_set"]["download"])
    print("Downloading test set file")
    download(CONFIG["test_set"]["path"], CONFIG["test_set"]["download"])


    
