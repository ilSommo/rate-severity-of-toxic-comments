#!/bin/bash

cd res/data
kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification -f train.csv
unzip train.csv.zip
rm train.csv.zip
