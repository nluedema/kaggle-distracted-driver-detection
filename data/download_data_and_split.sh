#!/bin/bash
export KAGGLE_USERNAME="$1"
export KAGGLE_KEY="$2"
kaggle competitions download -c state-farm-distracted-driver-detection
unzip state-farm-distracted-driver-detection.zip
python create_own_split.py
