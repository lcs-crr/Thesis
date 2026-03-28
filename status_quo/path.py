"""
Lucas Correia
LIACS | Leiden University
Einsteinweg 55 | 2333 CC Leiden | The Netherlands
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
import random
from dotenv import load_dotenv
from utilities import data_class

# Declare constants
SEED = 1
FREQ_TARGET = 5
FREQ_ORIG = 10
AD_MODE = "us"  # or 'ss'

# Set fixed seed for random operations
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# Load variables in .env file
load_dotenv()

# Load directory paths from .env file
data_path = os.environ["data_path"]
model_path = os.environ["model_path"]

data_load_path = os.path.join(data_path, "Thesis", "1_postsim")
data_save_path = os.path.join(data_path, "Thesis", "2_preprocessed")

processor = data_class.DataProcessor(
    original_sampling_rate=FREQ_ORIG,
    target_sampling_rate=FREQ_TARGET,
)

# Specify paths for loading and saving data
if AD_MODE == "us":
    train_list_folded = processor.load_pickle(os.path.join(data_load_path, "train.pkl"))
    prefix = "unsupervised"
else:
    train_list_folded = processor.load_pickle(
        os.path.join(data_load_path, "train_clean.pkl")
    )
    prefix = "semisupervised"
test_list_folded = processor.load_pickle(os.path.join(data_load_path, "test.pkl"))

for fold_idx, test_list in enumerate(test_list_folded):
    total_time_steps = 0
    anomaly_time_steps = 0
    for test_ts in test_list:
        total_time_steps += len(test_ts)
        if test_ts.dtype.metadata["file_name"].split("_")[-2] != "normal":
            groundtruth_anomaly_start = int(
                test_ts.dtype.metadata["file_name"].split("_")[-1].split(".")[0]
            )
            anomaly_time_steps += len(test_ts) - groundtruth_anomaly_start
    print(anomaly_time_steps / total_time_steps)
