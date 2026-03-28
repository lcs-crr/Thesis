"""
Lucas Correia
LIACS | Leiden University
Einsteinweg 55 | 2333 CC Leiden | The Netherlands
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from utilities import data_class

# Declare constants
SEED = 1
FREQ_TARGET = 2
FREQ_ORIG = 10
MODAL_FEATURES = 1

# Set fixed seed for random operations
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

# Load variables in .env file
load_dotenv()

# Load directory paths from .env file
data_path = os.path.join(os.environ["data_path"], "pdm")

data_processor = data_class.DataProcessor(
    original_sampling_rate=FREQ_ORIG,
    target_sampling_rate=FREQ_TARGET,
    scale_method="z-score",
)

# Specify paths for loading and saving data
data_load_path = os.path.join(data_path, "1_parsed")
data_save_path = os.path.join(data_path, "2_preprocessed")
os.makedirs(data_save_path, exist_ok=True)

# Load training and testing data
train_list = data_processor.load_pickle(os.path.join(data_load_path, "train.pkl"))
test_list = data_processor.load_pickle(os.path.join(data_load_path, "test.pkl"))

# Split train_list and test_list into dynamic and status channels to downsample separately
train_list_dynamic = [train_ts[:, :-MODAL_FEATURES] for train_ts in train_list]
train_list_status = [train_ts[:, -MODAL_FEATURES:] for train_ts in train_list]
test_list_dynamic = [test_ts[:, :-MODAL_FEATURES] for test_ts in test_list]
test_list_status = [test_ts[:, -MODAL_FEATURES:] for test_ts in test_list]

# Downsample data lists
train_list_resampled_dynamic = data_processor.downsample_list(train_list_dynamic)
train_list_resampled_status = [
    train_ts[:: FREQ_ORIG // FREQ_TARGET] for train_ts in train_list_status
]
test_list_resampled_dynamic = data_processor.downsample_list(test_list_dynamic)
test_list_resampled_status = [
    test_ts[:: FREQ_ORIG // FREQ_TARGET] for test_ts in test_list_status
]

# Combine dynamic and status channels back into lists
train_list_resampled = [
    np.hstack((train_dynamic, train_status))
    for train_dynamic, train_status in zip(
        train_list_resampled_dynamic, train_list_resampled_status
    )
]
test_list_resampled = [
    np.hstack((test_dynamic, test_status))
    for test_dynamic, test_status in zip(
        test_list_resampled_dynamic, test_list_resampled_status
    )
]

# Split training data into training and validation
train_list_resampled, val_list_resampled = train_test_split(
    train_list_resampled, random_state=SEED, test_size=0.2
)

# Find the scalers for each feature
data_processor.find_scalers_from_list(train_list_resampled)

# Scale data
train_list_scaled = data_processor.scale_list(train_list_resampled)
val_list_scaled = data_processor.scale_list(val_list_resampled)
test_list_scaled = data_processor.scale_list(test_list_resampled)

# Find the window size
data_processor.find_window_size_from_list(train_list_scaled)

# Window sequences inside lists
scaled_train_window = data_processor.window_list(train_list_scaled)
scaled_val_window = data_processor.window_list(val_list_scaled)

# Create tf.data objects
tfdata_train = tf.data.Dataset.from_tensor_slices(scaled_train_window)
tfdata_val = tf.data.Dataset.from_tensor_slices(scaled_val_window)

# Shuffle and batch tf.data objects
tfdata_train = tfdata_train.shuffle(tfdata_train.cardinality(), seed=SEED)
tfdata_val = tfdata_val.shuffle(tfdata_val.cardinality(), seed=SEED)

# Save training and validation data as tf.data
tf.data.Dataset.save(tfdata_train, os.path.join(data_save_path, "train"))
tf.data.Dataset.save(tfdata_val, os.path.join(data_save_path, "val"))

# Save training, validation and testing data as pickle files
data_processor.dump_pickle(train_list_scaled, os.path.join(data_save_path, "train.pkl"))
data_processor.dump_pickle(val_list_scaled, os.path.join(data_save_path, "val.pkl"))
data_processor.dump_pickle(test_list_scaled, os.path.join(data_save_path, "test.pkl"))
