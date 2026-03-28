"""
Lucas Correia
LIACS | Leiden University
Einsteinweg 55 | 2333 CC Leiden | The Netherlands
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import scipy
import numpy as np
import random
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from utilities import data_class

cycle_list = [
    "FTP75",
    "US06",
    "SC03",
    "HWFET",
    "NYCC",
    "HUDDS",
    "LA92",
    "IM240",
    "UDDS",
    "WLTP1",
    "WLTP2",
    "WLTP3",
    "CADCUrban",
    "CADCRural",
    "CADC130",
    "CADC150",
    "JC08",
    "JC08Hot",
    "WHVC",
    "BCDC",
    "RTS95",
    "ETCFIGE4",
    "CUEDCPetrol",
    "CUEDCSPC240",
    "CUEDCDieselMC",
    "CUEDCDieselNA",
    "CUEDCDieselNB",
    "CUEDCDieselME",
    "CUEDCDieselNC",
    "CUEDCDieselNCH",
    "CLTCPassenger",
    "CLTCCommercial",
    "CWTVC",
]

# Declare constants
SEED = 1

# Set fixed seed for random operations
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# Load variables in .env file
load_dotenv()

# Load directory paths from .env file
data_path = os.path.join(os.environ["data_path"], "path")

# Specify paths for loading and saving data
data_load_path = os.path.join(data_path, "0_simulation")
data_save_path = os.path.join(data_path, "generalisation", "1_postsim")
os.makedirs(data_save_path, exist_ok=True)

# Load file names of normal and anomalous time series
normal_file_name_list = sorted(
    [f for f in os.listdir(data_load_path) if f.endswith("_normal_0.mat")]
)
control_file_name_list = sorted(
    [f for f in os.listdir(data_load_path) if f.endswith("_control.mat")]
)
anomalous_file_name_list = sorted(
    [
        f
        for f in os.listdir(data_load_path)
        if not (f.endswith("_normal_0.mat") or f.endswith("_control.mat"))
    ]
)

# Load .mat files as numpy arrays and add file names as metadata
normal_list = [
    scipy.io.loadmat(os.path.join(data_load_path, file_name))["array"].astype(
        np.dtype("float32", metadata={"file_name": file_name})
    )
    for file_name in normal_file_name_list
]
control_list = [
    scipy.io.loadmat(os.path.join(data_load_path, file_name))["array"].astype(
        np.dtype("float32", metadata={"file_name": file_name})
    )
    for file_name in control_file_name_list
]
anomalous_list = [
    scipy.io.loadmat(os.path.join(data_load_path, file_name))["array"].astype(
        np.dtype("float32", metadata={"file_name": file_name})
    )
    for file_name in anomalous_file_name_list
]

mean_list = {}
for cycle in cycle_list:
    for normal_ts in normal_list:
        if cycle in normal_ts.dtype.metadata["file_name"]:
            mean_list.update({cycle: normal_ts[:, 9].mean()})
            break

mean_list = sorted(mean_list.items(), key=lambda item: item[1])
n = len(mean_list)
third = n // 3
mean_fold1 = mean_list[:third]
mean_fold2 = mean_list[third:-third]
mean_fold3 = mean_list[-third:]

std_list = {}
for cycle in cycle_list:
    for normal_ts in normal_list:
        if cycle in normal_ts.dtype.metadata["file_name"]:
            std_list.update({cycle: normal_ts[:, 9].std()})
            break

std_list = sorted(std_list.items(), key=lambda item: item[1])
std_fold1 = std_list[:third]
std_fold2 = std_list[third:-third]
std_fold3 = std_list[-third:]

# Make sure anomalous_list and control_list are equally long
assert len(anomalous_list) == len(control_list)

# Find indices of anomalous time series identic to their control counterparts
# Trim random amount from beginning of each anomalous time series and adjust file name
# Trim same amount from beginning of each corresponding control time series
remove_index = []
for anomalous_idx, anomalous_ts in enumerate(anomalous_list):
    # Find indices of anomalous time series identic to their control counterparts
    if np.mean(abs(control_list[anomalous_idx] - anomalous_ts)) < 1:
        remove_index.append(anomalous_idx)
    else:
        file_name = anomalous_ts.dtype.metadata["file_name"]
        anomaly_start_time_components = file_name.split("_")
        anomaly_start_time, _ = anomaly_start_time_components[-1].split(".")
        anomaly_start_time = int(anomaly_start_time)
        trimmed_start_time = int(
            len(anomalous_ts) * round(np.random.uniform(0, 0.1), 2)
        )
        # Trim random amount from beginning of each anomalous time series
        anomalous_ts = anomalous_ts[trimmed_start_time:]
        # Trim same amount from beginning of each corresponding control time series
        control_ts = control_list[anomalous_idx][trimmed_start_time:]
        # Adjust file_name in metadata to new anomaly start time
        if anomaly_start_time != 0:
            new_anomaly_start_time = anomaly_start_time - trimmed_start_time
            new_file_name = (
                "_".join(anomaly_start_time_components[:-1])
                + "_"
                + str(new_anomaly_start_time)
                + ".mat"
            )
        else:
            new_file_name = (
                "_".join(anomaly_start_time_components[:-1])
                + "_"
                + str(anomaly_start_time)
                + ".mat"
            )
        anomalous_ts = anomalous_ts.astype(
            np.dtype("float32", metadata={"file_name": new_file_name})
        )
        anomalous_list[anomalous_idx] = anomalous_ts
        control_list[anomalous_idx] = control_ts

# Remove anomalous time series identic to their control counterparts and viceversa
anomalous_list = [
    anomalous_ts
    for anomalous_idx, anomalous_ts in enumerate(anomalous_list)
    if anomalous_idx not in remove_index
]
control_list = [
    control_ts
    for control_idx, control_ts in enumerate(control_list)
    if control_idx not in remove_index
]

# Trim random amount from beginning of each normal time series
normal_list = [
    normal_ts[int(len(normal_ts) * round(np.random.uniform(0, 0.1), 2)) :, :]
    for normal_ts in normal_list
]

# Find standard deviation of all time series
stddev = np.std(np.vstack(normal_list + anomalous_list), axis=0)

for anomalous_idx, anomalous_ts in enumerate(anomalous_list):
    anomalous_file_name = anomalous_ts.dtype.metadata["file_name"]
    control_file_name = control_list[anomalous_idx].dtype.metadata["file_name"]
    # Sample noise vector
    noise = np.random.normal(0, 0.01 * stddev, (len(anomalous_ts), len(stddev)))
    # Add noise vector to anomalous sequences and their corresponding control sequences
    anomalous_list[anomalous_idx] = (anomalous_ts + noise).astype(
        np.dtype("float32", metadata={"file_name": anomalous_file_name})
    )
    control_list[anomalous_idx] = (control_list[anomalous_idx] + noise).astype(
        np.dtype("float32", metadata={"file_name": control_file_name})
    )

for normal_idx, normal_ts in enumerate(normal_list):
    file_name = normal_ts.dtype.metadata["file_name"]
    # Sample noise vector
    noise = np.random.normal(0, 0.01 * stddev, (len(normal_ts), len(stddev)))
    # Add noise vector to normal sequences
    normal_list[normal_idx] = (normal_ts + noise).astype(
        np.dtype("float32", metadata={"file_name": file_name})
    )

# Combine normal_list and anomalous_list and shuffle the resulting list
total_list = normal_list + anomalous_list
random.shuffle(total_list)

train_list, test_list = train_test_split(total_list, test_size=0.33, random_state=SEED)

train_fold1 = [
    train_ts
    for train_ts in train_list
    if any(
        cycle == train_ts.dtype.metadata["file_name"].split("_")[0]
        for cycle, _ in mean_fold1
    )
]
train_fold2 = [
    train_ts
    for train_ts in train_list
    if any(
        cycle == train_ts.dtype.metadata["file_name"].split("_")[0]
        for cycle, _ in mean_fold2
    )
]
train_fold3 = [
    train_ts
    for train_ts in train_list
    if any(
        cycle == train_ts.dtype.metadata["file_name"].split("_")[0]
        for cycle, _ in mean_fold3
    )
]
train_fold4 = [
    train_ts
    for train_ts in train_list
    if any(
        cycle == train_ts.dtype.metadata["file_name"].split("_")[0]
        for cycle, _ in std_fold1
    )
]
train_fold5 = [
    train_ts
    for train_ts in train_list
    if any(
        cycle == train_ts.dtype.metadata["file_name"].split("_")[0]
        for cycle, _ in std_fold2
    )
]
train_fold6 = [
    train_ts
    for train_ts in train_list
    if any(
        cycle == train_ts.dtype.metadata["file_name"].split("_")[0]
        for cycle, _ in std_fold3
    )
]

train_fold = [
    train_fold2 + train_fold3,  # train_fold1
    train_fold1 + train_fold3,  # train_fold2
    train_fold1 + train_fold2,  # train_fold3
    train_fold5 + train_fold6,  # train_fold4
    train_fold4 + train_fold6,  # train_fold5
    train_fold4 + train_fold5,  # train_fold6
    train_list,
]

train_fold_clean = []
for fold in train_fold:
    train_fold_temp = []
    for train_ts in fold:
        if "normal" in train_ts.dtype.metadata["file_name"]:
            train_fold_temp.append(train_ts)
    train_fold_clean.append(train_fold_temp)

# Save train_list, test_list, train_list_clean and test_list_clean as pickle files
data_class.DataProcessor().dump_pickle(
    train_fold, os.path.join(data_save_path, "train.pkl")
)
data_class.DataProcessor().dump_pickle(
    train_fold_clean, os.path.join(data_save_path, "train_clean.pkl")
)
data_class.DataProcessor().dump_pickle(
    [test_list] * 7, os.path.join(data_save_path, "test.pkl")
)
