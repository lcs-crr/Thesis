"""
Lucas Correia
LIACS | Leiden University
Einsteinweg 55 | 2333 CC Leiden | The Netherlands
"""

import os
from asammdf import MDF
from dotenv import load_dotenv
from utilities import data_class
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split, RepeatedKFold

SEED = 1

# Load variables in .env file
load_dotenv()

# Load directory paths from .env file
data_path = os.path.join(os.environ["data_path"], "prop")

channel_list = [
    "vehicle_speed",
    "hvb_voltage",
    "hvb_current",
    "hvb_temperature",
    "hvb_soc",
    "em_voltage",
    "em_current",
    "em_torque",
    "em_angular_speed",
    "em_stator_temperature",
    "inverter_temperature",
    "inverter_inlet_temperature",
    "inverter_outlet_temperature",
    "powertrain_controller_temperature",
    "left_axle_torque",
    "right_axle_torque",
    "left_axle_angular_speed",
    "right_axle_angular_speed",
    "coolling_loop_1_inlet_temperature",
    "coolling_loop_1_outlet_temperature",
    "coolling_loop_2_inlet_temperature",
    "coolling_loop_2_outlet_temperature",
]

data_load_path = os.path.join(data_path, "0_raw")
data_save_path = os.path.join(data_path, "1_parsed")
os.makedirs(data_save_path, exist_ok=True)

data_processor = data_class.DataProcessor()

normal_list = []
anomalous_list = []

file_name_list = [f for f in os.listdir(data_load_path) if f.upper().endswith(".MF4")]
for file_name_idx, file_name in enumerate(tqdm(file_name_list)):
    mdf = MDF(
        os.path.join(data_load_path, file_name),
        channel_list=channel_list,
        raise_on_multiple_occurrences=False,
    )
    mdf = mdf.resample(0.1)
    df_temp = mdf.to_dataframe(
        channels=channel_list, ignore_value2text_conversions=True
    )

    # Ensure channel has samples
    if len(df_temp) == 0:
        continue

    # Ensure only dynamic (non-stationary) measurements are kept
    if df_temp["AUSY_N_SWLH_ist"].std() < 0.1 or df_temp["AUSY_N_SWRH_ist"].std() < 0.1:
        continue

    ts_temp = df_temp.values

    # Make sure the resulting array has no NaNs
    if np.any(np.isnan(ts_temp)):
        continue

    ts_temp = ts_temp.astype(np.dtype("float32", metadata={"file_name": file_name}))

    if "_normal_" in file_name:
        normal_list.append(ts_temp)
    else:
        anomalous_list.append(ts_temp)

# Remove half of the normal time series to keep dataset size manageable
normal_list, _ = train_test_split(normal_list, random_state=SEED, test_size=0.5)

# Instatiate RepeatedKFold object with 3 folds
rkf = RepeatedKFold(n_splits=3, n_repeats=1, random_state=SEED)

# Split all data into 3 folds
train_list_fold = []
test_list_fold = []
for fold_idx, (train_indices, test_indices) in enumerate(rkf.split(normal_list)):
    train_list = [normal_list[train_idx] for train_idx in train_indices]
    train_list_fold.append(train_list)
    test_list = [normal_list[test_idx] for test_idx in test_indices]
    test_list = test_list + anomalous_list
    test_list_fold.append(test_list)

# Save normal_list, anomalous_list and control_list as pickle files
data_processor.dump_pickle(normal_list, os.path.join(data_save_path, "normal.pkl"))
data_processor.dump_pickle(
    anomalous_list, os.path.join(data_save_path, "anomalous.pkl")
)

# Save train_list, test_list, train_list_clean and test_list_clean as pickle files
data_processor.dump_pickle(train_list_fold, os.path.join(data_save_path, "train.pkl"))
data_processor.dump_pickle(test_list_fold, os.path.join(data_save_path, "test.pkl"))
