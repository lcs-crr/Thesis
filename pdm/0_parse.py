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

# Load variables in .env file
load_dotenv()

# Load directory paths from .env file
data_path = os.path.join(os.environ["data_path"], "pdm")

channel_list = [
    "vehicle_speed",
    "em_voltage",
    "em_current",
    "em_torque",
    "em_angular_speed",
    "em_stator_temperature",
    "inverter_temperature",
    "left_axle_torque",
    "right_axle_torque",
    "left_axle_angular_speed",
    "right_axle_angular_speed",
    "current_gear",
]

data_load_path = os.path.join(data_path, "0_raw")
data_save_path = os.path.join(data_path, "1_parsed")
os.makedirs(data_save_path, exist_ok=True)

data_processor = data_class.DataProcessor()

data_list = []
file_name_list = [f for f in os.listdir(data_load_path) if f.upper().endswith(".MF4")]
for file_name_idx, file_name in enumerate(tqdm(file_name_list)):
    try:
        mdf = MDF(
            os.path.join(data_load_path, file_name),
            channel_list=channel_list,
            raise_on_multiple_occurrences=False,
        )
        mdf = mdf.resample(0.1)
    except Exception:
        continue
    df_temp = mdf.to_dataframe(
        channels=channel_list, ignore_value2text_conversions=True
    )
    df_temp = df_temp[channel_list]

    ts_temp = df_temp.values

    # Make sure the resulting array has no NaNs
    if np.any(np.isnan(ts_temp)):
        continue

    ts_temp = ts_temp.astype(np.dtype("float32", metadata={"file_name": file_name}))
    data_list.append(ts_temp)

data_list = sorted(data_list, key=lambda x: x.dtype.metadata["file_name"])

train_list = data_list[: int(len(data_list) * 0.33)]
test_list = data_list[int(len(data_list) * 0.33) :]

# Save train_list and test_list as pickle files
data_processor.dump_pickle(train_list, os.path.join(data_save_path, "train.pkl"))
data_processor.dump_pickle(test_list, os.path.join(data_save_path, "test.pkl"))
