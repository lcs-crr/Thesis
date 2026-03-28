"""
Lucas Correia
LIACS | Leiden University
Einsteinweg 55 | 2333 CC Leiden | The Netherlands
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
from dotenv import load_dotenv
from utilities.detection_class import AnomalyDetector
from utilities.plotter import plot_fancy_ts

channel_list = [
    "Ch #1 [rad/s]",
    "Ch #2 [Nm]",
    "Ch #3 [Nm]",
    "Ch #4 [Nm]",
    "Ch #5 [%]",
    "Ch #6 [A]",
    "Ch #7 [W]",
    "Ch #8 [N]",
    "Ch #9 [N]",
    "Ch #10 [rad/s]",
    "Ch #11 [rad/s]",
    "Ch #12 [-]",
    "Ch #13 [-]",
    "Ch #14 [°C]",
    "Ch #15 [W]",
    "Ch #16 [W]",
]

# channel_list = [
#     r"$\text{Ch 1}\\ \left[\si[per-mode = symbol]{\radian\per\second}\right]$",
#     r"$\text{Ch 2}\\ \left[\si{\newton\metre}\right]$",
#     r"$\text{Ch 3}\\ \left[\si{\newton\metre}\right]$",
#     r"$\text{Ch 4}\\ \left[\si{\newton\metre}\right]$",
#     r"$\text{Ch 5}\\ \left[\si{\percent}\right]$",
#     r"$\text{Ch 6}\\ \left[\si{\ampere}\right]$",
#     r"$\text{Ch 7}\\ \left[\si{\watt}\right]$",
#     r"$\text{Ch 8}\\ \left[\si{\newton}\right]$",
#     r"$\text{Ch 9}\\ \left[\si{\newton}\right]$",
#     r"$\text{Ch 10}\\ \left[\si[per-mode = symbol]{\radian\per\second}\right]$",
#     r"$\text{Ch 11}\\ \left[\si[per-mode = symbol]{\radian\per\second}\right]$",
#     r"$\text{Ch 12}\\ \left[-\right]$",
#     r"$\text{Ch 13}\\ \left[-\right]$",
#     r"$\text{Ch 14}\\ \left[\si{\degreeCelsius}\right]$",
#     r"$\text{Ch 15}\\ \left[\si{\watt}\right]$",
#     r"$\text{Ch 16}\\ \left[\si{\watt}\right]$",
# ]

# Load variables in .env file
load_dotenv()

# Load directory paths from .env file
data_path = os.path.join(os.environ["data_path"], "path")
asset_path = os.environ["asset_path"]

# Iterate over all seeds and folds
model_seed = 1
fold_idx = 0

# Declare model name and paths
data_load_path = os.path.join(data_path, "1_postsim")

detector = AnomalyDetector()

# Load data
test_list = detector.load_pickle(os.path.join(data_load_path, "test.pkl"))
anomalous_list = detector.load_pickle(os.path.join(data_load_path, "anomalous.pkl"))
control_list = detector.load_pickle(os.path.join(data_load_path, "control.pkl"))

plot_fancy_ts(
    test_list[0][3],
    sampling_rate=10,
    channel_list=channel_list,
    save_path=os.path.join(asset_path, "data_plot_nominal.svg"),
)

plot_fancy_ts(
    control_list[12],
    output_ts=anomalous_list[12],
    sampling_rate=10,
    channel_list=channel_list,
    save_path=os.path.join(asset_path, "anomaly_plot_brakeregen.svg"),
)

plot_fancy_ts(
    control_list[50],
    output_ts=anomalous_list[50],
    sampling_rate=10,
    channel_list=channel_list,
    save_path=os.path.join(asset_path, "anomaly_plot_extwind.svg"),
)

plot_fancy_ts(
    control_list[74],
    output_ts=anomalous_list[74],
    sampling_rate=10,
    channel_list=channel_list,
    save_path=os.path.join(asset_path, "anomaly_plot_pumpspeed.svg"),
)

plot_fancy_ts(
    control_list[149],
    output_ts=anomalous_list[149],
    sampling_rate=10,
    channel_list=channel_list,
    save_path=os.path.join(asset_path, "anomaly_plot_torquereq.svg"),
)

plot_fancy_ts(
    control_list[157],
    output_ts=anomalous_list[157],
    sampling_rate=10,
    channel_list=channel_list,
    save_path=os.path.join(asset_path, "anomaly_plot_wheeldiameter.svg"),
)

plot_fancy_ts(
    control_list[203],
    output_ts=anomalous_list[203],
    sampling_rate=10,
    channel_list=channel_list,
    save_path=os.path.join(asset_path, "anomaly_plot_driverreaction.svg"),
)
