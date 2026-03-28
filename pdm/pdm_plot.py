"""
Lucas Correia
LIACS | Leiden University
Einsteinweg 55 | 2333 CC Leiden | The Netherlands
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
from dotenv import load_dotenv
from utilities import detection_class, plotter

# Load variables in .env file
load_dotenv()

# Load directory paths from .env file
data_path = os.path.join(os.environ["data_path"], "pdm")
asset_path = os.path.join(os.environ["asset_path"])

# Declare model name and paths
data_load_path = os.path.join(data_path, "2_preprocessed")

detector = detection_class.AnomalyDetector()

# Load data
test_list = detector.load_pickle(os.path.join(data_load_path, "test.pkl"))

plotter.plot_fancy_ts(
    test_list[0], sampling_rate=2, save_path=os.path.join(asset_path, "pdm_plot.svg")
)
