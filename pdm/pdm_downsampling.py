"""
Lucas Correia
LIACS | Leiden University
Einsteinweg 55 | 2333 CC Leiden | The Netherlands
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from dotenv import load_dotenv
from utilities import data_class
import pandas as pd
import numpy as np
import plotly.io as pio

import plotly.graph_objects as go  # noqa: E402

pio.renderers.default = "browser"

# Declare constants
SEED = 1
FREQ_ORIG = 10

# Set fixed seed for random operations
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

# Load variables in .env file
load_dotenv()

# Load directory paths from .env file
data_path = os.path.join(os.environ["data_path"], "pdm")
model_path = os.path.join(os.environ["model_path"], "pdm")

# Specify paths for loading and saving data
data_load_path = os.path.join(data_path, "1_parsed")
train_list = data_class.DataProcessor().load_pickle(
    os.path.join(data_load_path, "train.pkl")
)

results = []
for FREQ_TARGET in range(1, 10):
    data_processor = data_class.DataProcessor(
        original_sampling_rate=FREQ_ORIG,
        target_sampling_rate=FREQ_TARGET,
        scale_method="z-score",
    )

    # Downsample data lists
    train_list_resampled = data_processor.downsample_list(train_list, reduce=False)

    error = 0
    for ts_idx, ts in enumerate(train_list_resampled):
        error += mean_squared_error(train_list[ts_idx], ts)
    error /= len(train_list_resampled)

    results.append(
        {
            "Target frequency": FREQ_TARGET,
            "Error": error,
        }
    )

results = pd.DataFrame(results)
averages = results["Error"]
x = np.arange(0.5, 5, 0.5)

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=averages, mode="lines", line=dict(color="black")))

fig.update_xaxes(title_text="Cut-Off Frequency [Hz]")
fig.update_yaxes(title_text="Mean-squared error [-]")

fig.update_layout(
    font=dict(family="Times New Roman", size=18, color="black"),
    showlegend=False,
)
fig.show()
fig.write_image("../figures/pdm_mse.svg")
averages = results.groupby("Target frequency")["Fold error"].mean()
