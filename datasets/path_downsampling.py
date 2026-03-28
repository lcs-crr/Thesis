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
FREQ_TARGET = 2
FREQ_ORIG = 10
AD_MODE = "us"  # or 'ss'

# Set fixed seed for random operations
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

# Load variables in .env file
load_dotenv()

# Load directory paths from .env file
asset_path = os.path.join(os.environ["asset_path"])
data_path = os.path.join(os.environ["data_path"], "path")
model_path = os.path.join(os.environ["model_path"], "path")

# Specify paths for loading and saving data
data_load_path = os.path.join(data_path, "1_postsim")
if AD_MODE == "us":
    train_list_fold = data_class.DataProcessor().load_pickle(
        os.path.join(data_load_path, "train.pkl")
    )
else:
    train_list_fold = data_class.DataProcessor().load_pickle(
        os.path.join(data_load_path, "train_clean.pkl")
    )

results = []
for FREQ_TARGET in range(1, 10):
    data_processor = data_class.DataProcessor(
        original_sampling_rate=FREQ_ORIG,
        target_sampling_rate=FREQ_TARGET,
        scale_method="z-score",
    )

    for fold_idx, _ in enumerate(train_list_fold):
        # Downsample data lists
        train_list_resampled = data_processor.downsample_list(
            train_list_fold[fold_idx], reduce=False
        )

        fold_error = 0
        for ts_idx, ts in enumerate(train_list_resampled):
            fold_error += mean_squared_error(train_list_fold[fold_idx][ts_idx], ts)
        fold_error /= len(train_list_resampled)

        results.append(
            {
                "Target frequency": FREQ_TARGET,
                "Fold": fold_idx,
                "Fold error": fold_error,
            }
        )

results = pd.DataFrame(results)
averages = results.groupby("Target frequency")["Fold error"].mean().values
x = np.arange(0.5, 5, 0.5)

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=averages, mode="lines", line=dict(color="black")))

fig.update_xaxes(
    title_text="Cut-Off Frequency [Hz]",
    linecolor="black",
    showgrid=False,
    gridcolor="gray",
    gridwidth=0.3,
    zeroline=False,
    mirror=True,
)
fig.update_yaxes(
    title_text="Mean-squared error [-]",
    linecolor="black",
    showgrid=True,
    gridcolor="gray",
    gridwidth=0.3,
    zeroline=False,
    mirror=True,
)

fig.update_layout(
    font=dict(family="Times New Roman", size=18, color="black"),
    plot_bgcolor="white",
    showlegend=False,  # Remove the legend for the entire figure
)
fig.show()
fig.write_image(os.path.join(asset_path, "path_mse.svg"))
