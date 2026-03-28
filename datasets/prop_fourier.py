"""
Lucas Correia
LIACS | Leiden University
Einsteinweg 55 | 2333 CC Leiden | The Netherlands
"""

import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from dotenv import load_dotenv
from utilities import data_class
import scipy
import plotly.io as pio

import plotly.graph_objects as go  # noqa: E402

pio.renderers.default = "browser"

cycle_list = [
    "_1059_",
    "_1060_",
    "_1061_",
    "_1062_",
    "_1063_",
    "_1064_",
    "_1067_",
    "_1068_",
]

# Declare constants
SEED = 1
FREQ_ORIG = 10
AD_MODE = "us"  # or 'ss'

# Set fixed seed for random operations
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

# Load variables in .env file
load_dotenv()

# Load directory paths from .env file
asset_path = os.path.join(os.environ["asset_path"])
data_path = os.path.join(os.environ["data_path"], "prop")
model_path = os.path.join(os.environ["model_path"], "prop")

data_processor = data_class.DataProcessor(scale_method="z-score")

# Specify paths for loading and saving data
data_load_path = os.path.join(data_path, "1_parsed")
train_list_fold = data_processor.load_pickle(os.path.join(data_load_path, "train.pkl"))[
    0
]

# Find the scalers for each feature
data_processor.find_scalers_from_list(train_list_fold)

# Scale data
train_list_scaled = data_processor.scale_list(train_list_fold)

for cycle in cycle_list:
    for ts_idx, ts in enumerate(train_list_scaled):
        if cycle in ts.dtype.metadata["file_name"]:
            yf = 0
            xf = scipy.fft.fftfreq(len(ts), 1 / FREQ_ORIG)[: len(ts) // 2]
            for feature_idx, feature in enumerate(np.rollaxis(ts, axis=-1)):
                yf += 2.0 / len(ts) * np.abs(scipy.fft.fft(feature)[: len(ts) // 2])
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=xf, y=20 * np.log10(yf), mode="lines", line=dict(color="black")
                )
            )

            fig.update_xaxes(
                title_text="Frequency [Hz]",
                linecolor="black",
                showgrid=False,
                gridcolor="gray",
                gridwidth=0.3,
                zeroline=False,
                mirror=True,
            )
            fig.update_yaxes(
                title_text="Amplitude [dB]",
                linecolor="black",
                showgrid=True,
                gridcolor="gray",
                gridwidth=0.3,
                zeroline=False,
                mirror=True,
            )

            fig.update_layout(
                # title=dict(
                #     text=ts.dtype.metadata['file_name']
                # ),
                font=dict(family="Times New Roman", size=18, color="black"),
                plot_bgcolor="white",
                showlegend=False,  # Remove the legend for the entire figure
            )

            if "_1060_" in ts.dtype.metadata["file_name"]:
                fig.show()
                fig.write_image(os.path.join(asset_path, "1060_fft.svg"))
            break
