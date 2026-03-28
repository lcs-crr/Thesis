"""
Lucas Correia
LIACS | Leiden University
Einsteinweg 55 | 2333 CC Leiden | The Netherlands
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from dotenv import load_dotenv
from utilities import data_class
import numpy as np
import plotly.io as pio
from statsmodels.tsa import stattools
import plotly.graph_objects as go  # noqa: E402

pio.renderers.default = "browser"

# Declare constants
SEED = 1
FREQ_TARGET = 2
FREQ_ORIG = 10
AD_MODE = "ss"  # or 'ss'

# Set fixed seed for random operations
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

# Load variables in .env file
load_dotenv()

# Load directory paths from .env file
asset_path = os.path.join(os.environ["asset_path"])
data_path = os.path.join(os.environ["data_path"], "prop")
model_path = os.path.join(os.environ["model_path"], "prop")

# Specify paths for loading and saving data
data_load_path = os.path.join(data_path, "1_parsed")
train_list_fold = data_class.DataProcessor().load_pickle(
    os.path.join(data_load_path, "train.pkl")
)

data_processor = data_class.DataProcessor(
    original_sampling_rate=FREQ_ORIG,
    target_sampling_rate=FREQ_TARGET,
    scale_method="z-score",
)

results = []
fold_idx = 0
# Downsample data lists
train_list_resampled = data_processor.downsample_list(
    train_list_fold[fold_idx], reduce=True
)
# Pre-allocate list for possible window sizes
window_size_ts = []
# Iterate through list of multivariate time series
for ts_idx, ts in enumerate(train_list_resampled):
    # if cycle in ts.dtype.metadata['file_name']:
    window_size_channel = []
    # Iterate through channels
    for channel in range(ts.shape[-1]):
        corr_array = stattools.acf(ts[:, channel], alpha=0.01, nlags=4096)
        upper_y = corr_array[1][:, 1] - corr_array[0]
        corr = corr_array[0]
        try:
            window_size_channel.append(
                {channel: np.min(np.where(corr - upper_y < 0)[0])}
            )
        except Exception:
            continue
    # Append maximum window size for each channel
    window_size_ts.append(window_size_channel)

max_sublist = 0
max_key = 0
max_value = float("-inf")
for sublist_idx, sublist in enumerate(window_size_ts):
    for item in sublist:
        for key, value in item.items():
            if value > max_value:
                max_value = value
                max_key = key
                max_sublist = sublist_idx

channel_idx = max_key
ts = train_list_resampled[max_sublist]
fig = go.Figure()
corr_array = stattools.acf(ts[:, channel_idx], alpha=0.01, nlags=384)
corr = corr_array[0]
lower_y = corr_array[1][:, 0] - corr
upper_y = corr_array[1][:, 1] - corr
try:
    [
        fig.add_scatter(
            x=(x, x),
            y=(0, corr[x]),
            mode="lines",
            line_color="#3f3f3f",
        )
        for x in range(0, len(corr), 4)
    ]
    fig.add_scatter(
        x=np.arange(len(corr)),
        y=upper_y,
        mode="lines",
        line_color="rgba(255,255,255,0)",
    )
    fig.add_scatter(
        x=np.arange(len(corr)),
        y=lower_y,
        mode="lines",
        fillcolor="rgba(0, 0, 0, 0.2)",
        fill="tonexty",
        line_color="rgba(255,255,255,0)",
    )
except Exception:
    pass

fig.update_traces(showlegend=False)
fig.update_xaxes(
    range=[-1, 384],
    title_text="Lags [-]",
    linecolor="black",
    showgrid=False,
    gridcolor="gray",
    gridwidth=0.3,
    zeroline=False,
    mirror=True,
)
fig.update_yaxes(
    zerolinecolor="#000000",
    title_text="Autocorrelation [-]",
    linecolor="black",
    showgrid=True,
    gridcolor="gray",
    gridwidth=0.3,
    zeroline=False,
    mirror=True,
)

fig.update_layout(
    font=dict(family="Times New Roman", size=18, color="black"),
    showlegend=False,
    plot_bgcolor="white",
)
fig.update_xaxes(matches="x")
fig.show()
fig.write_image(os.path.join(asset_path, "prop_acf.svg"))
