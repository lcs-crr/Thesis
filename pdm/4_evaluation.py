"""
Lucas Correia
LIACS | Leiden University
Einsteinweg 55 | 2333 CC Leiden | The Netherlands
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
from dotenv import load_dotenv
from utilities import detection_class
import tensorflow as tf
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

# Declare constants
MODEL_NAME = "tevaemm"
SEED = 1

# Load variables in .env file
load_dotenv()

# Load directory paths from .env file
asset_path = os.environ["asset_path"]
data_path = os.path.join(os.environ["data_path"], "pdm")
model_path = os.path.join(os.environ["model_path"], "pdm")

# Declare model name and paths
model_name = MODEL_NAME + "_" + str(SEED)
data_load_path = os.path.join(data_path, "2_preprocessed")
model_load_path = os.path.join(model_path, model_name)

# Load tf.data to get window_size
tfdata_train = tf.data.Dataset.load(os.path.join(data_load_path, "train"))

detector = detection_class.AnomalyDetector(
    model_path=model_load_path,
    window_size=tfdata_train.element_spec.shape[0],  # type: ignore
    sampling_rate=2,
    original_sampling_rate=10,
    calculate_delay=True,
    label_keyword="normal",
)

# Load data
val_list = detector.load_pickle(os.path.join(data_load_path, "val.pkl"))
test_list = detector.load_pickle(os.path.join(data_load_path, "test.pkl"))

# Load inference results
val_output_list = detector.load_pickle(os.path.join(model_load_path, "val_output.pkl"))
val_detection_score_list = detector.load_pickle(
    os.path.join(model_load_path, "val_detection_score.pkl")
)
test_output_list = detector.load_pickle(
    os.path.join(model_load_path, "test_output.pkl")
)
test_detection_score_list = detector.load_pickle(
    os.path.join(model_load_path, "test_detection_score.pkl")
)

# Obtain the unsupervised threshold
threshold = detector.unsupervised_threshold(val_detection_score_list)

health_index = [score_ts.max() for score_ts in test_detection_score_list]

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=np.arange(len(health_index)),
        y=health_index,
        line=dict(color="black", width=2),
    )
)

# add horizontal line at height of threshold
fig.add_hline(y=threshold, line_width=3, line_dash="dash", line_color="red")

fig.update_xaxes(
    title_text="Test Index [-]",
    linecolor="black",
    showgrid=False,
    gridcolor="gray",
    gridwidth=0.3,
    zeroline=True,
    zerolinewidth=0.3,
    zerolinecolor="gray",
    mirror=True,
    matches="x",
)

fig.update_yaxes(
    title_text="Health Indicator [-]",
    linecolor="black",
    showgrid=True,
    gridcolor="gray",
    gridwidth=0.3,
    zeroline=True,
    zerolinewidth=0.3,
    zerolinecolor="gray",
    mirror=True,
)

fig.update_layout(
    showlegend=False,
    height=500,
    width=1000,
    plot_bgcolor="white",
    font=dict(size=20, family="Times New Roman", color="black"),
)

fig.show()
save_type = "svg"
fig.write_image(os.path.join(asset_path, "pdm" + "." + save_type))
