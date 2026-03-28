"""
Lucas Correia
LIACS | Leiden University
Einsteinweg 55 | 2333 CC Leiden | The Netherlands
"""

import os.path
import numpy as np
from sklearn import metrics
from utilities import data_class
from plotly import subplots
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

load_path = os.environ["data_path"]
asset_path = os.environ["asset_path"]

train_array = data_class.DataProcessor.load_pickle(
    os.path.join(load_path, "parsed/SWaT_train.pkl")
)
test_array = data_class.DataProcessor.load_pickle(
    os.path.join(load_path, "parsed/SWaT_test.pkl")
)
groundtruth_labels = data_class.DataProcessor.load_pickle(
    os.path.join(load_path, "parsed/SWaT_test_label.pkl")
)

# Find what channels have 0 variance
var_train = np.where(np.mean(train_array, axis=0) == 1)[0]
var_test = np.where(np.mean(test_array, axis=0) == 1)[0]
redundant_channels = np.intersect1d(
    var_train, var_test, assume_unique=True, return_indices=False
)

train_array_no_redundant = np.delete(train_array, redundant_channels, axis=1)
test_array_no_redundant = np.delete(test_array, redundant_channels, axis=1)

trivial_channel = test_array[:, 27]
# trivial_channel = test[:, 18]

predicted_labels = (trivial_channel < 1.6) * 1

# fig = px.line(trivial_channel)
# fig.show()

anomaly_density = sum(groundtruth_labels) / len(groundtruth_labels)
sum_groundtruth = sum(groundtruth_labels)
sum_predicted = sum(predicted_labels)

tn, fp, fn, tp = metrics.confusion_matrix(groundtruth_labels, predicted_labels).ravel()
precision = metrics.precision_score(groundtruth_labels, predicted_labels)
recall = metrics.recall_score(groundtruth_labels, predicted_labels)
f1 = metrics.f1_score(groundtruth_labels, predicted_labels)

fig = subplots.make_subplots(rows=3, cols=1)
fig.add_trace(
    go.Scatter(
        y=trivial_channel, x=np.arange(len(trivial_channel)), line=dict(color="black")
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        y=np.ones_like(trivial_channel) * 1.6,
        x=np.arange(len(trivial_channel)),
        line=dict(color="red"),
    ),
    row=1,
    col=1,
)
fig.update_yaxes(row=1, col=1, range=[0, 2])
fig.update_yaxes(row=1, col=1, title_text="FIT401 [-]")

fig.add_trace(
    go.Scatter(
        y=groundtruth_labels,
        x=np.arange(len(groundtruth_labels)),
        line=dict(color="black"),
    ),
    row=2,
    col=1,
)
fig.update_yaxes(row=2, col=1, title_text="Ground-truth Labels [-]")

fig.add_trace(
    go.Scatter(
        y=predicted_labels, x=np.arange(len(predicted_labels)), line=dict(color="black")
    ),
    row=3,
    col=1,
)
fig.update_xaxes(row=3, col=1, title_text="Samples [-]")
fig.update_yaxes(row=3, col=1, title_text="Predicted Labels [-]")

fig.update_xaxes(showline=True, linewidth=2, linecolor="black")
fig.update_yaxes(showline=True, linewidth=2, linecolor="black")

fig.update_layout(
    showlegend=False,
    plot_bgcolor="white",
    yaxis2=dict(tickvals=[0, 1]),
    yaxis3=dict(tickvals=[0, 1]),
    width=1000,
    height=3 * 150,
    font_color="black",
    yaxis=dict(tickvals=[0, 1, 2]),
)

fig.write_image(os.path.join(asset_path, "plots/swat.svg"))
