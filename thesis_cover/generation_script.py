import os
import plotly.io as pio
import numpy as np

import plotly.graph_objects as go
from dotenv import load_dotenv

pio.renderers.default = "browser"

# Load variables in .env file
load_dotenv()

# Load directory paths from .env file
asset_path = os.environ["asset_path"]

N = 10000
T = 0.001

x = np.linspace(0.0, N * T, N, endpoint=False)
y = np.sin(np.pi * x) + 1

# Build symmetric anomaly: peaks in the middle, normal at top and bottom.
num_anomaly_waves = 25  # odd number so there's a clear center wave
# num_normal_waves = (
#     31 - num_anomaly_waves
# )  # number of normal waves to add at top and bottom
num_normal_waves = 21
num_waves = num_anomaly_waves + 2 * num_normal_waves
y_list = []

# Add normal waves at the bottom
for i in range(num_normal_waves):
    y_list.append(y.copy())

# Add anomaly waves with symmetric pattern and plateau at the top
center = num_anomaly_waves // 2  # index of the middle anomaly wave
plateau_width = 6  # number of waves at maximum anomaly level (plateau)
half_plateau = plateau_width // 2

for i in range(num_anomaly_waves):
    # Calculate distance from center, but clamp to create plateau
    raw_distance = abs(i - center)
    # Waves within half_plateau of center are at max (distance = 0 from plateau edge)
    distance_from_plateau = max(0, raw_distance - half_plateau)
    max_distance = center - half_plateau  # max distance from plateau edge

    # Exponential scaling: normalized distance raised to a power, then scaled
    if max_distance > 0:
        normalized = 1 - (
            distance_from_plateau / max_distance
        )  # 1 at plateau, 0 at edges
    else:
        normalized = 1
    normalized = max(0, min(1, normalized))  # clamp to [0, 1]

    # anomaly_magnitude = (
    #     (np.exp(normalized * 2) - 1) / (np.exp(2) - 1) * 4
    # )  # exponential growth, max ~4
    anomaly_magnitude = (np.exp(normalized * 1.5) - 1) / (np.exp(1.5) - 1) * 4
    add_vector = np.ones_like(x)
    add_vector[5500:7500] = 1 + anomaly_magnitude
    y_list.append(y * add_vector)

# Add normal waves at the top
for i in range(num_normal_waves):
    y_list.append(y.copy())

# Stack all waves on a single axis with vertical offsets for a tight, overlapping effect.
fig = go.Figure()
offset_step = 1.1  # vertical gap between successive waves (smaller = tighter)

# Draw waves from top to bottom so lower waves cover upper ones
for idx in range(len(y_list) - 1, -1, -1):
    wave = y_list[idx]
    y_offset = wave + idx * offset_step
    baseline = idx * offset_step  # flat baseline for this wave

    # Create filled polygon (no visible outline)
    x_polygon = np.concatenate([x, x[::-1]])
    y_polygon = np.concatenate([y_offset, np.full_like(x, baseline)[::-1]])

    fig.add_trace(
        go.Scatter(
            x=x_polygon,
            y=y_polygon,
            mode="none",
            fill="toself",
            fillcolor="white",
            line=dict(width=0),
            showlegend=False,
        )
    )

    # Draw just the wave line on top
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_offset,
            mode="lines",
            line=dict(color="black", width=1.5),
            showlegend=False,
        )
    )

fig.update_xaxes(
    showgrid=False,
    zeroline=False,
    visible=False,
)
fig.update_yaxes(
    showgrid=False,
    zeroline=False,
    visible=False,
)

fig.update_layout(
    font=dict(
        family="Times New Roman",
        size=18,
        color="black",
    ),
    plot_bgcolor="white",
    showlegend=False,
    height=(num_anomaly_waves + num_normal_waves) * 30,
    width=800,
    margin=dict(l=0, r=0, t=0, b=0),
)
fig.show()
fig.write_image(os.path.join(asset_path, "cover.svg"))
