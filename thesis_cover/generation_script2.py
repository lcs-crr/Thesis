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
base_wave = 0 * np.sin(5 * np.pi * x)

# Create Gaussian curves forming a straight diagonal line from top-left to bottom-right
std = N * T / 32  # standard deviation (controls width)
time_shift = 100 * T  # shift each Gaussian by 100 time steps
n = 7
amplitudes = list(range(n + 1)) + list(range(n - 1, -1, -1))

# Build normal waves
num_waves = 40
y_list = []

# Calculate center and add Gaussian curves
center = num_waves // 2
num_gaussians = len(amplitudes)
half_gaussians = num_gaussians // 2

# Starting position for the first Gaussian (flipped along vertical axis)
horizontal_shift = 1500 * T  # Shift entire formation to the right
base_mean = N * T / 2 + half_gaussians * time_shift + horizontal_shift

# Add normal waves
for i in range(num_waves):
    # Determine if this wave should be a Gaussian
    offset_from_center = i - center
    if abs(offset_from_center) <= half_gaussians:
        # Map to the appropriate amplitude
        gaussian_idx = half_gaussians - abs(offset_from_center)
        amplitude = amplitudes[gaussian_idx]

        # Shift horizontally in reverse (creates diagonal line from top-right to bottom-left)
        mean = base_mean - i * time_shift
        gaussian = amplitude * np.exp(-0.5 * ((x - mean) / std) ** 2)

        # Suppress waves where Gaussian is strong, show waves where it's weak
        if amplitude > 0:
            normalized_gaussian = gaussian / amplitude  # 0 to 1, where 1 is at peak
            suppression_factor = normalized_gaussian**0.5  # More aggressive suppression
            wave_contribution = base_wave * (1 - suppression_factor)
            y_list.append(wave_contribution + gaussian)
        else:
            y_list.append(base_wave.copy())
    else:
        y_list.append(base_wave.copy())

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
    height=num_waves * 30,
    width=800,
    margin=dict(l=0, r=0, t=0, b=0),
)
fig.show()
fig.write_image(os.path.join(asset_path, "cover3.svg"))
