import os
import scipy
import plotly.io as pio
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

pio.renderers.default = "browser"


# Load variables in .env file
load_dotenv()

# Load directory paths from .env file
asset_path = os.path.join(os.environ["asset_path"])

N = 10000
T = 0.001

x = np.linspace(0.0, N * T, N, endpoint=False)

y = (
    100 * np.sin(2.0 * np.pi * x)
    + 100 * np.sin(2.0 * np.pi * 2 * x + np.pi)
    + 100 * np.cos(2.0 * np.pi * 5 * x)
    + 0.1 * np.sin(2.0 * np.pi * 20 * x)
)

fig = make_subplots(rows=2, cols=1)
fig.add_trace(
    go.Scatter(x=x, y=y, mode="lines", line=dict(color="black")), row=1, col=1
)
fig.update_xaxes(
    title_text="Time [s]",
    row=1,
    col=1,
    linecolor="black",
    showgrid=False,
    gridcolor="gray",
    gridwidth=0.3,
    zeroline=False,
    mirror=True,
)
fig.update_yaxes(
    title_text="Amplitude [-]",
    row=1,
    col=1,
    linecolor="black",
    showgrid=True,
    gridcolor="gray",
    gridwidth=0.3,
    zeroline=False,
    mirror=True,
)

xf = scipy.fft.fftfreq(len(y), T)[: len(y) // 2]
yf = 2.0 / len(y) * np.abs(scipy.fft.fft(y)[: len(y) // 2])

fig.add_trace(
    go.Scatter(x=xf, y=20 * np.log10(yf), mode="lines", line=dict(color="black")),
    row=2,
    col=1,
)
fig.update_xaxes(
    title_text="Frequency [Hz]",
    range=[0, 30],
    row=2,
    col=1,
    linecolor="black",
    showgrid=False,
    gridcolor="gray",
    gridwidth=0.3,
    zeroline=False,
    mirror=True,
)
fig.update_yaxes(
    title_text="Amplitude [dB]",
    row=2,
    col=1,
    linecolor="black",
    showgrid=True,
    gridcolor="gray",
    gridwidth=0.3,
    zeroline=False,
    mirror=True,
)
fig.update_layout(
    font=dict(
        family="Times New Roman",
        size=18,
        color="black",
    ),
    plot_bgcolor="white",
    showlegend=False,  # Remove the legend for the entire figure
)
fig.show()
fig.write_image(os.path.join(asset_path, "simple_wave.svg"))
