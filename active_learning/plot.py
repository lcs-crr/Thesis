"""
Lucas Correia
LIACS | Leiden University
Einsteinweg 55 | 2333 CC Leiden | The Netherlands
"""

import os
from dotenv import load_dotenv
import plotly.io as pio
from utilities import plotter

pio.renderers.default = "browser"
seeds = [1, 2, 3]
folds = [0, 1, 2]
splits = ["1day", "1week", "2weeks", "3weeks", "4weeks"]

# Load variables in .env file
load_dotenv()
model_path = os.path.join(os.environ["model_path"], "dqs")
asset_path = os.path.join(os.environ["asset_path"])

subplot_titles = ["B=1", "B=5", "B=10"]

plotter.plot_active_learning(
    sheet_path=os.path.join(model_path, "results.xlsx"),
    subplot_titles=subplot_titles,
    mode="budget",
    save_path=asset_path,
    save_type="svg",
)

subplot_titles = ["p_m=0.1", "p_m=0.2", "p_m=0.3"]
plotter.plot_active_learning(
    sheet_path=os.path.join(model_path, "results.xlsx"),
    subplot_titles=subplot_titles,
    mode="mislabelling",
    save_path=asset_path,
    save_type="svg",
)
