"""
Lucas Correia
LIACS | Leiden University
Einsteinweg 55 | 2333 CC Leiden | The Netherlands
"""

import plotly
import numpy as np
import random
from plotly import subplots
from statsmodels.tsa import stattools
from datetime import datetime
import os
import time
import plotly.graph_objects as go
from plotly.colors import sequential
import pandas as pd

viridis_colors = sequential.Viridis  # List of hex colors from the Viridis colormap


def generate_channel_list(
    channel_count: int,
    units: bool = True,
) -> list[str]:
    """
    This function generates a list of channel names.

    :param channel_count: number of channels
    :param units: whether to include units
    """
    # Check for correct input types
    if not isinstance(channel_count, int):
        raise ValueError("channel_count must be an integer.")

    channel_list = []
    for channel in range(channel_count):
        if units:
            channel_list.append(("Channel " + str(channel + 1) + " [-]"))
        else:
            channel_list.append(("Channel " + str(channel + 1)))
    return channel_list


def plot_vae_results(
    input_ts: np.ndarray,
    output: np.ndarray | list[np.ndarray],
    scores: np.ndarray,
    channel_list: list[str] | None = None,
    sampling_rate: float | None = None,
    threshold: float | None = None,
    title: str | None = None,
    save_path: str | None = None,
    fixed_scale: bool = True,
    channel_scores: bool = False,
) -> None:
    """
    This function plots the input, output and anomaly scores of a stochastic variational autoencoder.

    :param input_ts: multivariate input time series
    :type input_ts: array (time steps, channels)
    :param output: list of three multivariate time series (mean, std, sample) or single multivariate time series (sample)
    :type output: list[array (time steps, channels), array (time steps, channels), array (time steps, channels)] or array (time steps, channels)
    :param scores: channel-wise anomaly scores or total anomaly score
    :type scores: array (time steps, channels) or array (time steps,)
    :param channel_list: list of channel names
    :type channel_list: list[str]
    :param sampling_rate: Sampling rate of the signal
    :type sampling_rate: int
    :param threshold: threshold for anomaly score
    :type threshold: float
    :param title: title of the plot
    :type title: str
    :param save_path: path to save plots in
    :type save_path: str
    :param fixed_scale: whether to use a fixed scale for the y-axis
    :type fixed_scale: bool
    :param channel_scores: whether to plot channel-wise anomaly scores
    :type channel_scores: bool
    :return: plot
    """
    # Check for correct input types
    if not isinstance(input_ts, np.ndarray):
        raise ValueError(
            "input_ts must be a numpy array of shape (time steps, channels)."
        )
    if not isinstance(output, list) and not isinstance(output, np.ndarray):
        raise ValueError(
            "output must be a list of three numpy arrays of shape (time steps, channels) or a numpy array of shape (time steps, channels)."
        )
    if not isinstance(scores, np.ndarray):
        raise ValueError(
            "scores must be a numpy array of shape (time steps, channels) or (time steps,)."
        )

    if channel_list is None:
        channel_list = generate_channel_list(input_ts.shape[1])

    if fixed_scale:
        y_range = [-8, 8]
    else:
        y_range = None

    try:
        score_total = np.sum(scores, axis=1)
    except Exception:
        score_total = scores

    if sampling_rate is None:
        time_axis = np.arange(input_ts.shape[0])
        x_axis_label = "Samples [-]"
    else:
        time_axis = np.arange(input_ts.shape[0]) / sampling_rate
        x_axis_label = "Time [s]"

    subplot_spec = [[{"secondary_y": True}]] * len(channel_list)
    subplot_spec.append([{"secondary_y": False}])
    fig = subplots.make_subplots(rows=len(channel_list) + 1, cols=1, specs=subplot_spec)
    for channel_idx in range(input_ts.shape[1]):
        # Plot input time series
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=input_ts[:, channel_idx],
                line=dict(color="#1f77b4"),
                name="Input",
            ),
            row=channel_idx + 1,
            col=1,
        )

        # Check if output is an array (deterministic output) or a list (mean, std, sample)
        if isinstance(output, list):
            if channel_idx < output[0].shape[-1]:
                # Plot mean of distribution
                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=output[0][:, channel_idx],
                        line=dict(color="#ff7f0e"),
                        name="Reconstruction",
                    ),
                    row=channel_idx + 1,
                    col=1,
                )
                # Plot standard deviation bands
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([time_axis, time_axis[::-1]]),
                        y=np.concatenate(
                            [
                                output[0][:, channel_idx] + output[1][:, channel_idx],
                                (output[0][:, channel_idx] - output[1][:, channel_idx])[
                                    ::-1
                                ],
                            ]
                        ),
                        line=dict(color="#ff7f0e"),
                        name="Reconstruction",
                        fill="toself",
                        opacity=0.5,
                    ),
                    row=channel_idx + 1,
                    col=1,
                )

        elif isinstance(output, np.ndarray):
            if channel_idx < output.shape[-1]:
                # Plot output (reconstruction)
                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=output[:, channel_idx],
                        line=dict(color="#ff7f0e"),
                        name="Reconstruction",
                    ),
                    row=channel_idx + 1,
                    col=1,
                )

        fig.update_xaxes(
            title_text=x_axis_label,
            row=channel_idx + 1,
            col=1,
            linecolor="black",
            showgrid=True,
            gridcolor="gray",
            gridwidth=0.3,
            zeroline=True,
            zerolinewidth=0.3,
            zerolinecolor="gray",
            mirror=True,
        )

        fig.update_yaxes(
            title_text=channel_list[channel_idx],
            row=channel_idx + 1,
            col=1,
            linecolor="black",
            showgrid=True,
            gridcolor="gray",
            gridwidth=0.3,
            zeroline=True,
            zerolinewidth=0.3,
            zerolinecolor="gray",
            mirror=True,
            range=y_range,
        )

        # Plot channel-wise anomaly score
        if (
            channel_scores
            and len(scores.shape) > 1
            and channel_idx < output[0].shape[-1]
        ):
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=scores[:, channel_idx],
                    line=dict(color="red"),
                    name="Anomaly Score",
                ),
                row=channel_idx + 1,
                col=1,
                secondary_y=True,
            )

            fig.update_yaxes(
                title_text="Channel Anomaly Score",
                title_font_color="red",
                row=channel_idx + 1,
                col=1,
                linecolor="black",
                showgrid=False,
                # gridcolor='gray',
                # gridwidth=0.3,
                zeroline=False,
                # zerolinewidth=0.3,
                # zerolinecolor='gray',
                showticklabels=False,
                mirror=False,
                range=[np.min(scores), np.max(scores)],
                secondary_y=True,
            )

    # Plot total anomaly score
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=score_total,
            line=dict(color="red"),
            name="Anomaly Score",
        ),
        row=len(channel_list) + 1,
        col=1,
    )

    # Draw threshold if it is provided
    if threshold is not None:
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=np.repeat(threshold, len(score_total)),
                line=dict(color="black", width=5, dash="dash"),
                name="Threshold",
            ),
            row=len(channel_list) + 1,
            col=1,
        )

    fig.update_xaxes(
        title_text=x_axis_label,
        row=len(channel_list) + 1,
        col=1,
        linecolor="black",
        showgrid=True,
        gridcolor="gray",
        gridwidth=0.3,
        zeroline=True,
        zerolinewidth=0.3,
        zerolinecolor="gray",
        mirror=True,
    )

    fig.update_yaxes(
        title_text="Total Anomaly Score",
        title_font_color="red",
        row=len(channel_list) + 1,
        col=1,
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
        height=500 * len(channel_list),
        showlegend=False,
        plot_bgcolor="white",
        title_text=title,
    )
    fig.update_xaxes(matches="x")

    if save_path is None:
        plotly.io.renderers.default = "browser"
        fig.show()
    else:
        try:
            os.mkdir(save_path)
        except Exception:
            pass
        if title is not None:
            plot_file_name = title[:-4] + ".html"
        else:
            now = datetime.now()
            date = now.strftime("%Y%m%d")
            clock_time = now.strftime("%H%M%S")
            plot_file_name = date + "_" + clock_time + ".html"
            time.sleep(1)
        fig.write_html(os.path.join(save_path, plot_file_name))


def plot_overlaid_ts(
    input_list: list[np.ndarray],
    channel_list: list | None = None,
    sampling_rate: int | None = None,
    title: str | None = "",
    save_path: str | None = "",
) -> None:
    """
    This function plots a list of multivariate time series overlaid on each other.

    :param input_list: list of multivariate time series
    :type input_list: list[array (time steps, channels)]
    :param channel_list: list of channel names
    :type channel_list: list[str]
    :param sampling_rate: Sampling rate of the signal
    :type sampling_rate: int
    :param title: title of the plot
    :type title: str
    :param save_path: directory to save plot in
    :type save_path: str
    :return:
    """

    if channel_list is None:
        channel_list = generate_channel_list(input_list[0].shape[1])

    fig = subplots.make_subplots(rows=len(channel_list), cols=1)
    for j, input_ts in enumerate(input_list):
        if sampling_rate is None:
            time_axis = np.arange(input_ts.shape[0])
            x_axis_label = "Samples [-]"
        else:
            time_axis = np.arange(input_ts.shape[0]) / sampling_rate
            x_axis_label = "Time [s]"

        if j == 0:
            colour = "#1f77b4"
        elif j == 1:
            colour = "#ff7f0e"
        else:

            def r():
                return random.randint(0, 255)

            colour = "#%02X%02X%02X" % (r(), r(), r())
        for channel_idx in range(input_ts.shape[-1]):
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=input_ts[:, channel_idx],
                    line=dict(color=colour),
                    name="Input Time Series" + str(j + 1),
                ),
                row=channel_idx + 1,
                col=1,
            )

            fig.update_xaxes(
                title_text=x_axis_label,
                row=channel_idx + 1,
                col=1,
                linecolor="black",
                showgrid=False,
                gridcolor="gray",
                gridwidth=0.3,
                zeroline=False,
                zerolinewidth=0.3,
                zerolinecolor="gray",
                mirror=True,
            )

            fig.update_yaxes(
                title_text=channel_list[channel_idx],
                row=channel_idx + 1,
                col=1,
                linecolor="black",
                showgrid=True,
                gridcolor="gray",
                gridwidth=0.3,
                zeroline=True,
                zerolinewidth=0.3,
                zerolinecolor="gray",
                mirror=True,
                # range=[-8, 8]
            )

    fig.update_layout(
        height=50 * len(channel_list),
        # width=1050,
        showlegend=False,
        plot_bgcolor="white",
        title_text=title,
    )
    fig.update_xaxes(matches="x")

    if not save_path:
        plotly.io.renderers.default = "browser"
        fig.show()
    else:
        try:
            os.mkdir(save_path)
        except Exception:
            pass
        if title:
            if title[-4:] == ".mf4":
                plot_file_name = title[:-4] + ".html"
            else:
                plot_file_name = title + ".html"
        else:
            now = datetime.now()
            date = now.strftime("%Y%m%d")
            clock_time = now.strftime("%H%M%S")
            plot_file_name = date + "_" + clock_time + ".html"
            time.sleep(1)
        fig.write_html(os.path.join(save_path, plot_file_name))


def plot_fancy_ts(
    input_ts: np.ndarray,
    output_ts: np.ndarray | None = None,
    channel_list: list | None = [],
    sampling_rate: int | None = None,
    title: str | None = "",
    save_path: str | None = "",
) -> None:
    """
    This function plots a multivariate time series in a fancy way.

    :param input_ts: input multivariate time series
    :param output_ts: output multivariate time series
    :param channel_list: list of channel names
    :param sampling_rate: Sampling rate of the signal
    :param title: title of the plot
    :param save_path: directory to save plot in
    """

    if channel_list is None:
        channel_list = generate_channel_list(input_ts.shape[1])

    n_full_rows = len(channel_list) // 2
    if len(channel_list) % 2 == 0:
        n_rows = n_full_rows
    else:
        n_rows = n_full_rows + 1

    fig = subplots.make_subplots(
        rows=n_rows, cols=2, vertical_spacing=0.02, horizontal_spacing=0.02
    )
    for channel_idx in range(len(channel_list)):
        # Left column
        if channel_idx % 2 == 0:
            col_idx = 1
            row_idx = channel_idx // 2 + 1
        # Right column
        else:
            col_idx = 2
            row_idx = channel_idx // 2 + 1

        if sampling_rate is None:
            time_axis = np.arange(input_ts.shape[0])
            if channel_idx >= len(channel_list) - 2:
                x_axis_label = "Samples [-]"
                show_tick_labels = True
            else:
                x_axis_label = None
                show_tick_labels = False
        else:
            time_axis = np.arange(input_ts.shape[0]) / sampling_rate
            if channel_idx >= len(channel_list) - 2:
                x_axis_label = "Time [s]"
                show_tick_labels = True
            else:
                x_axis_label = None
                show_tick_labels = False

        if output_ts is not None:
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=output_ts[:, channel_idx],
                    mode="lines",
                    line=dict(color="red", width=0.5),
                ),
                row=row_idx,
                col=col_idx,
            )

        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=input_ts[:, channel_idx],
                mode="lines",
                line=dict(color="black", width=0.5),
            ),
            row=row_idx,
            col=col_idx,
        )

        fig.update_xaxes(
            title_text=x_axis_label,
            row=row_idx,
            col=col_idx,
            linecolor="black",
            showgrid=False,
            gridcolor="gray",
            gridwidth=0.3,
            zeroline=False,
            zerolinewidth=0.3,
            zerolinecolor="gray",
            mirror=True,
            showticklabels=show_tick_labels,
            tickfont=dict(
                family="Times New Roman", size=10, color="black"
            ),  # Tick font properties
            title_font=dict(
                family="Times New Roman", size=12, color="black"
            ),  # Title font properties
        )

        fig.update_yaxes(
            title_text=channel_list[channel_idx],
            row=row_idx,
            col=col_idx,
            linecolor="black",
            showgrid=True,
            gridcolor="gray",
            gridwidth=0.3,
            zeroline=True,
            zerolinewidth=0.3,
            zerolinecolor="gray",
            mirror=True,
            side="right" if col_idx == 2 else "left",
            # range=[-8, 8]
            tickfont=dict(
                family="Times New Roman", size=10, color="black"
            ),  # Tick font properties
            title_font=dict(
                family="Times New Roman", size=12, color="black"
            ),  # Title font properties
            # showexponent='all',      # 'all', 'first', 'last', or 'none'
            # exponentformat='power'   # This turns 'e+2' into a superscript 10^2
        )

    fig.update_layout(
        height=50 * len(channel_list),
        showlegend=False,
        plot_bgcolor="white",
        title_text=title,
    )
    fig.update_xaxes(matches="x")

    if not save_path:
        plotly.io.renderers.default = "browser"
        fig.show()
    else:
        fig.write_image(save_path)


def plot_autocorrelation(
    series: np.ndarray,
    lags: int | None = 512,
    title: str | None = None,
) -> None:
    """
    This function plots the autocorrelation for each channel in a multivariate time series.

    :param series: multivariate input time series
    :type series: array (time steps, channels)
    :param lags: number of lags to plot
    :type lags: int
    :param title: title of the plot
    :type title: str
    :return:
    """

    fig = subplots.make_subplots(rows=series.shape[-1] + 1, cols=1)
    for channel in range(series.shape[-1]):
        corr_array = stattools.acf(series[:, channel], alpha=0.01, nlags=lags)
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
                    row=channel + 1,
                    col=1,
                )
                for x in range(len(corr))
            ]
            fig.add_scatter(
                x=np.arange(len(corr)),
                y=upper_y,
                mode="lines",
                line_color="rgba(255,255,255,0)",
                row=channel + 1,
                col=1,
            )
            fig.add_scatter(
                x=np.arange(len(corr)),
                y=lower_y,
                mode="lines",
                fillcolor="rgba(32, 146, 230,0.3)",
                fill="tonexty",
                line_color="rgba(255,255,255,0)",
                row=channel + 1,
                col=1,
            )
        except Exception:
            pass
        fig.update_traces(showlegend=False)
        fig.update_xaxes(range=[-1, lags])
        fig.update_yaxes(zerolinecolor="#000000")

        fig.update_layout(
            height=500 * series.shape[-1],
            showlegend=False,
            plot_bgcolor="white",
            title_text="Autocorrelation Plot for Cycle " + str(title),
            legend_tracegroupgap=20 * series.shape[-1],
        )
        fig.update_xaxes(matches="x")
    fig.show()


def plot_vae_loss(
    series: np.ndarray,
    save_path: str | None = None,
) -> None:
    """
    This function plots the loss against the number of epochs needed for training.
    :param series: array of losses computed during model training
    :type series: array (time steps, number_of_losses)
    :param save_path: path to save the plot
    :type save_path: str
    :return:
    """

    fig = go.Figure()
    for loss_type_idx in range(series.shape[1]):
        fig.add_scatter(
            x=np.arange(series.shape[0]), y=series[:, loss_type_idx], mode="lines"
        )

    if not save_path:
        plotly.io.renderers.default = "browser"
        fig.show()
    else:
        try:
            os.mkdir(save_path)
        except Exception:
            pass
        plot_file_name = "losses.html"
        fig.write_html(os.path.join(save_path, plot_file_name))


def plot_active_learning(
    sheet_path: str,
    subplot_titles: list[str] = [],
    mode: str = "",
    save_path: str = "",
    save_type: str = "",
) -> None:
    """
    This function plots the results of active learning experiments from an Excel sheet.

    :param sheet_path: path to the Excel sheet containing the results
    :param subplot_titles: list of titles for each subplot
    :param mode: type of experiment, either 'budget' or 'mislabelling'
    :param save_path: path to save the plot to
    :param save_type: format to save the plot, either 'svg' or 'html'
    """

    time_axis = [1, 7, 14, 21, 28]

    # Get the sheet names
    sheet_names = pd.ExcelFile(sheet_path).sheet_names

    df_dict = {}
    for sheet in sheet_names:
        if mode == "budget":
            if (
                str(sheet).endswith("_10")
                or str(sheet).endswith("_20")
                or str(sheet).endswith("_30")
            ):
                pass
            else:
                df_dict[sheet] = pd.read_excel(
                    sheet_path,
                    header=0,
                    usecols=[0, 1, 2, 3, 4, 5, 6, 7],
                    sheet_name=sheet,
                )
        elif mode == "mislabelling":
            if str(sheet).endswith("_0"):
                pass
            else:
                df_dict[sheet] = pd.read_excel(
                    sheet_path,
                    header=0,
                    usecols=[0, 1, 2, 3, 4, 5, 6, 7],
                    sheet_name=sheet,
                )

    upper_baseline = df_dict["best"].groupby("Split")["F1"].mean().values
    lower_baseline = df_dict["unsupervised"].groupby("Split")["F1"].mean().values

    fig = subplots.make_subplots(
        rows=len(subplot_titles),
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.07,
    )

    for results_idx in range(len(subplot_titles)):
        # Add a helper trace for the y=1 line
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=[0.7] * len(time_axis),  # A constant y=1 for all x values
                line=dict(color="rgba(0, 0, 0, 0)"),  # Invisible line
                showlegend=False,
            ),
            row=results_idx + 1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=upper_baseline,
                line=dict(color="gray"),
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(0, 0, 0, 0.2)",  # Adjust the alpha value for transparency
            ),
            row=results_idx + 1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=lower_baseline,
                line=dict(color="gray"),
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(0, 0, 0, 0.2)",  # Adjust the alpha value for transparency
            ),
            row=results_idx + 1,
            col=1,
        )

        approach_list = ["rand", "unc", "top", "ds"]
        approach_name_list = ["random", "uncertainty", "top", "disimilarity"]
        colour_list_idcs = [
            round(i * (len(viridis_colors) - 1) / (len(approach_list) - 1))
            for i in range(len(approach_list))
        ]

        for approach_idx, approach in enumerate(approach_list):
            combination = []
            for sheet in [str(key) for key in df_dict.keys()]:
                if approach in sheet:
                    combination.append(sheet)

            df_temp = df_dict[combination[results_idx]]
            df_mean = df_temp.groupby("Split")["F1"].mean()

            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=df_mean,
                    line=dict(color=viridis_colors[colour_list_idcs[approach_idx]]),
                    name=approach_name_list[approach_idx],
                ),
                row=results_idx + 1,
                col=1,
            )

        if results_idx == len(subplot_titles) - 1:
            x_axis_label = "Time [days]"
        else:
            x_axis_label = ""

        fig.update_xaxes(
            linecolor="black",
            showgrid=False,
            mirror=True,
            range=[1, 28],
            title_text=x_axis_label,
            row=results_idx + 1,
            col=1,
        )

        fig.update_yaxes(
            linecolor="black",
            showgrid=True,
            gridcolor="black",
            mirror=True,
            range=[0, 0.7],
            title_text="F_1",
            row=results_idx + 1,
            col=1,
        )

        fig.update_layout(
            showlegend=True,
            plot_bgcolor="white",
            font=dict(size=20, family="Times New Roman", color="black"),
            height=1500,
            width=1500,
            legend=dict(
                x=0.4,  # Horizontal position (0=left, 1=right)
                y=0.8,  # Vertical position (0=bottom, 1=top)
                bgcolor="rgba(255, 255, 255, 1)",  # Background color with transparency
            ),
        )

    fig.update_annotations(font=dict(size=20, family="Times New Roman", color="black"))

    fig.show()
    if save_path is not None:
        fig.write_image(os.path.join(save_path, mode + "." + save_type))
