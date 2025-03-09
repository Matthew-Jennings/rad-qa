# Copyright (C) 2024 Matthew Jennings
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
box_plot.py

This module provides functionality for reading data from a CSV file that contains
ROI (Region Of Interest) statistics for DRCS (Dynamic Range Calibration System) images,
then generates various box and scatter plots using Plotly to visualize calibration unit (CU)
values across multiple sectors. It can also compute relative differences (e.g., normalized to
a particular sector or to a global mean) and plot those differences.

Usage:
    python box_plot.py /path/to/roi_stats.csv [--sector-norm A|B|C|D|E|Average]

Overview of main components:
    - SECTOR_COLOR_MAP: Dict specifying which color to use for plotting each sector.
    - DTYPES_BY_COL: Dict specifying the data types for each CSV column.
    - load_and_prepare_data: Loads and type-casts the CSV into a pandas DataFrame.
    - box_plot_sectors: Generates a box plot (and optional scatter plot) for sector data.
    - plot_raw_results, plot_normalized_results, plot_relative_to_sector_or_average:
      Higher-level routines that leverage `box_plot_sectors` to produce specific plots.
    - main: Entry point for command-line usage.

Author: Matthew Jennings
"""

import argparse
import logging
import pathlib
import sys
from typing import Dict

import pandas as pd
from plotly import graph_objects as go

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

#: Dict[str, str]: A mapping from sector labels (e.g. 'A', 'B') to their corresponding plot color.
SECTOR_COLOR_MAP: Dict[str, str] = {
    "A": "blue",
    "B": "red",
    "C": "goldenrod",
    "D": "purple",
    "E": "green",
}

#: Dict[str, type]: A mapping of CSV column names to their intended Python data types.
DTYPES_BY_COL: Dict[str, type] = {
    "File": str,
    "RTImageLabel": str,
    "AcquisitionDate": str,
    **{sector: float for sector in SECTOR_COLOR_MAP},
    "Average": float,
    "Max vs Min": float,
}

# Configure pandas display options
pd.options.display.float_format = "{:,.5f}".format


def _create_figure_layout(
    fig_title: str,
    x_title: str,
    y_title: str,
    sector_positions: Dict[str, float],
    min_x: float,
    max_x: float,
    major_ytick_step: float,
    minor_ytick_step: float,
    format_y_as_percentage: bool,
) -> dict:
    """
    Create layout configuration for the Plotly figure.

    Parameters
    ----------
    fig_title : str
        Title of the figure.
    x_title : str
        Label for the x-axis.
    y_title : str
        Label for the y-axis.
    sector_positions : dict
        A mapping from sector label to a numeric position on the x-axis.
    min_x : float
        Minimum x-axis value to display.
    max_x : float
        Maximum x-axis value to display.
    major_ytick_step : float
        Step size for major y-axis ticks.
    minor_ytick_step : float
        Step size for minor y-axis ticks.
    format_y_as_percentage : bool
        Whether to format the y-axis ticks as percentages.

    Returns
    -------
    dict
        A dictionary of layout configurations suitable for Plotly's `Figure.update_layout()`.
    """
    ytickformat = ".1%" if format_y_as_percentage else None
    return dict(
        title=dict(
            text=fig_title,
            x=0.5,
            xanchor="center",
            yanchor="top",
            font=dict(family="Arial", size=40, color="black"),
        ),
        xaxis=dict(
            title=dict(text=x_title, font=dict(family="Arial", size=32, color="black")),
            tickmode="array",
            tickvals=list(sector_positions.values()),
            ticktext=list(sector_positions.keys()),
            tickfont=dict(family="Arial", size=24, color="black"),
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            range=[min_x, max_x],
        ),
        yaxis=dict(
            title=dict(
                text=y_title,
                font=dict(family="Arial", size=32, color="black"),
            ),
            tickfont=dict(family="Arial", size=24, color="black"),
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            gridcolor="black",
            zeroline=True,
            zerolinewidth=1.5,
            zerolinecolor="black",
            dtick=major_ytick_step,
            minor=dict(
                showgrid=True,
                gridwidth=0.5,
                gridcolor="lightgrey",
                dtick=minor_ytick_step,
            ),
            tickformat=ytickformat,
        ),
        plot_bgcolor="white",
        showlegend=False,
        margin=dict(l=40, r=40, t=80, b=80),
        width=1600,
        height=1200,
    )


def _add_sectors_to_figure(
    fig: go.Figure, df: pd.DataFrame, sector_positions: Dict[str, float]
) -> None:
    """
    Add box and scatter traces for each sector to the Plotly figure.

    Parameters
    ----------
    fig : go.Figure
        The Plotly Figure object to update.
    df : pd.DataFrame
        DataFrame containing sector columns with numeric values.
    sector_positions : dict
        Mapping of sector labels to numeric positions on the x-axis.

    Returns
    -------
    None
        The function modifies the Figure object in place.
    """
    box_width = 0.4
    spacing = 0.05
    delta = (box_width / 2) + spacing

    for sector, position in sector_positions.items():
        box_x = [position + delta] * len(df)
        scatter_x = [position - delta] * len(df)

        # Box plot
        fig.add_trace(
            go.Box(
                y=df[sector],
                x=box_x,
                name=sector,
                marker_color=SECTOR_COLOR_MAP[sector],
                boxpoints=False,
                hoverinfo="y",
                width=box_width,
                line=dict(width=1.5),
                showlegend=False,
            )
        )

        # Scatter plot of individual data points
        fig.add_trace(
            go.Scatter(
                y=df[sector],
                x=scatter_x,
                mode="markers",
                marker=dict(color=SECTOR_COLOR_MAP[sector], size=9, opacity=0.7),
                showlegend=False,
                hoverinfo="y",
            )
        )


def box_plot_sectors(
    df: pd.DataFrame,
    fig_title: str,
    y_title: str,
    x_title: str = "Sector",
    minor_ytick_step: float = 0.001,
    major_ytick_step: float = 0.005,
    format_y_as_percentage: bool = False,
    display_average: bool = False,
    show: bool = True,
) -> go.Figure:
    """
    Create a box plot of sector values with an optional global average line.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing sector columns.
    fig_title : str
        The title of the figure.
    y_title : str
        The y-axis label.
    x_title : str, default="Sector"
        The x-axis label.
    minor_ytick_step : float, default=0.001
        The step size for minor y-axis ticks.
    major_ytick_step : float, default=0.005
        The step size for major y-axis ticks.
    format_y_as_percentage : bool, default=False
        If True, format y-ticks as percentages.
    display_average : bool, default=False
        If True, display a horizontal line at the global average.
    show : bool, default=True
        If True, display the figure in the browser.

    Returns
    -------
    go.Figure
        The created Plotly figure.
    """
    fig = go.Figure()

    # Determine which sectors are present
    existing_sectors = [sector for sector in SECTOR_COLOR_MAP if sector in df.columns]
    sector_positions = {s: i + 1 for i, s in enumerate(existing_sectors)}

    _add_sectors_to_figure(fig, df, sector_positions)

    # Calculate x-axis range
    box_width = 0.4
    spacing = 0.05
    delta = (box_width / 2) + spacing
    min_x = min(sector_positions.values()) - (delta + 0.5)
    max_x = max(sector_positions.values()) + (delta + 0.5)

    fig.update_layout(
        **_create_figure_layout(
            fig_title=fig_title,
            x_title=x_title,
            y_title=y_title,
            sector_positions=sector_positions,
            min_x=min_x,
            max_x=max_x,
            major_ytick_step=major_ytick_step,
            minor_ytick_step=minor_ytick_step,
            format_y_as_percentage=format_y_as_percentage,
        )
    )

    if display_average and existing_sectors:
        global_avg = df[existing_sectors].values.flatten().mean()
        annotation_text = (
            "Average: %.2f%%" if format_y_as_percentage else "Average: %.5f"
        )
        annotation_args = (
            (global_avg * 100,) if format_y_as_percentage else (global_avg,)
        )

        fig.add_hline(
            y=global_avg,
            line_color="brown",
            annotation_text=annotation_text % annotation_args,
            annotation_position="top right",
            line_width=2,
            annotation_font_size=18,
            annotation_font_color="brown",
        )

    logging.info(
        "Figure '%s' created with sectors: %s",
        fig_title,
        ", ".join(existing_sectors),
    )

    if show:
        fig.show(renderer="browser")
    return fig


def _make_relative_to_col(df: pd.DataFrame, col_label: str = "A") -> pd.DataFrame:
    """
    Adjust DataFrame values so that each row's sector columns are relative to a specified column.

    Specifically, the sector columns in each row are divided by the value in `col_label`
    (for that row), then 1 is subtracted to get the relative difference.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to adjust.
    col_label : str, default="A"
        Name of the sector column to use as the reference.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the sector columns adjusted.
    """
    cols_to_adjust = [s for s in SECTOR_COLOR_MAP if s in df.columns]
    df_rel_to_col = df.copy(deep=True)
    df_rel_to_col[cols_to_adjust] = (
        df[cols_to_adjust].div(df[col_label], axis=0).subtract(1).astype(float)
    )
    return df_rel_to_col


def load_and_prepare_data(data_path: pathlib.Path) -> pd.DataFrame:
    """
    Load and prepare the dataset from a CSV file.

    Reads the CSV, applies data types, and returns the resulting DataFrame.

    Parameters
    ----------
    data_path : pathlib.Path
        Path to the CSV file containing the data.

    Returns
    -------
    pd.DataFrame
        Loaded and type-cast data frame.
    """
    try:
        df = pd.read_csv(data_path).astype(DTYPES_BY_COL)
        logging.info("Data loaded successfully from '%s'.", data_path)
        return df
    except Exception as e:
        logging.error("Failed to load data from '%s': %s", data_path, e)
        raise


def plot_raw_results(df: pd.DataFrame) -> None:
    """
    Generate plots of mean CU (Calibration Units) values in sector ROIs,
    as well as their relative differences compared to planned values.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame containing the main dataset, including sector ROI CU values and
        planned data.
    """
    sector_cols = list(SECTOR_COLOR_MAP)
    df_unnormed = df.loc[df["ImageType"] == "DRCS"].copy(deep=True)

    # Raw
    box_plot_sectors(
        df_unnormed,
        fig_title="Mean CU Per Sector",
        y_title="Mean CU in ROI",
        major_ytick_step=0.0005,
        minor_ytick_step=0.0001,
        display_average=True,
    )
    df_rel_to_global_mean = df_unnormed.copy(deep=True)

    df_rel_to_global_mean[sector_cols] = (
        df_unnormed[sector_cols]
        .div(df_unnormed[sector_cols].mean().mean())
        .subtract(1)
        .astype(float)
    )

    # Relative to global mean
    box_plot_sectors(
        df_rel_to_global_mean,
        fig_title=None,
        y_title="Mean CU in ROI relative to mean of all sectors across all acquired images",
        major_ytick_step=0.005,
        minor_ytick_step=0.001,
        format_y_as_percentage=True,
        display_average=False,
    )

    # Relative to planned
    df_planned = (
        df.loc[df["ImageType"] == "DRCS PREDICTED"].copy(deep=True).reset_index()
    )
    if df_planned.empty:
        logging.error("No planned data found with ImageType 'DRCS PREDICTED'.")
        sys.exit(1)
    elif len(df_planned) > 1:
        logging.error(
            "Multiple planned data rows found with ImageType 'DRCS PREDICTED'. "
            "Please ensure only one exists."
        )
        sys.exit(1)
    else:
        planned_row = df_planned.loc[0]
        logging.debug("Planned data found:\n%s", planned_row)

    df_rel_to_plan = df_unnormed.copy(deep=True)
    df_rel_to_plan[sector_cols] = (
        df_rel_to_plan[sector_cols]
        .div(planned_row[sector_cols], axis=1)
        .subtract(1)
        .astype(float)
    )
    box_plot_sectors(
        df_rel_to_plan,
        fig_title=None,
        y_title="Mean CU in sector ROI relative to corresponding mean PDIP value",
        format_y_as_percentage=True,
        display_average=True,
    )


def plot_normalized_results(df: pd.DataFrame) -> None:
    """
    Generate plots of mean CU values in sector ROIs that have been normalized by a
    circular reference field.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame containing the main dataset, including normalized sector ROI CU
        values.
    """
    df_normed = df.loc[df["ImageType"] == "NORMALIZED"].copy(deep=True)

    # Normalized by output
    box_plot_sectors(
        df_normed,
        fig_title="Mean CU Per Sector Normalized by Circle Field",
        y_title="Mean CU in DR-CS ROI / Mean CU in Circle Field ROI",
        major_ytick_step=0.0005,
        minor_ytick_step=0.0001,
        display_average=True,
    )


def plot_relative_to_sector_or_average(
    df: pd.DataFrame, sector_norm: str = "A"
) -> None:
    """
    Generate plots of sector ROI means relative to a specified sector or the average.

    If sector_norm is a sector label, values are plotted relative to that sector.
    If sector_norm is "Average", values are plotted relative to the mean of all sectors.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame containing the main dataset.
    sector_norm : str, optional
        Sector to use as a reference, or "Average" for mean reference, by default "A".
    """
    df_unnormed = df.loc[df["ImageType"] == "DRCS"].copy(deep=True)

    if sector_norm == "Average":
        df_rel_to_col = df_unnormed.copy(deep=True)
        sector_cols = list(SECTOR_COLOR_MAP)
        mean_values = df_rel_to_col[sector_cols].mean(axis=1)
        df_rel_to_col[sector_cols] = (
            df_rel_to_col[sector_cols]
            .div(mean_values, axis=0)
            .subtract(1)
            .astype(float)
        )
    else:
        df_rel_to_col = _make_relative_to_col(df_unnormed, col_label=sector_norm)

    sectors_to_plot = [
        sector
        for sector in SECTOR_COLOR_MAP
        if sector != sector_norm and sector in df_rel_to_col.columns
    ]

    if sector_norm == "Average":
        title = None
        y_title = (
            "Mean CU in ROI relative to average of across all sectors in same image"
        )
        display_average = False  # Average line not needed when normalized to average
    else:
        title = f"Mean CU Per Sector Relative to Sector {sector_norm}"
        y_title = f"Mean CU in ROI relative to mean CU in Sector {sector_norm} ROI"
        display_average = True

    box_plot_sectors(
        df_rel_to_col[sectors_to_plot],
        fig_title=title,
        y_title=y_title,
        format_y_as_percentage=True,
        display_average=display_average,
    )


def main():
    """
    Main entry point of the script with CLI implementation.

    The script expects a path to a CSV file containing ROI statistics. It generates a set of
    box plots (and some scatter overlay) that can be used to analyze calibration unit values
    across multiple sectors, along with optional normalization options.

    Command-line Arguments
    ----------------------
    data_path : pathlib.Path
        Path to the 'roi_stats.csv' data file.
    --sector-norm : str
        Sector to normalize against or 'Average' for mean normalization (default is 'Average').

    Example
    -------
    python box_plot.py /path/to/roi_stats.csv --sector-norm A
    """
    parser = argparse.ArgumentParser(
        description="Generate box plots for sector ROI Calibration Units (CU) data."
    )
    parser.add_argument(
        "data_path",
        type=pathlib.Path,
        help="Path to the 'roi_stats.csv' data file.",
    )
    parser.add_argument(
        "--sector-norm",
        type=str,
        default="Average",
        choices=["A", "B", "C", "D", "E", "Average"],
        help="Sector to normalize against or 'Average' for mean normalization.",
    )
    args = parser.parse_args()

    data_path: pathlib.Path = args.data_path
    sector_norm: str = args.sector_norm

    # Load main data
    df = load_and_prepare_data(data_path)

    # Generate plots
    plot_raw_results(df)
    plot_normalized_results(df)
    plot_relative_to_sector_or_average(df, sector_norm=sector_norm)

    logging.info("All plots generated successfully.")


if __name__ == "__main__":
    main()
