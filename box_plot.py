import logging
import os
import pathlib
from typing import Sequence, Union

import pandas as pd
from plotly import graph_objects as go


logging.basicConfig(level=logging.DEBUG)

HERE = pathlib.Path(__file__).parent
pd.options.display.float_format = "{:,.5f}".format

# Define custom colors
SECTOR_COLOR_MAP = {
    "A": "blue",
    "B": "red",
    "C": "goldenrod",
    "D": "purple",
    "E": "green",
}


def _create_figure_layout(
    fig_title: str,
    x_title: str,
    y_title: str,
    sector_positions: dict,
    min_x: float,
    max_x: float,
    major_ytick_step: float,
    minor_ytick_step: float,
    format_y_as_percentage: bool,
) -> dict:
    """Update the layout of the figure with titles, axes, grid, and formatting."""
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
            range=[min_x, max_x],  # Adjust x-axis range to include all data
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
    fig: go.Figure, df: pd.DataFrame, sector_positions: dict
) -> None:
    """Add box and scatter traces for each sector to the figure."""
    box_width = 0.4
    spacing = 0.05
    delta = (box_width / 2) + spacing

    for sector, position in sector_positions.items():
        x_label = position
        box_x = [x_label + delta] * len(df)
        scatter_x = [x_label - delta] * len(df)

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
    Create a box plot of sector values with optional global average line.

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
        The created plotly figure.
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
            fig_title,
            x_title,
            y_title,
            sector_positions,
            min_x,
            max_x,
            major_ytick_step,
            minor_ytick_step,
            format_y_as_percentage,
        )
    )

    if display_average and existing_sectors:
        global_avg = df[existing_sectors].values.flatten().mean()

        fig.add_hline(
            y=global_avg,
            line_color="brown",
            annotation_text=f"Average: {global_avg:.2%}"
            if format_y_as_percentage
            else f"Average: {global_avg:.5f}",
            annotation_position="top right",
            line_width=2,
            annotation_font_size=18,  # Increase font size as desired
            annotation_font_color="brown",
        )

    logging.debug("Figure created")

    if show:
        fig.show(renderer="browser")
    return fig


def _make_relative_to_col(df: pd.DataFrame, col_label: str = "A") -> pd.DataFrame:
    cols_to_adjust = [s for s in SECTOR_COLOR_MAP if s in df.columns]
    df_rel_to_col = df.copy(deep=True)
    df_rel_to_col[cols_to_adjust] = (
        df[cols_to_adjust].div(df[col_label], axis=0).subtract(1).astype(float)
    )
    return df_rel_to_col


def load_and_prepare_data(
    data_path: pathlib.Path, sector_cols: Sequence[str]
) -> pd.DataFrame:
    """
    Load and prepare the dataset from a CSV file.

    Reads the CSV, applies data types, and returns the resulting DataFrame.

    Parameters
    ----------
    data_path : pathlib.Path
        Path to the CSV file containing the data.
    sector_cols : Sequence[str]
        List of sector column names to be interpreted as floats.

    Returns
    -------
    pd.DataFrame
        Loaded and type-cast data frame.
    """
    dtypes_by_col = {
        "File": str,
        "RTImageLabel": str,
        "AcquisitionDate": str,
        **{sector: float for sector in sector_cols},
        "Average": float,
        "Max vs Min": float,
    }

    df = pd.read_csv(data_path).astype(dtypes_by_col)
    return df


def plot_raw_results(
    df: pd.DataFrame, df_planned: pd.DataFrame, data_path: pathlib.Path
) -> None:
    """
    Generate plots of mean CU (Calibration Units) values in sector ROIs,
    as well as their relative differences compared to planned values.

    Creates:
    - A plot showing mean CU per sector ROI.
    - A plot showing mean CU per sector ROI relative to the planned mean CU.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame containing the main dataset, including sector ROI CU values.
    df_planned : pd.DataFrame
        Data frame containing planned sector CU values for comparison.
    data_path : pathlib.Path
        Path used for saving the plots if needed.
    """

    sector_cols = [s for s in SECTOR_COLOR_MAP]
    df_unnormed = df.loc[df["ImageType"] == "DRCS"].copy(deep=True)

    # Raw
    box_plot_sectors(
        df_unnormed,
        fig_title="Mean CU Per Sector)",
        y_title="Mean CU in ROI",
        major_ytick_step=0.0005,
        minor_ytick_step=0.0001,
        display_average=True,
    )

    # Relative to planned
    df_unnormed_rel_to_plan = df_unnormed.copy(deep=True)
    df_unnormed_rel_to_plan[sector_cols] = (
        df_unnormed_rel_to_plan[sector_cols]
        .div(df_planned[sector_cols].loc[0])
        .subtract(1)
        .astype(float)
    )
    box_plot_sectors(
        df_unnormed_rel_to_plan,
        fig_title="Mean CU Per Sector Relative to Planned Values",
        y_title="Mean CU in ROI relative to mean planned value in ROI",
        format_y_as_percentage=True,
        display_average=True,
    )

    # # Relative to first acquired image
    # df_unnormed_rel_to_first = df_unnormed.copy(deep=True).reset_index(drop=True)
    # df_unnormed_rel_to_first.loc[1:, sector_cols] = (
    #     df_unnormed_rel_to_first.loc[1:, sector_cols]
    #     .div(df_unnormed_rel_to_first.loc[0, sector_cols])
    #     .subtract(1)
    #     .astype(float)
    # )
    # box_plot_sectors(
    #     df_unnormed_rel_to_first.loc[1:],
    #     fig_title="Mean CU Per Sector Relative to Acquired Baseline",
    #     y_title="Mean ROI pixel intensity relative to first acquired image",
    #     format_y_as_percentage=True,
    #     display_average=True,
    # )


def plot_normalized_results(df: pd.DataFrame, data_path: pathlib.Path) -> None:
    """
    Generate plots of mean CU values in sector ROIs that have been normalized by a
    circular reference field.

    Creates:
    - A plot showing mean CU per sector ROI normalized by the circle field CU (i.e.,
    Mean CU in DR-CS ROI / Mean CU in Circle Field ROI).

    Parameters
    ----------
    df : pd.DataFrame
        Data frame containing the main dataset, including normalized sector ROI CU values.
    data_path : pathlib.Path
        Path used for saving the plots if needed.
    """
    df_normed = df.loc[df["ImageType"] == "NORMALIZED"].copy(deep=True)

    # Normalized by output
    box_plot_sectors(
        df_normed,
        fig_title="Mean CU Per Sector Normalised by Circle Field",
        y_title="Mean CU in DR-CS ROI / Mean CU in Circle Field ROI",
        major_ytick_step=0.0005,
        minor_ytick_step=0.0001,
        display_average=True,
    )

    # # Relative to first acquired image (normalized)
    # sector_cols = [s for s in SECTOR_COLOR_MAP]
    # df_normed_rel_to_first = df_normed.copy(deep=True).reset_index(drop=True)
    # df_normed_rel_to_first.loc[1:, sector_cols] = (
    #     df_normed_rel_to_first.loc[1:, sector_cols]
    #     .div(df_normed_rel_to_first.loc[0, sector_cols])
    #     .subtract(1)
    #     .astype(float)
    # )
    # box_plot_sectors(
    #     df_normed_rel_to_first.loc[1:],
    #     fig_title="Sector ROI Means (Normalised) Relative to Acquired Baseline Image",
    #     y_title="Mean ROI pixel intensity normalised by output relative to first acquired image",
    #     format_y_as_percentage=True,
    #     display_average=True,
    # )


def plot_relative_to_sector_or_average(
    df: pd.DataFrame, data_path: pathlib.Path, sector_norm: str = "A"
) -> None:
    """
    Generate plots of sector ROI means relative to a specified sector or the average.

    If sector_norm is a sector label, values are plotted relative to that sector.
    If sector_norm is "Average", values are plotted relative to the mean of all sectors.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame containing the main dataset.
    data_path : pathlib.Path
        Path used for saving the plots if needed.
    sector_norm : str, optional
        Sector to use as a reference, or "Average" for mean reference, by default "A".
    """
    df_rel_to_col = _make_relative_to_col(
        df.loc[df["ImageType"] == "DRCS"], col_label=sector_norm
    )
    sectors_to_plot = [
        sector
        for sector in SECTOR_COLOR_MAP
        if sector != sector_norm and sector in df_rel_to_col.columns
    ]

    title = f"Mean CU Per Sector Relative to Sector {sector_norm}"
    y_title = f"Mean CU in ROI relative to mean CU in Sector {sector_norm} ROI"
    if sector_norm == "Average":
        title = "Mean CU Per Sector Relative to Average of Sector Means"
        y_title = "Mean CU in ROI relative to average of mean CU values across all sector ROIs"

    box_plot_sectors(
        df_rel_to_col[sectors_to_plot],
        fig_title=title,
        y_title=y_title,
        format_y_as_percentage=True,
        display_average=False if sector_norm == "Average" else True,
    )


def main(data_path: Union[pathlib.Path, None] = None) -> None:
    """
    The main entry point of the script.

    This function:
    - Loads the dataset from the provided path (or prompts the user if no path is given).
    - Generates and displays/saves plots showing:
        * Mean CU per sector ROI, both absolute and relative to planned values.
        * Mean CU per sector ROI normalized by a reference circle field.
        * Mean CU per sector ROI relative to a chosen reference sector or to the
          average of all sectors.

    Parameters
    ----------
    data_path : Union[pathlib.Path, None], optional
        Path to the CSV file containing the dataset. If None, the user is prompted to
        enter a path.
    """
    if data_path is None:
        data_path = pathlib.Path(
            input("Please enter the full path to the 'roi_stats.csv' file: ")
        )

    sector_labels = list(SECTOR_COLOR_MAP)
    DTYPES_BY_COL = {
        "File": str,
        "RTImageLabel": str,
        "AcquisitionDate": str,
        **{sector: float for sector in sector_labels},
        "Average": float,
        "Max vs Min": float,
    }

    df_planned = pd.DataFrame.from_dict(
        {
            "File": ["RI.1.2.246.352.71.3.581896633164.50047.20240930042904"],
            "RTImageLabel": ["DRCS PLANNED"],
            "AcquisitionDate": ["20240930"],
            "A": [0.043024831],
            "B": [0.043044262],
            "C": [0.043095744],
            "D": [0.043080568],
            "E": [0.043016632],
            "Average": [0.043052408],
            "Max vs Min": [0.001839102],
        }
    ).astype(DTYPES_BY_COL)

    df = load_and_prepare_data(data_path, sector_labels)

    # Plot various results
    plot_raw_results(df, df_planned, data_path)
    plot_normalized_results(df, data_path)
    # plot_relative_to_sector_or_average(df, data_path, sector_norm="A")
    plot_relative_to_sector_or_average(df, data_path, sector_norm="Average")


if __name__ == "__main__":
    main()
