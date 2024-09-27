import pathlib
from pprint import pprint
from typing import Sequence

import pandas as pd
from plotly import express as px, graph_objects as go


HERE = pathlib.Path(__file__).parent

pd.options.display.float_format = "{:,.5f}".format

# Define custom colors
# fmt: off
SECTOR_COLOR_MAP = {
    "A": "blue",  
    "B": "red",
    "C": "goldenrod", 
    "D": "purple",
    "E": "green"
}
# fmt: on


def box_plot_sectors(
    df: pd.DataFrame, fig_title: str, y_title: str, x_title: str = "Sector"
):
    # Create an empty figure
    fig = go.Figure()

    # Map sectors to numerical positions for custom x-axis
    sector_positions = {}
    last_pos = 0
    for sector in SECTOR_COLOR_MAP:
        if sector in df.columns:
            sector_positions[sector] = last_pos + 1
            last_pos += 1

    # Define delta for positioning box and scatter plots
    box_width = 0.4  # Width of the box plot
    spacing = 0.05  # Desired spacing between box/scatter and x_label
    delta = (box_width / 2) + spacing

    # Loop through sectors and add box and scatter traces
    for sector in sector_positions.keys():
        # Central position for the sector label
        x_label = sector_positions[sector]

        # Positions for the box and scatter plots
        box_x = [x_label + delta] * len(df)
        scatter_x = [x_label - delta] * len(df)

        # Box plot with whiskers covering the full data range
        fig.add_trace(
            go.Box(
                y=df[sector],
                x=box_x,
                name=sector,
                marker_color=SECTOR_COLOR_MAP[sector],
                boxpoints=False,  # Do not show individual points
                hoverinfo="y",
                width=box_width,
                line=dict(width=1.5),
                showlegend=False,
            )
        )

        # Scatter plot of individual data points to the left of the box plot
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

    # Calculate the x-axis range
    min_x = min(sector_positions.values()) - (delta + 0.5)
    max_x = max(sector_positions.values()) + (delta + 0.5)

    # Update layout for a scientific and formal appearance
    fig.update_layout(
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
            linewidth=1,
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
            linewidth=1,
            linecolor="black",
            mirror=True,
            gridcolor="lightgrey",
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="grey",
        ),
        plot_bgcolor="white",
        showlegend=False,
        margin=dict(l=40, r=40, t=80, b=80),
        width=1600,
        height=1200,
    )

    fig.show()


def box_plot_sectors_express(df: pd.DataFrame) -> None:
    df_melted = df.melt(var_name="Sector", value_name="CU")

    fig = px.box(df_melted, x="Sector", y="CU", points="all", color="Sector")
    fig.show()


def norm_df_by_col(df: pd.DataFrame, col_label: str, drop_col=False) -> pd.DataFrame:
    cols_to_divide = [
        sector for sector in SECTOR_COLOR_MAP.keys() if col_label != sector
    ]
    df_normed_by_col = df.copy(deep=True)
    df_normed_by_col[cols_to_divide] = df[cols_to_divide].div(df[col_label], axis=0)
    if drop_col:
        df_normed_by_col.drop(columns=col_label, inplace=True)
    return df_normed_by_col


def make_relative_to_col(df, col_label="A"):
    cols_to_adjust = [
        sector
        for sector in SECTOR_COLOR_MAP.keys()  # if col_label != sector
    ]
    df_rel_to_col = df.copy(deep=True)
    df_rel_to_col[cols_to_adjust] = 100 * (
        df[cols_to_adjust].div(df[col_label], axis=0) - 1
    )
    return df_rel_to_col


if __name__ == "__main__":
    data_source = "Karolinska"
    data_path = HERE / "pac-man-karolinska-results.csv"

    df = pd.read_csv(data_path)
    df_normed = norm_df_by_col(df, "Ropen", drop_col=False)
    df_normed["Source"] = data_source

    box_plot_sectors(
        df_normed,
        fig_title="Sector ROI means normalised by linac output",
        y_title="Mean pixel intensity normalised by linac output",
    )

    sector_norm = "A"
    df_normed_rel_to_col = make_relative_to_col(df_normed, col_label=sector_norm)

    sectors_to_plot = [
        sector for sector in SECTOR_COLOR_MAP.keys() if sector != sector_norm
    ]
    # box_plot_sectors(
    #     df_normed_rel_to_col[sectors_to_plot],
    #     fig_title="Sector ROI means relative to sector '{sector_norm}'",
    # )
