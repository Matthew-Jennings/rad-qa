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
Module for analyzing the Dose Rate - Collimator Speed (DR-CS) quality control test for
Varian TrueBeam radiotherapy systems.

This script processes DICOM images (predicted or acquired on a TrueBeam EPID) to
calculate statistics for five Regions of Interest (ROIs) defined in a configuration
file. It outputs the results to CSV and Excel files and provides options for visual
inspection of the ROIs overlaid on the images.

Optionally, it can normalize the DR-CS ROI statistics by normalization field statistics
when analyzing image pairs.

Usage:
    python analyse_drcs.py input_directory [--config CONFIG_PATH] [--inspect-live]
        [--inspect-save] [--normalize] [--open-csv] [--open-excel]

Arguments:
    input_directory:
        Path to the input directory containing DICOM files. By default, this directory
        should also contain a config.json file with ROI configurations, unless a
        different config file is specified via --config.

Optional arguments:
    --config:
        Path to the configuration JSON file. If not provided, defaults to
        input_directory/config.json.
    --inspect-live:
        Display images with ROIs overlaid during processing.
    --inspect-save:
        Save images with ROIs overlaid to input_directory.
    --normalize:
        Normalize DR-CS ROI stats by the normalization field stats.
    --open-csv:
        Automatically open the output CSV file after processing.
    --open-excel:
        Automatically open the output Excel file after processing.
"""

import argparse
import copy
import os
import pathlib
import json
import sys
import platform
import subprocess
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon
from pydicom.dataset import FileDataset

# Configure default logging to INFO. This will be overridden after loading config.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Hardcoded ROI labels
ROI_LABELS: List[str] = ["A", "B", "C", "D", "E"]
NUM_ROIS: int = len(ROI_LABELS)


def get_rotated_rectangle_vertices(
    cx: float, cy: float, width: float, height: float, angle_deg: float
) -> np.ndarray:
    """Compute the vertices of a rotated rectangle.

    Given the rectangle's center coordinates, dimensions, and rotation angle, this
    function computes the (x, y) coordinates of the rectangle's four corners.

    Args:
        cx: X-coordinate of rectangle center.
        cy: Y-coordinate of rectangle center.
        width: Rectangle width in pixels.
        height: Rectangle height in pixels.
        angle_deg: Rotation angle in degrees, counter-clockwise.

    Returns:
        np.ndarray: Array of shape (4, 2) containing the vertices of the rectangle.
    """
    logger.debug(
        "Calculating rotated rectangle vertices with center=(%.2f, %.2f), "
        "width=%.2f, height=%.2f, angle_deg=%.2f",
        cx,
        cy,
        width,
        height,
        angle_deg,
    )
    angle_rad = np.deg2rad(-angle_deg)
    w = width / 2
    h = height / 2

    corners = np.array([[-w, -h], [w, -h], [w, h], [-w, h]])
    R = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )
    rotated_corners = corners @ R
    vertices = rotated_corners + np.array([cx, cy])
    logger.debug("Computed vertices:\n%s", vertices)
    return vertices


def load_roi_config(config_path: pathlib.Path) -> Dict[str, Any]:
    """Load ROI configuration parameters from a JSON file.

    This function reads a JSON configuration file specifying ROI geometry, angles, and
    optional image labels for normalization fields.

    The JSON file should specify:
    - roi_center_offset_from_image_centre_mm (float)
    - roi_width_mm (float)
    - roi_height_mm (float)
    - roi_angles (list of floats, length 5)
    - roi_colors (list of strings, length 5)
    - open_rtimage_labels (list of strings)
    - drcs_rtimage_labels (list of strings)
    - log_level (string, optional): DEBUG, INFO, WARNING, ERROR, CRITICAL

    Args:
        config_path: Path to the configuration JSON file.

    Returns:
        A dictionary containing ROI configuration, including 'roi_list',
        'open_rtimage_labels', 'drcs_rtimage_labels', and 'log_level'.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        ValueError: If JSON is invalid or missing required keys.
    """
    logger.info("Loading ROI configuration from %s", config_path)
    if not config_path.is_file():
        logger.error("Configuration file not found at %s", config_path)
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8-sig") as f:
            config = json.load(f)
    except json.JSONDecodeError:
        logger.exception("Invalid JSON format in configuration file %s", config_path)
        raise

    logger.debug("Loaded config: %s", config)

    required_shared_keys = {
        "roi_center_offset_from_image_centre_mm",
        "roi_width_mm",
        "roi_height_mm",
    }
    required_list_keys = {"roi_angles", "roi_colors"}

    missing_shared_keys = required_shared_keys - set(config.keys())
    missing_list_keys = required_list_keys - set(config.keys())

    if missing_shared_keys:
        logger.error("Missing required shared keys: %s", missing_shared_keys)
        raise ValueError(f"Missing required shared keys: {missing_shared_keys}")
    if missing_list_keys:
        logger.error("Missing required list keys: %s", missing_list_keys)
        raise ValueError(f"Missing required list keys: {missing_list_keys}")

    list_lengths = {key: len(config[key]) for key in required_list_keys}
    if not all(length == NUM_ROIS for length in list_lengths.values()):
        logger.error(
            "All lists must have length %d. Current lengths: %s", NUM_ROIS, list_lengths
        )
        raise ValueError(
            f"All lists must have length {NUM_ROIS}. "
            f"Current lengths: {list_lengths}"
        )

    roi_config_list = []
    for i in range(NUM_ROIS):
        roi_dict = {
            "roi_center_offset_from_image_centre_mm": config[
                "roi_center_offset_from_image_centre_mm"
            ],
            "roi_width_mm": config["roi_width_mm"],
            "roi_height_mm": config["roi_height_mm"],
            "roi_angle": config["roi_angles"][i],
            "roi_color": config["roi_colors"][i],
            "roi_label": ROI_LABELS[i],
        }
        roi_config_list.append(roi_dict)

    open_rtimage_labels = config.get("open_rtimage_labels", [])
    drcs_rtimage_labels = config.get("drcs_rtimage_labels", [])
    open_rtimage_labels = [label.lower() for label in open_rtimage_labels]
    drcs_rtimage_labels = [label.lower() for label in drcs_rtimage_labels]

    # Read log_level from config, default to INFO if not specified
    log_level_str = config.get("log_level", "INFO").upper()
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level_str not in valid_log_levels:
        logger.warning(
            "Invalid 'log_level' specified in config: '%s'. Defaulting to 'INFO'.",
            log_level_str,
        )
        log_level_str = "INFO"

    roi_config: Dict[str, Any] = {
        "roi_list": roi_config_list,
        "open_rtimage_labels": open_rtimage_labels,
        "drcs_rtimage_labels": drcs_rtimage_labels,
        "log_level": log_level_str,  # Added log_level to configuration
    }

    logger.debug("Final ROI configuration:\n%s", roi_config)
    return roi_config


def get_roi_stats(
    ds: FileDataset,
    roi_config: Dict[str, Any],
    inspect_mode: Optional[str] = None,
    output_dir: Optional[pathlib.Path] = None,
    image_name: Optional[str] = None,
) -> Tuple[Dict[str, float], bool]:
    """Calculate statistics for ROIs in a single DICOM image.

    This function:
    1. Reads and scales the DICOM image data into cGy or cGy/MU.
    2. Computes mean pixel values within each specified ROI.
    3. Optionally displays or saves an image with ROIs overlaid.

    Args:
        ds: The DICOM dataset of the image.
        roi_config: Dictionary containing ROI configuration parameters.
        inspect_mode: 'live' to display images, 'save' to save images with ROIs, or
            None for no visualization.
        output_dir: Directory to save images if inspect_mode='save'.
        image_name: Name used for saved images.

    Returns:
        A tuple containing:
            - A dictionary mapping each ROI label (e.g., "A") to its mean pixel value.
            - A boolean indicating whether the image is acquired (`True`) or predicted
              (`False`).

    Raises:
        AttributeError: If required DICOM attributes (RescaleSlope, ImageType, etc.)
            are missing.
    """
    logger.debug("Extracting ROI stats for image: %s", image_name)
    rescale_slope = getattr(ds, "RescaleSlope", None)
    rescale_intercept = getattr(ds, "RescaleIntercept", None)

    if rescale_slope is None or rescale_intercept is None:
        logger.error("Missing 'RescaleSlope' or 'RescaleIntercept' in DICOM.")
        raise AttributeError(
            "DICOM file missing 'RescaleSlope' or 'RescaleIntercept' attributes."
        )

    image = ds.pixel_array * rescale_slope + rescale_intercept
    image *= 100  # Gy to cGy
    logger.debug("Image scaled to cGy units.")

    image_type_attr = getattr(ds, "ImageType", None)
    if image_type_attr is None:
        logger.error("Missing 'ImageType' attribute in DICOM.")
        raise AttributeError("DICOM file missing 'ImageType' attribute.")

    # Determine if the image is acquired or predicted
    try:
        is_acquired = image_type_attr[3] == "ACQUIRED_DOSE"
    except IndexError:
        is_acquired = image_type_attr[0] == "ORIGINAL"

    if is_acquired:
        exposure_sequence = getattr(ds, "ExposureSequence", None)
        if exposure_sequence is None or len(exposure_sequence) == 0:
            logger.error("Missing 'ExposureSequence' in ACQUIRED_DOSE image.")
            raise AttributeError(
                "DICOM file missing 'ExposureSequence' or it is empty."
            )

        meterset_exposure = getattr(exposure_sequence[0], "MetersetExposure", None)
        if meterset_exposure is None:
            logger.error("Missing 'MetersetExposure' in ACQUIRED_DOSE image.")
            raise AttributeError(
                "DICOM file missing 'MetersetExposure' in 'ExposureSequence'."
            )

        image /= meterset_exposure  # cGy/MU
        logger.debug("Normalized acquired dose image by MetersetExposure.")
    else:
        logger.debug(
            "Image is predicted; no normalization by MetersetExposure applied."
        )

    image_height, image_width = image.shape
    image_center_x, image_center_y = image_width / 2, image_height / 2

    row_spacing_mm = (
        float(ds.ImagePlanePixelSpacing[0])
        if hasattr(ds, "ImagePlanePixelSpacing")
        else None
    )
    col_spacing_mm = (
        float(ds.ImagePlanePixelSpacing[1])
        if hasattr(ds, "ImagePlanePixelSpacing")
        else None
    )

    if row_spacing_mm is None or col_spacing_mm is None:
        logger.error("Missing 'ImagePlanePixelSpacing' in DICOM.")
        raise AttributeError("DICOM file missing 'ImagePlanePixelSpacing' attribute.")

    logger.debug(
        "Image dimensions (px): %dx%d, center=(%.2f, %.2f), "
        "pixel spacing=(%.2fmm, %.2fmm)",
        image_width,
        image_height,
        image_center_x,
        image_center_y,
        col_spacing_mm,
        row_spacing_mm,
    )

    fig = None
    ax = None
    if inspect_mode in ["live", "save"]:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image, cmap="gray")

    roi_stats: Dict[str, float] = {}

    for roi in roi_config["roi_list"]:
        logger.debug("Processing ROI: %s", roi["roi_label"])
        theta_rad = np.deg2rad(-roi["roi_angle"])

        rect_center_x = (
            image_center_x
            + roi["roi_center_offset_from_image_centre_mm"]
            * np.cos(theta_rad)
            / col_spacing_mm
        )
        rect_center_y = (
            image_center_y
            - roi["roi_center_offset_from_image_centre_mm"]
            * np.sin(theta_rad)
            / row_spacing_mm
        )

        roi_width_px = round(roi["roi_width_mm"] / col_spacing_mm)
        roi_height_px = round(roi["roi_height_mm"] / row_spacing_mm)

        vertices = get_rotated_rectangle_vertices(
            rect_center_x, rect_center_y, roi_width_px, roi_height_px, roi["roi_angle"]
        )

        vertices[:, 0] = np.clip(vertices[:, 0], 0, image_width - 1)
        vertices[:, 1] = np.clip(vertices[:, 1], 0, image_height - 1)

        mask = np.zeros_like(image, dtype=bool)
        rr, cc = polygon(vertices[:, 1], vertices[:, 0], image.shape)
        mask[rr, cc] = True

        roi_pixels = image[mask]
        roi_mean = float(np.mean(roi_pixels))
        roi_stats[roi["roi_label"]] = roi_mean
        logger.debug("ROI %s mean pixel value: %.4f cGy/MU", roi["roi_label"], roi_mean)

        if inspect_mode in ["live", "save"] and fig is not None and ax is not None:
            polygon_patch = plt.Polygon(
                vertices,
                closed=True,
                edgecolor=roi["roi_color"],
                facecolor="none",
                linewidth=2,
            )
            ax.add_patch(polygon_patch)
            ax.text(
                rect_center_x,
                rect_center_y,
                roi["roi_label"],
                color=roi["roi_color"],
                fontsize=10,
                ha="center",
                va="center",
                fontweight="bold",
            )

    if inspect_mode == "live" and fig is not None:
        plt.title("DICOM Image with Rotated ROIs")
        plt.axis("off")
        plt.show()
    elif (
        inspect_mode == "save"
        and output_dir is not None
        and image_name is not None
        and fig is not None
    ):
        plt.title("DICOM Image with Rotated ROIs")
        plt.axis("off")
        save_path = output_dir / f"{image_name}.png"
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        logger.debug("Saved ROI overlay image to %s", save_path)

    return roi_stats, is_acquired  # Modified return statement


def get_roi_stats_for_images_in_dir(
    dirpath: pathlib.Path,
    roi_config: Dict[str, Any],
    inspect_mode: Optional[str] = None,
    normalize: bool = False,
) -> pd.DataFrame:
    """Process all DICOM images in a directory to calculate ROI statistics.

    This function:
    1. Iterates through all .dcm files in the given directory.
    2. Identifies which images are DR-CS and which are OPEN, based on RTImageLabels
       provided in the ROI configuration.
    3. Calculates ROI statistics for each valid image.
    4. Optionally normalizes DR-CS ROI stats by corresponding OPEN fields grouped by
       AcquisitionDate and Group (acquired/predicted).
    5. Returns a pandas DataFrame with all ROI statistics, as well as computed
       "Average" and "Max vs Min" columns.

    Args:
        dirpath: Path to the directory containing DICOM files.
        roi_config: Dictionary containing ROI configurations and RTImageLabels.
        inspect_mode: 'live' or 'save' for visualization, None otherwise.
        normalize: If True, normalizes DR-CS ROI stats by OPEN field stats.

    Returns:
        A DataFrame containing ROI statistics for all processed images.
    """
    logger.info("Processing images in directory: %s", dirpath)
    roi_stats_all: List[Dict[str, Any]] = []
    dicom_fpaths = list(dirpath.glob("*.dcm"))
    dicom_fpath_count = len(dicom_fpaths)
    logger.debug("Found %d DICOM files in directory.", dicom_fpath_count)

    if inspect_mode == "save":
        output_image_dir = dirpath / "output_images"
        output_image_dir.mkdir(exist_ok=True)
        logger.debug("Created output_images directory at %s", output_image_dir)
    else:
        output_image_dir = None

    for i, dicom_fpath in enumerate(dicom_fpaths, start=1):
        logger.info(
            "Analyzing DICOM file %i of %i: %s", i, dicom_fpath_count, dicom_fpath.name
        )
        ds_modality_only = pydicom.dcmread(dicom_fpath, specific_tags=["Modality"])
        if ds_modality_only.Modality != "RTIMAGE":
            logger.info("Skipping non-RTIMAGE file: %s", dicom_fpath.name)
            continue

        ds = pydicom.dcmread(dicom_fpath)
        rt_image_label = getattr(ds, "RTImageLabel", "").lower()
        acquisition_date = getattr(ds, "AcquisitionDate", None)

        if acquisition_date is None and normalize:
            logger.warning(
                "Missing 'AcquisitionDate' in file '%s'. "
                "This is required for normalization",
                dicom_fpath.name,
            )
            continue

        if rt_image_label in roi_config["drcs_rtimage_labels"]:
            image_type = "DRCS"
        elif rt_image_label in roi_config["open_rtimage_labels"]:
            image_type = "OPEN"
        else:
            logger.warning(
                "Unrecognized RTImageLabel '%s' in file '%s'. "
                "Consider adding to config.",
                rt_image_label,
                dicom_fpath.name,
            )
            image_type = "UNKNOWN"

        if image_type == "UNKNOWN":
            continue

        roi_config_copy = copy.deepcopy(roi_config)
        # Adjust ROI angles for rotated images (indicated by "_A" in filename)
        if "_A" in dicom_fpath.stem:
            logger.debug("Adjusting ROI angles for rotated image: %s", dicom_fpath.name)
            for roi in roi_config_copy["roi_list"]:
                roi["roi_angle"] = (roi["roi_angle"] - 180) % 360

        try:
            roi_stats, is_acquired = get_roi_stats(
                ds,
                roi_config_copy,
                inspect_mode=inspect_mode,
                output_dir=output_image_dir,
                image_name=dicom_fpath.stem,
            )
            # Modify ImageType based on is_acquired flag
            if image_type == "DRCS":
                image_type_final = "DRCS" if is_acquired else "DRCS PREDICTED"
            elif image_type == "OPEN":
                image_type_final = "OPEN" if is_acquired else "OPEN PREDICTED"
            else:
                image_type_final = image_type  # Shouldn't reach here

            roi_stats["File"] = dicom_fpath.stem
            roi_stats["RTImageLabel"] = rt_image_label
            roi_stats["ImageType"] = image_type_final  # Updated ImageType
            roi_stats["AcquisitionDate"] = acquisition_date
            roi_stats_all.append(roi_stats)
        except AttributeError as e:
            logger.error("Error processing file %s: %s", dicom_fpath.name, e)
            continue

    df = pd.DataFrame(roi_stats_all)
    if df.empty:
        logger.warning("No valid images were processed; the DataFrame is empty.")
        return df

    for col in ROI_LABELS:
        df[col] = pd.to_numeric(df[col])

    # Warn if there is more than one "OPEN PREDICTED" or "DRCS PREDICTED" per
    # AcquisitionDate
    for image_type in ["OPEN PREDICTED", "DRCS PREDICTED"]:
        counts = df[df["ImageType"] == image_type].groupby("AcquisitionDate").size()
        multiple = counts[counts > 1]
        if not multiple.empty:
            for date, count in multiple.iteritems():
                logger.warning(
                    "More than one '%s' image for AcquisitionDate '%s'. "
                    "Found %d instances.",
                    image_type,
                    date,
                    count,
                )

    # Add a 'Group' column to differentiate between acquired and predicted images
    def determine_group(image_type: str) -> Optional[str]:
        if image_type in ["DRCS", "OPEN"]:
            return "acquired"
        elif image_type in ["DRCS PREDICTED", "OPEN PREDICTED"]:
            return "predicted"
        else:
            return None

    df["Group"] = df["ImageType"].apply(determine_group)

    if normalize:
        logger.debug("Normalizing ROI stats by corresponding normalization fields.")
        non_roi_columns = [
            "File",
            "RTImageLabel",
            "ImageType",
            "AcquisitionDate",
            "Group",
        ]
        roi_columns = [
            c
            for c in df.columns
            if c not in non_roi_columns and pd.api.types.is_numeric_dtype(df[c])
        ]
        normalized_data: List[Dict[str, Any]] = []

        # Group by AcquisitionDate and Group (acquired/predicted)
        grouped = df.groupby(["AcquisitionDate", "Group"])

        for (acq_date, group), group_df in grouped:
            if group == "acquired":
                open_images = group_df[group_df["ImageType"] == "OPEN"]
                if len(open_images) == 0:
                    logger.warning(
                        "Missing 'OPEN' image for AcquisitionDate '%s'. "
                        "Skipping normalization for this group.",
                        acq_date,
                    )
                    continue
                elif len(open_images) > 1:
                    logger.warning(
                        "Multiple 'OPEN' images for AcquisitionDate '%s'. "
                        "Using the first one for normalization.",
                        acq_date,
                    )
                open_image = open_images.iloc[0]

                drcs_images = group_df[group_df["ImageType"] == "DRCS"]

                for _, drcs_image in drcs_images.iterrows():
                    normalized_values = (
                        drcs_image[roi_columns] / open_image[roi_columns]
                    )
                    norm_row_dict = drcs_image[non_roi_columns].to_dict()
                    norm_row_dict["ImageType"] = "NORMALIZED"
                    for c in roi_columns:
                        norm_row_dict[c] = float(normalized_values[c])
                    normalized_data.append(norm_row_dict)

            elif group == "predicted":
                open_pred_images = group_df[group_df["ImageType"] == "OPEN PREDICTED"]
                if len(open_pred_images) == 0:
                    logger.warning(
                        "Missing 'OPEN PREDICTED' image for AcquisitionDate '%s'. "
                        "Skipping normalization for this group.",
                        acq_date,
                    )
                    continue
                elif len(open_pred_images) > 1:
                    logger.warning(
                        "Multiple 'OPEN PREDICTED' images for AcquisitionDate '%s'. "
                        "Using the first one for normalization.",
                        acq_date,
                    )
                open_pred_image = open_pred_images.iloc[0]

                drcs_pred_images = group_df[group_df["ImageType"] == "DRCS PREDICTED"]

                for _, drcs_image in drcs_pred_images.iterrows():
                    normalized_values = (
                        drcs_image[roi_columns] / open_pred_image[roi_columns]
                    )
                    norm_row_dict = drcs_image[non_roi_columns].to_dict()
                    norm_row_dict["ImageType"] = "NORMALIZED"
                    for c in roi_columns:
                        norm_row_dict[c] = float(normalized_values[c])
                    normalized_data.append(norm_row_dict)

        if normalized_data:
            df_normalized = pd.DataFrame(normalized_data)
            df = pd.concat([df, df_normalized], ignore_index=True)
            logger.debug("Added normalized data to the DataFrame.")
        else:
            logger.debug(
                "No normalized data was generated (no DRCS-OPEN or "
                "DRCS PREDICTED-OPEN PREDICTED pairs matched)."
            )

    # Calculate 'Average' and 'Max vs Min' columns
    non_roi_columns = ["File", "RTImageLabel", "ImageType", "AcquisitionDate", "Group"]
    roi_columns = [
        c
        for c in df.columns
        if c not in non_roi_columns and pd.api.types.is_numeric_dtype(df[c])
    ]

    if roi_columns:
        df["Average"] = df[roi_columns].mean(axis=1)
        min_values = df[roi_columns].min(axis=1).replace(0, np.nan)
        df["Max vs Min"] = df[roi_columns].max(axis=1) / min_values - 1
        df.fillna({"Max vs Min": np.inf}, inplace=True)
    else:
        logger.warning("No ROI columns found in the DataFrame; skipping calculations.")

    desired_order = (
        ["File", "RTImageLabel", "ImageType", "AcquisitionDate"]
        + roi_columns
        + ["Average", "Max vs Min"]
    )
    ordered_columns = [c for c in desired_order if c in df.columns]
    other_columns = [c for c in df.columns if c not in ordered_columns]
    df = df[ordered_columns + other_columns]

    # Drop the 'Group' column before saving to exclude it from output files
    if "Group" in df.columns:
        df = df.drop(columns=["Group"])
        logger.debug("Dropped 'Group' column from the DataFrame for output.")

    logger.debug("Final DataFrame:\n%s", df.head())
    return df


def open_file(filepath: pathlib.Path) -> None:
    """Open a file using the default application based on the operating system.

    Args:
        filepath: The path to the file to be opened.

    Raises:
        subprocess.CalledProcessError: If the command to open the file fails.
    """
    logger.debug("Opening file with default application: %s", filepath)
    if platform.system() == "Windows":
        os.startfile(str(filepath))
    elif platform.system() == "Darwin":  # macOS
        subprocess.run(["open", str(filepath)], check=True)
    else:  # Linux and other Unix systems
        subprocess.run(["xdg-open", str(filepath)], check=True)


def main() -> None:
    """Main function to execute the DR-CS analysis workflow.

    Steps performed:
    1. Parse command-line arguments.
    2. Load ROI configuration from a specified or default config file.
    3. Configure logging based on the 'log_level' from config.
    4. Process all DICOM images in the input directory to calculate ROI statistics.
    5. Optionally normalize the DR-CS statistics by corresponding OPEN fields.
    6. Save results to CSV and Excel files.
    7. Optionally open the CSV or Excel results file.

    Raises:
        SystemExit: If the configuration cannot be loaded or if image processing
            encounters critical issues that prevent completion.
    """
    parser = argparse.ArgumentParser(description="Process DICOM images in a directory.")
    parser.add_argument(
        "input_directory",
        type=str,
        help="Path to the input directory containing DICOM files.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the configuration JSON file. If not provided, defaults to "
        "'config.json' in the input directory.",
    )
    parser.add_argument(
        "--inspect-live",
        action="store_true",
        help="Display images with ROIs overlaid during processing.",
    )
    parser.add_argument(
        "--inspect-save",
        action="store_true",
        help="Save images with ROIs overlaid to files to the input directory.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize DR-CS ROI stats by the normalization field stats.",
    )
    parser.add_argument(
        "--open-csv",
        action="store_true",
        help="Automatically open the output CSV file after processing.",
    )
    parser.add_argument(
        "--open-excel",
        action="store_true",
        help="Automatically open the output Excel file after processing.",
    )
    args = parser.parse_args()

    data_dirpath = pathlib.Path(args.input_directory)
    # Use the provided config path if given, otherwise default to
    # input_directory/config.json
    config_path = (
        pathlib.Path(args.config) if args.config else data_dirpath / "config.json"
    )
    # Initial log to inform user about config loading
    logger.info("Starting DR-CS analysis for directory: %s", data_dirpath)
    logger.info("Using config file: %s", config_path)

    inspect_mode: Optional[str] = None
    if args.inspect_live:
        inspect_mode = "live"
    elif args.inspect_save:
        inspect_mode = "save"

    try:
        roi_config = load_roi_config(config_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Error loading ROI configuration: %s", e)
        sys.exit(1)

    # Configure logging level based on config after loading
    log_level_str = roi_config.get("log_level", "INFO").upper()
    numeric_level = getattr(logging, log_level_str, None)
    if not isinstance(numeric_level, int):
        logger.warning("Invalid 'log_level' '%s'. Defaulting to 'INFO'.", log_level_str)
        numeric_level = logging.INFO

    # Update logger level
    logger.setLevel(numeric_level)
    for handler in logger.handlers:
        handler.setLevel(numeric_level)
    logger.debug("Logging level set to %s.", log_level_str)

    try:
        df = get_roi_stats_for_images_in_dir(
            data_dirpath,
            roi_config,
            inspect_mode=inspect_mode,
            normalize=args.normalize,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error("Error processing images: %s", e)
        sys.exit(1)

    # Drop the 'Group' column if it exists to exclude it from output files
    if "Group" in df.columns:
        df = df.drop(columns=["Group"])
        logger.debug("Dropped 'Group' column from the DataFrame for output.")

    csv_savepath = data_dirpath / "roi_stats.csv"
    excel_savepath = data_dirpath / "roi_stats.xlsx"

    df.to_csv(csv_savepath, index=False)
    logger.info("Saved ROI statistics to CSV: %s", csv_savepath)

    with pd.ExcelWriter(excel_savepath, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")

        workbook = writer.book
        worksheet = writer.sheets["Results"]

        percentage_format = workbook.add_format({"num_format": "0.00%"})
        dp4_format = workbook.add_format({"num_format": "0.0000"})
        dp5_format = workbook.add_format({"num_format": "0.00000"})

        # Set 4 dp float format for ROI columns
        for roi_label in ["A", "B", "C", "D", "E"]:
            if roi_label in df.columns:
                col_index = df.columns.get_loc(roi_label)
                worksheet.set_column(col_index, col_index, None, dp4_format)

        # Set 5 dp float format for "Average"
        if "Average" in df.columns:
            avg_col = df.columns.get_loc("Average")
            worksheet.set_column(avg_col, avg_col, None, dp5_format)

        # Set 2 dp percentage format for "Max vs Min"
        if "Max vs Min" in df.columns:
            max_vs_min_col = df.columns.get_loc("Max vs Min")
            worksheet.set_column(
                max_vs_min_col, max_vs_min_col, None, percentage_format
            )

    logger.info("Saved ROI statistics to Excel: %s", excel_savepath)
    logger.debug("Saved CSV and Excel results.")

    if args.open_csv:
        try:
            open_file(csv_savepath)
            logger.debug("Opened CSV file: %s", csv_savepath)
        except Exception:
            logger.exception("Failed to open CSV file '%s'.", csv_savepath)

    if args.open_excel:
        try:
            open_file(excel_savepath)
            logger.debug("Opened Excel file: %s", excel_savepath)
        except Exception:
            logger.exception("Failed to open Excel file '%s'", excel_savepath)


if __name__ == "__main__":
    main()
