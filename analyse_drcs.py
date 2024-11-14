# Copyright (C) 2024 Matthew Jennings

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

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
when analyzing image pairs. The script is designed to facilitate quality assurance by
automating the analysis of DR-CS tests.

Usage:

    python analyse_drcs.py input_directory [--inspect-live] [--inspect-save]
        [--normalize] [--open-csv] [--open-excel]

Arguments:

    input_directory:
        Path to the input directory containing DICOM files and a config.json
        file with ROI configurations.

Optional arguments:

    --inspect-live:
        Display images with ROIs overlaid during processing.
    --inspect-save:
        Save images with ROIs overlaid to files.
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

import pandas as pd
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardcoded ROI labels
ROI_LABELS = ["A", "B", "C", "D", "E"]
NUM_ROIS = len(ROI_LABELS)


def get_rotated_rectangle_vertices(cx, cy, width, height, angle_deg):
    """
    Compute the vertices of a rotated rectangle given its center, dimensions, and
    rotation angle.

    Args:
        cx (float): X-coordinate of the rectangle center in pixels.
        cy (float): Y-coordinate of the rectangle center in pixels.
        width (float): Width of the rectangle in pixels.
        height (float): Height of the rectangle in pixels.
        angle_deg (float): Rotation angle in degrees. Positive values correspond to
            counter-clockwise rotation.

    Returns:
        np.ndarray: An array of shape `(4, 2)` containing the (x, y) pixel coordinates
            of the rectangle's corners.
    """
    # Convert angle to radians, and negate for image coordinate system
    angle_rad = np.deg2rad(-angle_deg)

    # Define half-dimensions
    w = width / 2
    h = height / 2

    # Corners relative to center
    corners = np.array([[-w, -h], [w, -h], [w, h], [-w, h]])

    # Rotation matrix
    R = np.array(  # pylint: disable=invalid-name
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )

    # Rotate corners
    rotated_corners = corners @ R

    # Translate to center position
    vertices = rotated_corners + np.array([cx, cy])

    return vertices


def load_roi_config(config_path):
    """
    Load ROI configuration parameters from a JSON file.

    The configuration file should contain shared parameters and lists of angles,
    and colors for each of the five ROIs. The shared parameters include the
    center offset, width, and height of the ROIs. The lists provide specific attributes
    for each ROI.

    Expected JSON structure::

        {
            "roi_center_offset_from_image_centre_mm": <float>,
            "roi_width_mm": <float>,
            "roi_height_mm": <float>,
            "roi_angles": [<float>, <float>, ..., <float>],    # List of length NUM_ROIS
            "roi_colors": [<str>, <str>, ..., <str>],          # List of length NUM_ROIS
            "open_rtimage_labels": [<str>, <str>, ...],
            "drcs_rtimage_labels": [<str>, <str>, ...]
        }

    Args:
        config_path (pathlib.Path): Path to the configuration JSON file.

    Returns:
        dict: A dictionary containing ROI configurations and normalization identifiers.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the configuration file is invalid or missing required keys.
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON format in configuration file {config_path}"
        ) from exc

    # Validate configuration
    required_shared_keys = {
        "roi_center_offset_from_image_centre_mm",
        "roi_width_mm",
        "roi_height_mm",
    }
    required_list_keys = {"roi_angles", "roi_colors"}

    # Check for required keys
    missing_shared_keys = required_shared_keys - set(config.keys())
    missing_list_keys = required_list_keys - set(config.keys())

    if missing_shared_keys:
        raise ValueError(f"Missing required shared keys: {missing_shared_keys}")
    if missing_list_keys:
        raise ValueError(f"Missing required list keys: {missing_list_keys}")

    # Check all lists have length NUM_ROIS
    list_lengths = {key: len(config[key]) for key in required_list_keys}
    if not all(length == NUM_ROIS for length in list_lengths.values()):
        raise ValueError(
            f"All lists must have length {NUM_ROIS}. Current lengths: {list_lengths}"
        )

    # Convert configuration format to list of dictionaries
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

    # Get normalization identifiers
    open_rtimage_labels = config.get("open_rtimage_labels", [])
    drcs_rtimage_labels = config.get("drcs_rtimage_labels", [])

    # Ensure that the labels are in lowercase for case-insensitive comparison
    open_rtimage_labels = [label.lower() for label in open_rtimage_labels]
    drcs_rtimage_labels = [label.lower() for label in drcs_rtimage_labels]

    roi_config = {
        "roi_list": roi_config_list,
        "open_rtimage_labels": open_rtimage_labels,
        "drcs_rtimage_labels": drcs_rtimage_labels,
    }

    return roi_config


def get_roi_stats(ds, roi_config, inspect_mode=None, output_dir=None, image_name=None):
    """
    Calculate statistics for Regions of Interest (ROIs) in a DICOM image.

    This function processes a DICOM image to extract mean pixel values within specified
    ROIs. It handles image scaling to ensure pixel values are in consistent units
    (cGy/MU). The ROIs are defined based on a configuration that specifies their
    positions relative to the image center, dimensions, and rotation angles.

    Key steps:

    1. Image Extraction and Scaling:
        - Extract the image data from the DICOM dataset.
        - Apply scaling factors (`RescaleSlope` and `RescaleIntercept`) to convert raw
          pixel data to physical units.
        - Convert dose units from Gy to cGy.
        - If the image is an acquired dose, normalize by the `MetersetExposure` to
          obtain cGy/MU.

    2. ROI Processing:
        - For each ROI in the configuration:
            - Calculate the center position in image coordinates, accounting for pixel
              spacing and rotation.
            - Determine the vertices of the rotated rectangle representing the ROI.
            - Create a mask for the ROI and extract the pixel values within it.
            - Compute the mean pixel value for the ROI.

    3. Visualization (Optional):
        - If `inspect_mode` is 'live', display the image with ROIs overlaid for visual
          inspection.
        - If `inspect_mode` is 'save', save the image with ROIs overlaid to a file.

    Args:
        ds (pydicom.dataset.FileDataset): DICOM dataset containing the image and
            metadata.
        roi_config (dict): Dictionary containing ROI configuration parameters.
        inspect_mode (str, optional): If 'live', displays the image with ROIs overlaid.
            If 'save', saves the image with ROIs overlaid to a file.
            Defaults to None.
        output_dir (pathlib.Path, optional): Directory to save images if inspect_mode
            is 'save'.
        image_name (str, optional): Name to use for the saved image file.

    Returns:
        dict: A dictionary where keys are ROI labels and values are the mean pixel
            values within each ROI.

    Raises:
        AttributeError: If necessary DICOM attributes are missing.
    """
    # Extract image data
    rescale_slope = getattr(ds, "RescaleSlope", None)
    rescale_intercept = getattr(ds, "RescaleIntercept", None)

    if rescale_slope is None or rescale_intercept is None:
        raise AttributeError(
            "DICOM file missing 'RescaleSlope' or 'RescaleIntercept' attributes."
        )

    image = ds.pixel_array * rescale_slope + rescale_intercept
    image *= 100  # Gy to cGy

    image_type = getattr(ds, "ImageType", None)
    if image_type is None or len(image_type) < 4:
        raise AttributeError(
            "DICOM file missing 'ImageType' attribute or it is not in the expected "
            "format."
        )

    if image_type[3] == "ACQUIRED_DOSE":
        # Check if 'ExposureSequence' exists and has at least one item
        exposure_sequence = getattr(ds, "ExposureSequence", None)
        if exposure_sequence is None or len(exposure_sequence) == 0:
            raise AttributeError(
                "DICOM file missing 'ExposureSequence' or it is empty."
            )

        # Check if 'MetersetExposure' exists in the first item of ExposureSequence
        meterset_exposure = getattr(exposure_sequence[0], "MetersetExposure", None)
        if meterset_exposure is None:
            raise AttributeError(
                "DICOM file missing 'MetersetExposure' in 'ExposureSequence'."
            )

        image /= meterset_exposure  # cGy / MU

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
        raise AttributeError("DICOM file missing 'ImagePlanePixelSpacing' attribute.")

    # Prepare to display or save the image
    if inspect_mode in ["live", "save"]:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image, cmap="gray")
    else:
        fig, ax = None, None

    # Store statistics
    roi_stats = {}

    for roi in roi_config["roi_list"]:
        # Convert angle to radians
        theta_rad = np.deg2rad(-roi["roi_angle"])

        # Compute rectangle center positions
        rect_center_x = (
            image_center_x
            + roi["roi_center_offset_from_image_centre_mm"]
            * np.cos(theta_rad)
            / col_spacing_mm
        )
        rect_center_y = (
            image_center_y
            - roi[
                "roi_center_offset_from_image_centre_mm"
            ]  # Negate for image coordinate system
            * np.sin(theta_rad)
            / row_spacing_mm
        )

        roi_width_px = round(roi["roi_width_mm"] / col_spacing_mm)
        roi_height_px = round(roi["roi_height_mm"] / row_spacing_mm)

        # Get rectangle vertices
        vertices = get_rotated_rectangle_vertices(
            rect_center_x, rect_center_y, roi_width_px, roi_height_px, roi["roi_angle"]
        )

        # Ensure vertices are within image bounds
        vertices[:, 0] = np.clip(vertices[:, 0], 0, image_width - 1)
        vertices[:, 1] = np.clip(vertices[:, 1], 0, image_height - 1)

        # Create a mask for the ROI
        mask = np.zeros_like(image, dtype=bool)
        rr, cc = polygon(vertices[:, 1], vertices[:, 0], image.shape)
        mask[rr, cc] = True

        # Extract pixel values within the ROI
        roi_pixels = image[mask]

        # Compute statistics
        roi_stats[roi["roi_label"]] = np.mean(roi_pixels)

        if inspect_mode in ["live", "save"]:
            # Overlay the ROI on the image
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

    if inspect_mode == "live":
        plt.title("DICOM Image with Rotated ROIs")
        plt.axis("off")
        plt.show()
    elif inspect_mode == "save" and output_dir is not None and image_name is not None:
        plt.title("DICOM Image with Rotated ROIs")
        plt.axis("off")
        # Save the figure
        fig.savefig(output_dir / f"{image_name}.png", bbox_inches="tight")
        plt.close(fig)

    return roi_stats


def get_roi_stats_for_images_in_dir(
    dirpath, roi_config, inspect_mode=None, normalize=False
):
    """
    Process all DICOM images in a directory to calculate ROI statistics.

    This function processes each DICOM file in the specified directory to calculate
    mean pixel values within specified ROIs. It compiles the results into a pandas
    DataFrame. Optionally, it can normalize the DR-CS ROI statistics by corresponding
    normalization field statistics.

    Key steps:

    1. Iterate Over DICOM Files:
        - For each DICOM file in the directory:
            - Read the DICOM dataset.
            - Extract 'AcquisitionDate' for grouping.
            - Determine image type based on 'RTImageLabel'.
            - Adjust ROI angles for rotated images if necessary.
            - Calculate ROI statistics using `get_roi_stats`.
            - Store the results along with the file name, RTImageLabel, and
              AcquisitionDate.

    2. DataFrame Creation:
        - Compile the collected statistics into a pandas DataFrame.

    3. Normalization (Optional):
        - If normalization is requested:
            - Use 'AcquisitionDate' to group images.
            - Match DR-CS images to open images within the same AcquisitionDate.
            - Perform normalization by dividing DR-CS ROI stats by the open field ROI
              stats.

    4. Additional Statistics:
        - Calculate additional statistics such as `Average` and `Max vs Min` for each
          image.

    Args:
        dirpath (pathlib.Path): Path to the directory containing DICOM files.
        roi_config (dict): Dictionary containing ROI configuration parameters and
            RTImageLabels.
        inspect_mode (str, optional): If 'live', displays images with ROIs overlaid
            during processing. If 'save', saves images with ROIs overlaid to files.
            Defaults to None.
        normalize (bool, optional): If `True`, normalizes DR-CS ROI stats by the
            normalization field stats. Defaults to `False`.

    Returns:
        pandas.DataFrame: A DataFrame containing ROI statistics for all images,
            including any normalized statistics if requested.

    Raises:
        FileNotFoundError: If the configuration file is missing when normalization is
            requested.
        ValueError: If the configuration file has invalid JSON format.
    """
    roi_stats_all = []
    dicom_fpaths = list(dirpath.glob("*.dcm"))
    dicom_fpath_count = len(dicom_fpaths)

    if inspect_mode == "save":
        output_image_dir = dirpath / "output_images"
        output_image_dir.mkdir(exist_ok=True)
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

        # Get RTImageLabel and make it lowercase for case-insensitive comparison
        rt_image_label = getattr(ds, "RTImageLabel", "").lower()

        # Get AcquisitionDate
        acquisition_date = getattr(ds, "AcquisitionDate", None)
        if acquisition_date is None:
            logger.warning("Missing 'AcquisitionDate' in file '%s'", dicom_fpath.name)
            continue

        # Determine image type based on RTImageLabel
        if rt_image_label in roi_config["drcs_rtimage_labels"]:
            image_type = "DRCS"
        elif rt_image_label in roi_config["open_rtimage_labels"]:
            image_type = "OPEN"
        else:
            logger.warning(
                "Unrecognized RTImageLabel '%s' in file '%s'",
                rt_image_label,
                dicom_fpath.name,
            )
            image_type = "UNKNOWN"

        # Skip images with unrecognized RTImageLabel
        if image_type == "UNKNOWN":
            continue

        # Adjust ROI angles for "_A" files (rotated images)
        roi_config_copy = copy.deepcopy(roi_config)
        if "_A" in dicom_fpath.stem:
            for roi in roi_config_copy["roi_list"]:
                roi["roi_angle"] = (roi["roi_angle"] - 180) % 360

        try:
            # Get ROI statistics for the current image
            roi_stats = get_roi_stats(
                ds,
                roi_config_copy,
                inspect_mode=inspect_mode,
                output_dir=output_image_dir,
                image_name=dicom_fpath.stem,
            )
            roi_stats["File"] = dicom_fpath.stem  # Add the file name to the stats
            roi_stats["RTImageLabel"] = rt_image_label
            roi_stats["ImageType"] = image_type
            roi_stats["AcquisitionDate"] = acquisition_date
            roi_stats_all.append(roi_stats)
        except AttributeError as e:
            logger.error("Error processing file %s: %s", dicom_fpath.name, e)
            continue

    # Compile the statistics into a DataFrame
    df = pd.DataFrame(roi_stats_all)

    # Check if the DataFrame is empty
    if df.empty:
        logger.warning("No valid images were processed; the DataFrame is empty.")
        return df  # Return the empty DataFrame to avoid further processing

    for col in ROI_LABELS:
        df[col] = pd.to_numeric(df[col])

    # If normalize is True, process normalization
    if normalize:
        # Group the DataFrame by 'AcquisitionDate'
        group_key = "AcquisitionDate"
        grouped = df.groupby(group_key)

        # Initialize a list to store normalized data
        normalized_data = []

        for group_id, group in grouped:
            # Check if both DRCS and OPEN images are present in the group
            image_types = set(group["ImageType"])
            if "OPEN" not in image_types:
                logger.warning(
                    "Open image not found for AcquisitionDate '%s'", group_id
                )
                continue
            open_images = group[group["ImageType"] == "OPEN"]
            open_image = open_images.iloc[0]

            drcs_images = group[group["ImageType"] == "DRCS"]

            for _, drcs_image in drcs_images.iterrows():
                # Exclude non-ROI columns
                non_roi_columns = ["File", "RTImageLabel", "ImageType", group_key]
                roi_columns = [
                    col
                    for col in df.columns
                    if col not in non_roi_columns and not df[col].dtype == "object"
                ]

                # Perform element-wise division of DRCS ROI stats by OPEN ROI stats
                normalized_values = drcs_image[roi_columns] / open_image[roi_columns]

                # Create a new row as a dictionary with both non-ROI and ROI columns
                norm_row_dict = drcs_image[non_roi_columns].to_dict()
                norm_row_dict["ImageType"] = "NORMALIZED"
                for col in roi_columns:
                    norm_row_dict[col] = normalized_values[col]

                normalized_data.append(norm_row_dict)
        # Create a DataFrame from the normalized data
        df_normalized = pd.DataFrame(normalized_data)

        # Append the normalized data to the original DataFrame
        df = pd.concat([df, df_normalized], ignore_index=True)

    # Now, compute additional statistics like 'Average' and 'Max vs Min'
    # Exclude non-ROI columns
    non_roi_columns = ["File", "RTImageLabel", "ImageType", "AcquisitionDate"]
    roi_columns = [
        col
        for col in df.columns
        if col not in non_roi_columns and not df[col].dtype == "object"
    ]

    if roi_columns:
        # Perform calculations
        df["Average"] = df[roi_columns].mean(axis=1)
        min_values = df[roi_columns].min(axis=1).replace(0, np.nan)
        df["Max vs Min"] = df[roi_columns].max(axis=1) / min_values - 1
        df.fillna({"Max vs Min": np.inf}, inplace=True)
    else:
        logger.warning("No ROI columns found in the DataFrame; skipping calculations.")

    # Reorder columns to have 'File' as the leftmost column
    desired_order = (
        ["File", "RTImageLabel", "ImageType", "AcquisitionDate"]
        + roi_columns
        + ["Average", "Max vs Min"]
    )
    # Ensure all desired columns are present in the DataFrame
    ordered_columns = [col for col in desired_order if col in df.columns]
    # Add any other columns that are not specified in desired_order
    other_columns = [col for col in df.columns if col not in ordered_columns]
    # Reorder the DataFrame columns
    df = df[ordered_columns + other_columns]

    return df


def open_file(filepath):
    """
    Open a file using the default application based on the operating system.

    Args:
        filepath (pathlib.Path): The path to the file to be opened.
    """
    if platform.system() == "Windows":
        os.startfile(filepath)
    elif platform.system() == "Darwin":  # macOS
        subprocess.run(["open", filepath], check=True)
    else:  # Linux and other Unix systems
        subprocess.run(["xdg-open", filepath], check=True)


def main():
    """
    Main function to execute the DR-CS analysis workflow.

    This function orchestrates the processing of DICOM images in a specified directory
    using the ROI configurations provided in a `config.json` file located in the same
    directory. It calculates statistics for each ROI and outputs the results to CSV and
    Excel files.

    Features:

    - Processes all DICOM images in the input directory.
    - Optionally displays images with ROIs overlaid for visual inspection.
    - Optionally normalizes DR-CS ROI statistics by normalization field statistics
      based on image pairs.
    - Optionally opens the output CSV or Excel file after processing.

    Command-line arguments:

    - `input_directory` (str): Path to the input directory containing DICOM files and
      `config.json`.
    - `--inspect-live`: If specified, displays images with ROIs overlaid during
      processing.
    - `--inspect-save`: If specified, saves images with ROIs overlaid to files.
    - `--normalize`: If specified, normalizes DR-CS ROI stats by the normalization
      field stats.
    - `--open-csv`: If specified, automatically opens the output CSV file after
      processing.
    - `--open-excel`: If specified, automatically opens the output Excel file after
      processing.
    """
    # Set up argument parser to receive input directory path
    parser = argparse.ArgumentParser(description="Process DICOM images in a directory.")
    parser.add_argument(
        "input_directory",
        type=str,
        help="Path to the input directory containing DICOM files.",
    )
    parser.add_argument(
        "--inspect-live",
        action="store_true",
        help="Display images with ROIs overlaid during processing.",
    )
    parser.add_argument(
        "--inspect-save",
        action="store_true",
        help="Save images with ROIs overlaid to files.",
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
    config_path = data_dirpath / "config.json"

    # Determine inspect mode
    inspect_mode = None
    if args.inspect_live:
        inspect_mode = "live"
    elif args.inspect_save:
        inspect_mode = "save"

    # Load ROI configuration
    try:
        roi_config = load_roi_config(config_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Error loading ROI configuration: %s", e)
        sys.exit(1)

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

    csv_savepath = data_dirpath / "roi_stats.csv"
    excel_savepath = data_dirpath / "roi_stats.xlsx"

    # Save the results
    df.to_csv(csv_savepath, index=False)
    df.to_excel(excel_savepath, index=False)

    logger.info(
        "Processing complete. Results saved to:\n\t%s\n\t%s",
        csv_savepath,
        excel_savepath,
    )

    if args.open_csv:
        open_file(csv_savepath)

    if args.open_excel:
        open_file(excel_savepath)


if __name__ == "__main__":
    main()
