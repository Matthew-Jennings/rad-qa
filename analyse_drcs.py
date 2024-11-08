"""
Module for analyzing the Dose Rate - Collimator Speed (DR-CS) test for Varian TrueBeam
radiotherapy systems.

This script processes DICOM images (predicted or acquired on a TrueBeam EPID) to
calculate statistics for five Regions of Interest (ROIs) defined in a configuration
file. It outputs the results to CSV and Excel files and provides options for visual
inspection of the ROIs overlaid on the images.

Optionally, it can normalize the DR-CS ROI statistics by normalization field statistics
when analyzing image pairs. The script is designed to facilitate quality assurance by
automating the analysis of DR-CS tests.

**Usage:**

```bash
python analyse_drcs.py input_directory [--inspect] [--normalize] [--open-output]

**Arguments:**

- input_directory: Path to the input directory containing DICOM files and a config.json
file with ROI configurations.

**Optional arguments:**

- --inspect: Display images with ROIs overlaid for visual inspection.
- --normalize: Normalize DR-CS ROI stats by the normalization field stats.
- --open-output: Automatically open the output Excel file after processing.


Copyright (c) 2024, Matthew Jennings, Icon Group.
"""

import argparse
import copy
import os
import pathlib
import json

import pandas as pd
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon

NUM_ROIS = 5


def get_rotated_rectangle_vertices(cx, cy, width, height, angle_deg):
    """
    Compute the vertices of a rotated rectangle given its center, dimensions, and rotation angle.

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
    R = np.array(
        [
            [np.cos(angle_rad), np.sin(angle_rad)],
            [-np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )

    # Rotate corners
    rotated_corners = corners @ R.T

    # Translate to center position
    vertices = rotated_corners + np.array([cx, cy])

    return vertices


def load_roi_config(config_path):
    """
    Load ROI configuration parameters from a JSON file.

    The configuration file should contain shared parameters and lists of angles,
    colors, and labels for each of the five ROIs. The shared parameters include the
    center offset, width, and height of the ROIs. The lists provide specific attributes
    for each ROI.

    **Expected JSON structure:**

    ```json
    {
        "roi_center_offset_from_image_centre_mm": <float>,
        "roi_width_mm": <float>,
        "roi_height_mm": <float>,
        "roi_angles": [<float>, <float>, ..., <float>],     // List of length NUM_ROIS
        "roi_colors": [<str>, <str>, ..., <str>],           // List of length NUM_ROIS
        "roi_labels": [<str>, <str>, ..., <str>]            // List of length NUM_ROIS
    }
    ```

    Args:
        config_path (pathlib.Path): Path to the configuration JSON file.

    Returns:
        list: A list of dictionaries, each containing the parameters for one ROI.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the configuration file is invalid or missing required keys.
    """
    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        # Validate configuration
        required_shared_keys = {
            "roi_center_offset_from_image_centre_mm",
            "roi_width_mm",
            "roi_height_mm",
        }
        required_list_keys = {"roi_angles", "roi_colors", "roi_labels"}

        # Check for required keys
        missing_shared_keys = required_shared_keys - set(config.keys())
        missing_list_keys = required_list_keys - set(config.keys())

        if missing_shared_keys:
            raise ValueError(f"Missing required shared keys: {missing_shared_keys}")
        if missing_list_keys:
            raise ValueError(f"Missing required list keys: {missing_list_keys}")

        # Check all lists have length 5
        list_lengths = {key: len(config[key]) for key in required_list_keys}
        if not all(length == NUM_ROIS for length in list_lengths.values()):
            raise ValueError(
                f"All lists must have length {NUM_ROIS}. Current lengths: {list_lengths}"
            )

        # Convert configuration format to list of dictionaries
        roi_config = []
        for i in range(NUM_ROIS):
            roi_dict = {
                "roi_center_offset_from_image_centre_mm": config[
                    "roi_center_offset_from_image_centre_mm"
                ],
                "roi_width_mm": config["roi_width_mm"],
                "roi_height_mm": config["roi_height_mm"],
                "roi_angle": config["roi_angles"][i],
                "roi_color": config["roi_colors"][i],
                "roi_label": config["roi_labels"][i],
            }
            roi_config.append(roi_dict)

        return roi_config

    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON format in configuration file {config_path}"
        ) from exc


def get_roi_stats(ds, roi_config, inspect=False):
    """
    Calculate statistics for Regions of Interest (ROIs) in a DICOM image.

    This function processes a DICOM image to extract mean pixel values within specified
    ROIs. It handles image scaling to ensure pixel values are in consistent units
    (cGy/MU). The ROIs are defined based on a configuration that specifies their
    positions relative to the image center, dimensions, and rotation angles.

    **Key steps:**

    1. **Image Extraction and Scaling:**
    - Extract the image data from the DICOM dataset.
    - Apply scaling factors (`RescaleSlope` and `RescaleIntercept`) to convert raw
      pixel data to physical units.
    - Convert dose units from Gy to cGy.
    - If the image is an acquired dose, normalize by the `MetersetExposure` to obtain
      cGy/MU.

    2. **ROI Processing:**
    - For each ROI in the configuration:
        - Calculate the center position in image coordinates, accounting for pixel
          spacing and rotation.
        - Determine the vertices of the rotated rectangle representing the ROI.
        - Create a mask for the ROI and extract the pixel values within it.
        - Compute the mean pixel value for the ROI.

    3. **Visualization (Optional):**
    - If `inspect` is `True`, display the image with ROIs overlaid for visual
      inspection.

    Args:
        ds (pydicom.dataset.FileDataset): DICOM dataset containing the image and
            metadata.
        roi_config (list): List of dictionaries containing ROI configuration parameters.
        inspect (bool, optional): If `True`, displays the image with ROIs overlaid for
            visual inspection. Defaults to `False`.

    Returns:
        dict: A dictionary where keys are ROI labels and values are the mean pixel
            values within each ROI.

    Raises:
        AttributeError: If necessary DICOM attributes are missing.
    """
    # Extract image data
    image = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
    image *= 100  # Gy to cGy

    if ds.ImageType[3] == "ACQUIRED_DOSE":
        image /= ds.ExposureSequence[0].MetersetExposure  # cGy / MU

    image_height, image_width = image.shape
    image_center_x, image_center_y = image_width / 2, image_height / 2

    row_spacing_mm = float(ds.ImagePlanePixelSpacing[0])
    col_spacing_mm = float(ds.ImagePlanePixelSpacing[1])

    # Prepare to display the image
    if inspect:
        _, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image, cmap="gray")

    # Store statistics
    roi_stats = {}

    for roi in roi_config:
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

        if inspect:
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

    if inspect:
        plt.title("DICOM Image with Rotated ROIs")
        plt.axis("off")
        plt.show()

    return roi_stats


def get_roi_stats_for_images_in_dir(
    dirpath, roi_config, inspect=False, normalize=False
):
    """
    Process all DICOM images in a directory to calculate ROI statistics.

    This function processes each DICOM file in the specified directory to calculate
    mean pixel values within specified ROIs. It compiles the results into a pandas
    DataFrame. Optionally, it can normalize the DR-CS ROI statistics by corresponding
    normalization field statistics.

    **Key steps:**

    1. **Iterate Over DICOM Files:**
    - For each DICOM file in the directory:
        - Read the DICOM dataset.
        - Adjust ROI angles for rotated images if necessary.
        - Calculate ROI statistics using `get_roi_stats`.
        - Store the results along with the file name.

    2. **DataFrame Creation:**
    - Compile the collected statistics into a pandas DataFrame.
    - Extract `BaseName` and `Identifier` from the file names, assuming they are in the
      format `<name>.<identifier>`.

    3. **Normalization (Optional):**
    - If normalization is requested:
        - Read normalization identifiers from the configuration file (`config.json`).
        - For each base name, check if both the DR-CS field and normalization field
          images are present.
        - Perform element-wise division of the DR-CS ROI stats by the normalization
          field stats.
        - Add the normalized results to the DataFrame.

    4. **Additional Statistics:**
    - Calculate additional statistics such as `Average` and `Max vs Min` for each image.

    Args:
        dirpath (pathlib.Path): Path to the directory containing DICOM files.
        roi_config (list): List of dictionaries containing ROI configuration parameters.
        inspect (bool, optional): If `True`, displays images with ROIs overlaid during
            processing. Defaults to `False`.
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

    for i, dicom_fpath in enumerate(dicom_fpaths, start=1):
        print(f"Analyzing DICOM file {i} of {dicom_fpath_count}: {dicom_fpath.name}")
        ds = pydicom.dcmread(dicom_fpath)

        # Create a deep copy to avoid modifying the original roi_config
        roi_config_copy = copy.deepcopy(roi_config)

        # Adjust ROI angles for "_A" files (rotated images)
        if "_A" in dicom_fpath.stem:
            for roi in roi_config_copy:
                roi["roi_angle"] = (roi["roi_angle"] - 180) % 360

        # Get ROI statistics for the current image
        roi_stats = get_roi_stats(ds, roi_config_copy, inspect=inspect)
        roi_stats["File"] = dicom_fpath.stem  # Add the file name to the stats
        roi_stats_all.append(roi_stats)

    # Compile the statistics into a DataFrame
    df = pd.DataFrame(roi_stats_all)

    # Use the 'File' column to extract basenames and identifiers
    # Assuming file names are in the format <name>.<identifier>
    df[["BaseName", "Identifier"]] = df["File"].str.rsplit(".", n=1, expand=True)
    df["Identifier"] = df["Identifier"].str.lower()  # Ensure identifiers are lowercase

    # If normalize is True, process normalization
    if normalize:
        # Read the identifiers from the JSON config
        config_path = dirpath / "config.json"
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            normalization_identifiers = config.get("normalization_identifiers", {})

        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Configuration file not found at {config_path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON format in configuration file {config_path}"
            ) from exc

        # Get the normalization identifiers from the config
        normalization_identifiers = config.get("normalization_identifiers", {})
        drcs_field = normalization_identifiers.get("drcs_field", "rad").lower()
        norm_field = normalization_identifiers.get("norm_field", "open").lower()

        # Initialize a list to store normalized data
        normalized_data = []

        # Group the DataFrame by 'BaseName'
        grouped = df.groupby("BaseName")

        for base_name, group in grouped:
            # Check if both drcs_field and norm_field images are present
            if {drcs_field, norm_field}.issubset(set(group["Identifier"])):
                # Get the rows for drcs_field and norm_field
                dr_row = group[group["Identifier"] == drcs_field].iloc[0]
                norm_row = group[group["Identifier"] == norm_field].iloc[0]

                # Exclude non-ROI columns
                non_roi_columns = ["File", "BaseName", "Identifier"]
                roi_columns = [
                    col
                    for col in df.columns
                    if col not in non_roi_columns and not df[col].dtype == "object"
                ]

                # Perform element-wise division of drcs_field by norm_field ROI stats
                normalized_values = dr_row[roi_columns] / norm_row[roi_columns]

                # Create a new row with 'Identifier' = 'DR_NORM'
                norm_row_dict = {
                    "File": f"{base_name}.DR_NORM",
                    "BaseName": base_name,
                    "Identifier": "DR_NORM",
                }
                norm_row_dict.update(normalized_values.to_dict())

                normalized_data.append(norm_row_dict)
            else:
                print(
                    f"Warning: Missing '{drcs_field}' or '{norm_field}' image for base name '{base_name}'"
                )

        # Create a DataFrame from the normalized data
        df_normalized = pd.DataFrame(normalized_data)

        # Append the normalized data to the original DataFrame
        df = pd.concat([df, df_normalized], ignore_index=True)

        df.drop(columns=["BaseName", "Identifier"], inplace=True)

    # Now, compute additional statistics like 'Average' and 'Max vs Min'
    # Exclude non-ROI columns
    non_roi_columns = ["File", "BaseName", "Identifier"]
    roi_columns = [
        col
        for col in df.columns
        if col not in non_roi_columns and not df[col].dtype == "object"
    ]

    df["Average"] = df[roi_columns].mean(axis=1)
    df["Max vs Min"] = df[roi_columns].max(axis=1) / df[roi_columns].min(axis=1) - 1

    return df


def main():
    """
    Main function to execute the DR-CS analysis workflow.

    This function orchestrates the processing of DICOM images in a specified directory
    using the ROI configurations provided in a `config.json` file located in the same
    directory. It calculates statistics for each ROI and outputs the results to CSV and
    Excel files.

    **Features:**

    - Processes all DICOM images in the input directory.
    - Optionally displays images with ROIs overlaid for visual inspection.
    - Optionally normalizes DR-CS ROI statistics by normalization field statistics
      based on image pairs.
    - Optionally opens the output CSV or Excel file after processing.

    **Command-line arguments:**

    - `input_directory` (str): Path to the input directory containing DICOM files and
      `config.json`.
    - `--inspect`: If specified, displays images with ROIs overlaid during processing.
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
        "--inspect",
        action="store_true",
        help="Display images with ROIs overlaid.",
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

    # Load ROI configuration
    roi_config = load_roi_config(config_path)

    df = get_roi_stats_for_images_in_dir(
        data_dirpath, roi_config, inspect=args.inspect, normalize=args.normalize
    )

    csv_savepath = data_dirpath / "roi_stats.csv"
    excel_savepath = data_dirpath / "roi_stats.xlsx"

    # Save the results
    df.to_csv(csv_savepath)
    df.to_excel(excel_savepath)

    print(
        f"Processing complete. Results saved to:\n\t{csv_savepath}\n\t{excel_savepath}"
    )

    if args.open_csv:
        os.startfile(csv_savepath)

    if args.open_excel:
        os.startfile(excel_savepath)


if __name__ == "__main__":
    main()
