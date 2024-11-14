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

import os
import pathlib
import platform
import sys
from unittest import mock

import matplotlib
import numpy as np
import pandas as pd
import pydicom
import pytest

import analyse_drcs

matplotlib.use("Agg")

HERE = pathlib.Path(__file__).parent
SCRIPT_DIR = HERE.parent
DATA_DIR = HERE / "data"

sys.path.append(str(SCRIPT_DIR))

EXPECTED_COLUMN_TYPES = {
    "A": np.float64,
    "B": np.float64,
    "C": np.float64,
    "D": np.float64,
    "E": np.float64,
    "File": str,
    "AcquisitionDate": str,
    "ImageType": str,
    "RTImageLabel": str,
    "Average": np.float64,
    "Max vs Min": np.float64,
}

# Typical values for ROI "C".
TEST_WIDTH, TEST_HEIGHT = 102, 25
TEST_CX, TEST_CY = 512.0, 369.55


@pytest.fixture
def roi_config():
    """Load ROI configuration from the test data directory."""
    return analyse_drcs.load_roi_config(DATA_DIR / "config.json")


@pytest.fixture
def baseline_stats():
    """Load baseline statistics from CSV file."""
    # Read CSV with specified data types
    return pd.read_csv(DATA_DIR / "roi_stats_baseline.csv", dtype=EXPECTED_COLUMN_TYPES)


@pytest.fixture
def baseline_excel():
    """Load baseline statistics from Excel file."""
    return pd.read_excel(
        DATA_DIR / "roi_stats_baseline.xlsx", dtype=EXPECTED_COLUMN_TYPES
    )


@pytest.mark.parametrize(
    "angle_deg, expected_vertices",
    [
        (
            0.0,
            np.array(
                [
                    [461.0, 357.05],
                    [563.0, 357.05],
                    [563.0, 382.05],
                    [461.0, 382.05],
                ]
            ),
        ),
        (
            45.0,
            np.array(
                [
                    [484.77638892, 324.64871939],
                    [556.90128061, 396.77361108],
                    [539.22361108, 414.45128061],
                    [467.09871939, 342.32638892],
                ]
            ),
        ),
        (
            -45.0,
            np.array(
                [
                    [467.09871939, 396.77361108],
                    [539.22361108, 324.64871939],
                    [556.90128061, 342.32638892],
                    [484.77638892, 414.45128061],
                ]
            ),
        ),
    ],
)
def test_rotated_rectangle_vertices(angle_deg, expected_vertices):
    """Test the rectangle vertex calculation function for multiple rotation angles."""
    vertices = analyse_drcs.get_rotated_rectangle_vertices(
        TEST_CX,
        TEST_CY,
        TEST_WIDTH,
        TEST_HEIGHT,
        angle_deg,
    )
    assert np.allclose(
        vertices, expected_vertices
    ), f"Vertices do not match expected values at {angle_deg} degrees rotation."


def test_load_roi_config(roi_config):
    """Test loading ROI configuration from JSON."""
    roi_list = roi_config["roi_list"]
    assert len(roi_list) == analyse_drcs.NUM_ROIS
    assert all("roi_angle" in roi for roi in roi_list)
    assert all("roi_color" in roi for roi in roi_list)
    assert all("roi_label" in roi for roi in roi_list)

    # Test specific values from your config
    for roi in roi_list:
        assert isinstance(roi["roi_center_offset_from_image_centre_mm"], (int, float))
        assert isinstance(roi["roi_width_mm"], (int, float))
        assert isinstance(roi["roi_height_mm"], (int, float))
        assert isinstance(roi["roi_angle"], (int, float))
        assert isinstance(roi["roi_color"], str)
        assert isinstance(roi["roi_label"], str)

    # Check that 'open_rtimage_labels' and 'drcs_rtimage_labels' are lists of strings
    assert isinstance(roi_config["open_rtimage_labels"], list)
    assert isinstance(roi_config["drcs_rtimage_labels"], list)
    assert all(isinstance(label, str) for label in roi_config["open_rtimage_labels"])
    assert all(isinstance(label, str) for label in roi_config["drcs_rtimage_labels"])


def test_full_analysis_against_baseline(roi_config, baseline_stats):
    """
    Integration test comparing analysis results against baseline data.
    """
    # Run analysis
    df = analyse_drcs.get_roi_stats_for_images_in_dir(
        DATA_DIR, roi_config, inspect_mode=None, normalize=True
    )

    # Compare results with baseline, ensuring columns match
    common_columns = sorted(list(set(df.columns) & set(baseline_stats.columns)))
    pd.testing.assert_frame_equal(
        df[common_columns].sort_values(["File", "ImageType"]).reset_index(drop=True),
        baseline_stats[common_columns]
        .sort_values(["File", "ImageType"])
        .reset_index(drop=True),
    )


def test_excel_output(roi_config, baseline_excel):
    """Test Excel file output matches baseline."""
    # Run analysis
    df = analyse_drcs.get_roi_stats_for_images_in_dir(
        DATA_DIR, roi_config, inspect_mode=None, normalize=True
    )

    # Compare with baseline using common columns
    common_columns = sorted(list(set(df.columns) & set(baseline_excel.columns)))
    pd.testing.assert_frame_equal(
        df[common_columns].sort_values(["File", "ImageType"]).reset_index(drop=True),
        baseline_excel[common_columns]
        .sort_values(["File", "ImageType"])
        .reset_index(drop=True),
    )


@mock.patch("matplotlib.pyplot.show")
def test_inspect_live(mock_show, roi_config):
    """Test that inspect_live mode attempts to display images."""
    # Run analysis with inspect_mode='live'
    analyse_drcs.get_roi_stats_for_images_in_dir(
        DATA_DIR, roi_config, inspect_mode="live", normalize=False
    )

    # Verify that plt.show() was called at least once
    assert mock_show.called, "plt.show() was not called during inspect_live mode"


@mock.patch("matplotlib.figure.Figure.savefig")
def test_inspect_save(mock_savefig, tmp_path, roi_config):
    """Test that inspect_save mode saves images correctly."""
    # Create a temporary directory to save images
    temp_data_dir = tmp_path / "data"
    temp_data_dir.mkdir()
    temp_output_dir = temp_data_dir / "output_images"
    temp_output_dir.mkdir()

    # Copy DICOM files to temporary directory
    for dicom_file in DATA_DIR.glob("*.dcm"):
        target_file = temp_data_dir / dicom_file.name
        target_file.write_bytes(dicom_file.read_bytes())

    # Copy config.json to temporary directory
    config_file = DATA_DIR / "config.json"
    target_config_file = temp_data_dir / "config.json"
    target_config_file.write_text(config_file.read_text())

    # Run analysis with inspect_mode='save'
    df = analyse_drcs.get_roi_stats_for_images_in_dir(
        temp_data_dir, roi_config, inspect_mode="save", normalize=False
    )

    # Verify that savefig was called for each processed image
    expected_save_calls = df.shape[0]
    assert mock_savefig.call_count == expected_save_calls, (
        f"Expected savefig to be called {expected_save_calls} times, "
        f"but was called {mock_savefig.call_count} times."
    )


def test_normalization_multiple_drcs_per_open(roi_config):
    """
    Test normalization where multiple DR-CS images correspond to a single open image.
    """
    # Run analysis with normalize=True
    df = analyse_drcs.get_roi_stats_for_images_in_dir(
        DATA_DIR, roi_config, normalize=True
    )

    # Ensure ROI columns are numeric
    roi_columns = ["A", "B", "C", "D", "E"]
    df[roi_columns] = df[roi_columns].apply(pd.to_numeric, errors="coerce")

    # Group by AcquisitionDate
    grouped = df.groupby("AcquisitionDate")

    for _, group in grouped:
        open_images = group[group["ImageType"] == "OPEN"]
        drcs_images = group[group["ImageType"] == "DRCS"]
        normalized_images = group[group["ImageType"] == "NORMALIZED"]

        if open_images.empty or drcs_images.empty or normalized_images.empty:
            continue  # Skip if any image type is missing

        open_image = open_images.iloc[0]
        for _, drcs_image in drcs_images.iterrows():
            norm_image = normalized_images[
                normalized_images["File"] == drcs_image["File"]
            ]
            if norm_image.empty:
                continue  # Skip if normalized image not found
            norm_image = norm_image.iloc[0]

            # Extract ROI values and ensure they are numpy arrays of float type
            drcs_roi = pd.to_numeric(drcs_image[roi_columns])
            open_roi = pd.to_numeric(open_image[roi_columns])
            expected_normalized_roi = drcs_roi / open_roi

            actual_normalized_roi = pd.to_numeric(norm_image[roi_columns])

            # Check for invalid values
            if not np.isfinite(expected_normalized_roi).all():
                print(
                    f"Invalid values in expected_normalized_roi for file {drcs_image['File']}"
                )
                print(expected_normalized_roi)
            if not np.isfinite(actual_normalized_roi).all():
                print(
                    f"Invalid values in actual_normalized_roi for file {norm_image['File']}"
                )
                print(actual_normalized_roi)

            assert np.allclose(
                expected_normalized_roi, actual_normalized_roi, rtol=1e-5, atol=1e-8
            ), f"Normalization failed for file {norm_image['File']}"


def test_columns_order(roi_config):
    """Test that the 'File' column is the leftmost column."""
    df = analyse_drcs.get_roi_stats_for_images_in_dir(
        DATA_DIR, roi_config, normalize=False
    )

    # Get the first column
    first_column = df.columns[0]
    assert (
        first_column == "File"
    ), f"Expected 'File' as the first column, got '{first_column}'"

    # Check the full column order
    expected_order = [
        "File",
        "RTImageLabel",
        "ImageType",
        "AcquisitionDate",
        "A",
        "B",
        "C",
        "D",
        "E",
        "Average",
        "Max vs Min",
    ]
    for idx, col in enumerate(expected_order):
        if col in df.columns:
            assert (
                df.columns[idx] == col
            ), f"Expected column '{col}' at position {idx}, got '{df.columns[idx]}'"


@pytest.mark.skipif(
    not hasattr(os, "startfile"), reason="os.startfile is only available on Windows"
)
def test_open_file_windows(monkeypatch):
    """Test the open_file function on Windows."""
    filepath = pathlib.Path("test_file.csv")
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    with mock.patch("os.startfile") as mock_startfile:
        analyse_drcs.open_file(filepath)
        mock_startfile.assert_called_once_with(filepath)


def test_open_file_macos(monkeypatch):
    """Test the open_file function on macOS."""
    filepath = pathlib.Path("test_file.csv")
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    with mock.patch("subprocess.run") as mock_run:
        analyse_drcs.open_file(filepath)
        mock_run.assert_called_once_with(["open", filepath], check=True)


def test_open_file_linux(monkeypatch):
    """Test the open_file function on Linux."""
    filepath = pathlib.Path("test_file.csv")
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    with mock.patch("subprocess.run") as mock_run:
        analyse_drcs.open_file(filepath)
        mock_run.assert_called_once_with(["xdg-open", filepath], check=True)


def test_no_dicom_files(tmp_path, roi_config, caplog):
    """Test behavior when no DICOM files are present in the directory."""
    # Create an empty temporary directory
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    # Run analysis
    df = analyse_drcs.get_roi_stats_for_images_in_dir(
        empty_dir, roi_config, normalize=False
    )

    # Check that the DataFrame is empty
    assert df.empty, "Expected empty DataFrame when no DICOM files are present."

    # Check for log messages indicating no DICOM files were processed
    analysis_logs = [
        record for record in caplog.records if "Analyzing DICOM file" in record.message
    ]
    assert len(analysis_logs) == 0, "Expected no analysis logs for empty directory."


def test_unrecognized_rtimagelabels(tmp_path, roi_config, caplog):
    """Test behavior when DICOM images have unrecognized RTImageLabel."""
    # Create a temporary directory and copy DICOM files
    temp_data_dir = tmp_path / "data"
    temp_data_dir.mkdir()
    for dicom_file in DATA_DIR.glob("*.dcm"):
        target_file = temp_data_dir / dicom_file.name
        target_file.write_bytes(dicom_file.read_bytes())

    # Modify RTImageLabel in copied DICOM files to unrecognized labels
    for dicom_file in temp_data_dir.glob("*.dcm"):
        ds = pydicom.dcmread(dicom_file)
        ds.RTImageLabel = "UnknownLabel"
        ds.save_as(dicom_file)

    # Copy config.json to temporary directory
    config_file = DATA_DIR / "config.json"
    target_config_file = temp_data_dir / "config.json"
    target_config_file.write_text(config_file.read_text())

    # Run analysis and assert that no exceptions are raised
    try:
        df = analyse_drcs.get_roi_stats_for_images_in_dir(
            temp_data_dir, roi_config, normalize=False
        )
    except Exception as e:
        pytest.fail(f"Exception occurred during analysis: {e}")

    # Check that the DataFrame is empty
    assert df.empty, "Expected empty DataFrame when all RTImageLabels are unrecognized."

    # Check for warning logs
    warnings = [
        record
        for record in caplog.records
        if "Unrecognized RTImageLabel" in record.message
    ]
    assert len(warnings) >= 1, "Expected warnings for unrecognized RTImageLabels."


def test_missing_acquisition_date(tmp_path, roi_config, caplog):
    """Test behavior when DICOM images are missing the AcquisitionDate attribute."""
    # Create a temporary directory and copy DICOM files
    temp_data_dir = tmp_path / "data"
    temp_data_dir.mkdir()
    for dicom_file in DATA_DIR.glob("*.dcm"):
        target_file = temp_data_dir / dicom_file.name
        target_file.write_bytes(dicom_file.read_bytes())

    # Remove AcquisitionDate from copied DICOM files
    for dicom_file in temp_data_dir.glob("*.dcm"):
        ds = pydicom.dcmread(dicom_file)
        if hasattr(ds, "AcquisitionDate"):
            del ds.AcquisitionDate
            ds.save_as(dicom_file)

    # Copy config.json to temporary directory
    config_file = DATA_DIR / "config.json"
    target_config_file = temp_data_dir / "config.json"
    target_config_file.write_text(config_file.read_text())

    # Run analysis
    df = analyse_drcs.get_roi_stats_for_images_in_dir(
        temp_data_dir, roi_config, normalize=True
    )

    # Check that the DataFrame is empty since images without AcquisitionDate are skipped
    assert df.empty, "Expected empty DataFrame when images lack AcquisitionDate."

    # Check for warning logs
    warnings = [
        record
        for record in caplog.records
        if "Missing 'AcquisitionDate'" in record.message
    ]
    assert len(warnings) >= 1, "Expected warnings for missing 'AcquisitionDate'."


def test_normalization_no_open_image(tmp_path, roi_config, caplog):
    """
    Test normalization when no corresponding open image is available for a DRCS image.
    """
    # Create a temporary directory and copy DICOM files
    temp_data_dir = tmp_path / "data"
    temp_data_dir.mkdir()
    for dicom_file in DATA_DIR.glob("*.dcm"):
        ds = pydicom.dcmread(dicom_file)
        # Modify AcquisitionDate to a future date for DRCS images
        if getattr(ds, "RTImageLabel", "").lower() in roi_config["drcs_rtimage_labels"]:
            ds.AcquisitionDate = "20991231"
            ds.save_as(temp_data_dir / dicom_file.name)
        else:
            # Skip copying OPEN images
            pass

    # Copy config.json to temporary directory
    config_file = DATA_DIR / "config.json"
    target_config_file = temp_data_dir / "config.json"
    target_config_file.write_text(config_file.read_text())

    # Run analysis with normalize=True
    df = analyse_drcs.get_roi_stats_for_images_in_dir(
        temp_data_dir, roi_config, normalize=True
    )

    # Check that no normalized images are present
    normalized_df = df[df["ImageType"] == "NORMALIZED"]
    assert (
        normalized_df.empty
    ), "Expected no normalized images when no OPEN images are available."

    # Check that warnings are logged for missing OPEN images
    missing_open_warnings = [
        record for record in caplog.records if "Open image not found" in record.message
    ]
    assert len(missing_open_warnings) >= 1, "Expected warnings for missing OPEN images."


if __name__ == "__main__":
    pytest.main(["-v"])
