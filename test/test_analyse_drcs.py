# test_analyse_drcs.py
# Copyright (C) 2024 Matthew Jennings
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
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

# Expected column types for baseline DataFrame comparisons.
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

############################
# TEST FIXTURES
############################

@pytest.fixture
def roi_config():
    """
    Load ROI configuration from the test data directory.
    Assumes there's a 'config.json' in DATA_DIR with
    valid ROI definitions, open_rtimage_labels, drcs_rtimage_labels, etc.
    """
    return analyse_drcs.load_roi_config(DATA_DIR / "config.json")

@pytest.fixture
def baseline_stats():
    """Load baseline CSV statistics for direct comparison."""
    return pd.read_csv(DATA_DIR / "roi_stats_baseline.csv", dtype=EXPECTED_COLUMN_TYPES)

@pytest.fixture
def baseline_excel():
    """Load baseline Excel statistics for direct comparison."""
    return pd.read_excel(
        DATA_DIR / "roi_stats_baseline.xlsx",
        dtype=EXPECTED_COLUMN_TYPES,
    )

############################
# UNIT TESTS
############################

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
    """Unit test: verify corners for rotated rectangles at multiple angles."""
    TEST_WIDTH, TEST_HEIGHT = 102, 25
    TEST_CX, TEST_CY = 512.0, 369.55
    vertices = analyse_drcs.get_rotated_rectangle_vertices(
        TEST_CX,
        TEST_CY,
        TEST_WIDTH,
        TEST_HEIGHT,
        angle_deg,
    )
    assert np.allclose(vertices, expected_vertices), f"Vertices do not match expected at {angle_deg} deg rotation."

def test_load_roi_config(roi_config):
    """Unit test: validate ROI config structure."""
    roi_list = roi_config["roi_list"]
    assert len(roi_list) == analyse_drcs.NUM_ROIS
    assert all("roi_angle" in roi for roi in roi_list)
    assert all("roi_color" in roi for roi in roi_list)
    assert all("roi_label" in roi for roi in roi_list)
    for roi in roi_list:
        assert isinstance(roi["roi_center_offset_from_image_centre_mm"], (int, float))
        assert isinstance(roi["roi_width_mm"], (int, float))
        assert isinstance(roi["roi_height_mm"], (int, float))
        assert isinstance(roi["roi_angle"], (int, float))
        assert isinstance(roi["roi_color"], str)
        assert isinstance(roi["roi_label"], str)
    open_labels = roi_config["open_rtimage_labels"]
    drcs_labels = roi_config["drcs_rtimage_labels"]
    assert isinstance(open_labels, list)
    assert isinstance(drcs_labels, list)
    assert all(isinstance(label, str) for label in open_labels)
    assert all(isinstance(label, str) for label in drcs_labels)

############################
# Helper function tests
############################

def test_compute_roi_polygon():
    """Test compute_roi_polygon returns correctly shaped outputs."""
    image = np.zeros((1000, 1000), dtype=np.float32)
    roi = {
        "roi_center_offset_from_image_centre_mm": 50,
        "roi_width_mm": 20,
        "roi_height_mm": 10,
        "roi_angle": 30,
        "roi_color": "red",
        "roi_label": "A"
    }
    image_center = (500, 500)
    col_spacing_mm = 1.0
    row_spacing_mm = 1.0
    vertices, mask, rect_center_x, rect_center_y = analyse_drcs.compute_roi_polygon(
        image, roi, image_center, col_spacing_mm, row_spacing_mm
    )
    assert vertices.shape == (4, 2)
    assert mask.shape == image.shape
    assert isinstance(rect_center_x, float)
    assert isinstance(rect_center_y, float)

def test_adjust_roi_angles():
    """Test that filenames containing '_V' trigger a 180 deg adjustment in ROI angles."""
    dummy_config = {
        "roi_list": [{
            "roi_center_offset_from_image_centre_mm": 10,
            "roi_width_mm": 20,
            "roi_height_mm": 30,
            "roi_angle": 45,
            "roi_color": "blue",
            "roi_label": "A"
        }],
        "open_rtimage_labels": ["open"],
        "drcs_rtimage_labels": ["drcs"],
        "log_level": "INFO"
    }
    # Pass a filename with extension; adjust_roi_angles uses the stem.
    adjusted = analyse_drcs.adjust_roi_angles(dummy_config, "test_V.dcm")
    for roi in adjusted["roi_list"]:
        assert roi["roi_angle"] == 225  # (45-180)%360

def test_normalize_images():
    """Test normalization of DRCS ROI stats given matching OPEN image."""
    data = {
        "File": ["open1", "drcs1"],
        "RTImageLabel": ["open", "drcs"],
        "ImageType": ["OPEN", "DRCS"],
        "AcquisitionDate": ["20240101", "20240101"],
        "Group": ["acquired", "acquired"],
        "A": [100.0, 150.0],
        "B": [200.0, 300.0],
        "C": [300.0, 450.0],
        "D": [400.0, 600.0],
        "E": [500.0, 750.0],
    }
    df = pd.DataFrame(data)
    non_roi_columns = ["File", "RTImageLabel", "ImageType", "AcquisitionDate", "Group"]
    roi_columns = ["A", "B", "C", "D", "E"]
    normalized = analyse_drcs.normalize_images(
        df, "20240101", "acquired", "OPEN", "DRCS", non_roi_columns, roi_columns
    )
    assert len(normalized) == 1
    norm_row = normalized[0]
    expected = {"A": 1.5, "B": 1.5, "C": 1.5, "D": 1.5, "E": 1.5}
    for key, val in expected.items():
        assert abs(norm_row[key] - val) < 1e-6
    assert norm_row["ImageType"] == "NORMALIZED"

############################
# INTEGRATION TESTS
############################

def test_full_analysis_against_baseline(roi_config, baseline_stats):
    """
    Integration test: run ROI analysis on DICOM test data,
    compare results to previously generated baseline CSV.
    """
    df = analyse_drcs.get_roi_stats_for_images_in_dir(
        DATA_DIR,
        roi_config,
        inspect_mode=None,
        normalize=True,
    )
    if "Group" in df.columns:
        df = df.drop(columns=["Group"])
    common_columns = sorted(set(df.columns).intersection(baseline_stats.columns))
    df_sorted = df[common_columns].sort_values(["File", "ImageType"]).reset_index(drop=True)
    baseline_sorted = baseline_stats[common_columns].sort_values(["File", "ImageType"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(df_sorted, baseline_sorted, obj="Full analysis DataFrame vs. baseline CSV")

def test_excel_output(roi_config, baseline_excel):
    """
    Integration test: confirm Excel file output from the pipeline
    is consistent with baseline.
    """
    df = analyse_drcs.get_roi_stats_for_images_in_dir(
        DATA_DIR,
        roi_config,
        inspect_mode=None,
        normalize=True,
    )
    if "Group" in df.columns:
        df = df.drop(columns=["Group"])
    common_columns = sorted(set(df.columns).intersection(baseline_excel.columns))
    df_sorted = df[common_columns].sort_values(["File", "ImageType"]).reset_index(drop=True)
    baseline_sorted = baseline_excel[common_columns].sort_values(["File", "ImageType"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(df_sorted, baseline_sorted, obj="Excel DataFrame vs. baseline")

def test_inspect_live(roi_config):
    """
    Test that inspect_live mode attempts to display images (plt.show()).
    """
    with mock.patch("matplotlib.pyplot.show") as mock_show:
        analyse_drcs.get_roi_stats_for_images_in_dir(
            DATA_DIR,
            roi_config,
            inspect_mode="live",
            normalize=False,
        )
        assert mock_show.call_count > 0, "plt.show() not called in 'live' mode"

def test_inspect_save(roi_config, tmp_path):
    """
    Test that inspect_save mode saves overlaid images.
    """
    with mock.patch("matplotlib.figure.Figure.savefig") as mock_savefig:
        temp_data_dir = tmp_path / "data"
        temp_data_dir.mkdir()
        for dicom_file in DATA_DIR.glob("*.dcm"):
            (temp_data_dir / dicom_file.name).write_bytes(dicom_file.read_bytes())
        _ = analyse_drcs.get_roi_stats_for_images_in_dir(
            temp_data_dir,
            roi_config,
            inspect_mode="save",
            normalize=False,
        )
        expected_saves = len(_)
        assert mock_savefig.call_count == expected_saves, f"Expected {expected_saves} calls to savefig(), got {mock_savefig.call_count}"

############################
# SPECIAL CASE TESTS
############################

def test_multiple_drcs_per_single_open(roi_config):
    """
    Test normalization logic when multiple DRCS images share one OPEN image.
    """
    df = analyse_drcs.get_roi_stats_for_images_in_dir(
        DATA_DIR,
        roi_config,
        normalize=True,
    )
    drcs_df = df[df["ImageType"].isin(["DRCS", "DRCS PREDICTED"])]
    norm_df = df[df["ImageType"] == "NORMALIZED"]
    if not drcs_df.empty:
        for col in ["A", "B", "C", "D", "E", "Average", "Max vs Min"]:
            assert col in norm_df.columns
    else:
        pass

def test_file_suffix_for_rotation(roi_config, caplog, tmp_path):
    """
    Test that a filename containing '_V' triggers a 180 deg angle shift in each ROI.
    """
    caplog.set_level(logging.DEBUG)
    temp_data_dir = tmp_path / "data"
    temp_data_dir.mkdir()
    for dicom_file in DATA_DIR.glob("*.dcm"):
        new_name = dicom_file.stem + "_V.dcm"
        (temp_data_dir / new_name).write_bytes(dicom_file.read_bytes())
    _ = analyse_drcs.get_roi_stats_for_images_in_dir(temp_data_dir, roi_config)
    angle_logs = [rec for rec in caplog.records if "Adjusting ROI angles" in rec.message]
    assert len(angle_logs) > 0, "Expected debug log about adjusting ROI angles."

def test_warn_multiple_predicted(roi_config, caplog, tmp_path):
    """
    Test that a warning is logged if >1 predicted images exist for the same AcquisitionDate.
    """
    temp_data_dir = tmp_path / "data"
    temp_data_dir.mkdir()
    for dicom_file in DATA_DIR.glob("*.dcm"):
        ds = pydicom.dcmread(dicom_file)
        ds.ImageType = ["ORIGINAL", "PRIMARY", "", "FAKE_DOSE"]  # forces predicted (not ACQUIRED_DOSE)
        ds.RTImageLabel = "open"
        ds.AcquisitionDate = "20241212"
        for copy_i in range(2):
            out_path = temp_data_dir / f"{dicom_file.stem}_pred{copy_i}.dcm"
            ds.save_as(out_path)
    _ = analyse_drcs.get_roi_stats_for_images_in_dir(temp_data_dir, roi_config, normalize=True)
    multi_pred_logs = [rec for rec in caplog.records if "More than one 'OPEN PREDICTED'" in rec.message]
    assert len(multi_pred_logs) > 0, "Expected warning about multiple predicted images per date."

def test_columns_order(roi_config):
    """
    Test that columns appear in expected order.
    """
    df = analyse_drcs.get_roi_stats_for_images_in_dir(DATA_DIR, roi_config, normalize=False)
    if "Group" in df.columns:
        df = df.drop(columns=["Group"])
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
    for i, col in enumerate(expected_order):
        if col in df.columns:
            assert df.columns[i] == col, f"Expected '{col}' at position {i}, got '{df.columns[i]}'"

############################
# OPEN FILE FUNCTION TESTS
############################

import platform

@pytest.mark.skipif(
    not hasattr(os, "startfile"), reason="os.startfile is only available on Windows"
)
def test_open_file_windows(monkeypatch):
    """Test the open_file function on Windows."""
    filepath = pathlib.Path("test_file.csv")
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    with mock.patch("os.startfile") as mock_startfile:
        analyse_drcs.open_file(filepath)
        mock_startfile.assert_called_once_with(str(filepath))

def test_open_file_macos(monkeypatch):
    """Test the open_file function on macOS."""
    filepath = pathlib.Path("test_file.csv")
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    with mock.patch("subprocess.run") as mock_run:
        analyse_drcs.open_file(filepath)
        mock_run.assert_called_once_with(["open", str(filepath)], check=True)

def test_open_file_linux(monkeypatch):
    """Test the open_file function on Linux."""
    filepath = pathlib.Path("test_file.csv")
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    with mock.patch("subprocess.run") as mock_run:
        analyse_drcs.open_file(filepath)
        mock_run.assert_called_once_with(["xdg-open", str(filepath)], check=True)

############################
# EDGE CASE TESTS
############################

def test_no_dicom_files(tmp_path, roi_config, caplog):
    """Test behavior when no DICOM files are present."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    df = analyse_drcs.get_roi_stats_for_images_in_dir(empty_dir, roi_config, normalize=False)
    assert df.empty, "Expected an empty DataFrame if no DICOM files are present."
    analysis_logs = [r for r in caplog.records if "Analyzing DICOM file" in r.message]
    assert len(analysis_logs) == 0, "No logs for empty directory expected."

def test_unrecognized_rtimagelabels(tmp_path, roi_config, caplog):
    """Test behavior when DICOM images have unrecognized RTImageLabel."""
    temp_data_dir = tmp_path / "data"
    temp_data_dir.mkdir()
    for dicom_file in DATA_DIR.glob("*.dcm"):
        ds = pydicom.dcmread(dicom_file)
        ds.RTImageLabel = "UnknownLabel"
        out_path = temp_data_dir / dicom_file.name
        ds.save_as(out_path)
    df = analyse_drcs.get_roi_stats_for_images_in_dir(temp_data_dir, roi_config, normalize=False)
    assert df.empty, "No recognized images if RTImageLabel is unknown."
    warnings = [r for r in caplog.records if "Unrecognized RTImageLabel" in r.message]
    assert len(warnings) == len(list(temp_data_dir.glob("*.dcm"))), "Expected warnings for each unrecognized label."

def test_missing_acquisition_date(tmp_path, roi_config, caplog):
    """Test behavior when DICOM images lack AcquisitionDate (required for normalization)."""
    temp_data_dir = tmp_path / "data"
    temp_data_dir.mkdir()
    for dicom_file in DATA_DIR.glob("*.dcm"):
        ds = pydicom.dcmread(dicom_file)
        if hasattr(ds, "AcquisitionDate"):
            del ds.AcquisitionDate
        ds.save_as(temp_data_dir / dicom_file.name)
    df = analyse_drcs.get_roi_stats_for_images_in_dir(temp_data_dir, roi_config, normalize=True)
    assert df.empty, "Expected empty DataFrame if images are skipped due to no AcquisitionDate."
    warnings = [r for r in caplog.records if "Missing 'AcquisitionDate'" in r.message]
    assert len(warnings) > 0, "Expected warnings about missing AcquisitionDate."

def test_no_open_images_for_normalization(tmp_path, roi_config, caplog):
    """
    Test behavior when DRCS images exist but no corresponding OPEN images exist.
    Expect no 'NORMALIZED' rows and warnings about missing OPEN.
    """
    temp_data_dir = tmp_path / "data"
    temp_data_dir.mkdir()
    for dicom_file in DATA_DIR.glob("*.dcm"):
        ds = pydicom.dcmread(dicom_file)
        ds.RTImageLabel = roi_config["drcs_rtimage_labels"][0]
        ds.AcquisitionDate = "20240501"
        ds.save_as(temp_data_dir / dicom_file.name)
    df = analyse_drcs.get_roi_stats_for_images_in_dir(temp_data_dir, roi_config, normalize=True)
    assert df.empty or "NORMALIZED" not in df["ImageType"].values, "Expected no NORMALIZED rows if no OPEN images exist."
    missing_open_warnings = [r for r in caplog.records if "Missing 'OPEN'" in r.message]
    missing_open_pred_warnings = [r for r in caplog.records if "Missing 'OPEN PREDICTED'" in r.message]
    assert len(missing_open_warnings) > 0 or len(missing_open_pred_warnings) > 0, "Expected warning logs about missing open images."

if __name__ == "__main__":
    pytest.main(["-v"])
