import pathlib

import pandas as pd
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon


def get_rotated_rectangle_vertices(cx, cy, width, height, angle_deg):
    """
    Compute the vertices of a rotated rectangle.

    Parameters:
    - cx, cy: Center coordinates of the rectangle
    - width, height: Dimensions of the rectangle
    - angle_deg: Rotation angle in degrees

    Returns:
    - vertices: Array of x, y coordinates of the rectangle corners
    """
    # Convert angle to radians
    angle_rad = np.deg2rad(angle_deg)

    # Define half-dimensions
    w = width / 2
    h = height / 2

    # Corners relative to center
    corners = np.array([[-w, -h], [w, -h], [w, h], [-w, h]])

    # Rotation matrix
    R = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )

    # Rotate corners
    rotated_corners = corners @ R.T

    # Translate to center position
    vertices = rotated_corners + np.array([cx, cy])

    return vertices


def get_roi_stats(
    ds,
    center_offset=150 + (33 / 2),
    roi_width=125,
    roi_height=30,
    roi_angles=(150, 210, 270, 330, 30),
    roi_colors=("blue", "red", "yellow", "purple", "green"),
    roi_labels=("A", "B", "C", "D", "E"),
    inspect=False,
):
    # Extract image data
    image = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept

    height, width = image.shape
    center_x, center_y = width / 2, height / 2

    # Prepare to display the image
    if inspect:
        _, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image, cmap="gray")

    # Store statistics
    roi_stats = {}

    for idx, angle in enumerate(roi_angles):
        # Convert angle to radians
        theta_rad = np.deg2rad(angle)

        # Compute rectangle center positions
        rect_center_x = center_x + center_offset * np.cos(theta_rad)
        rect_center_y = center_y + center_offset * np.sin(theta_rad)

        # Get rectangle vertices
        vertices = get_rotated_rectangle_vertices(
            rect_center_x, rect_center_y, roi_width, roi_height, angle
        )

        # Ensure vertices are within image bounds
        vertices[:, 0] = np.clip(vertices[:, 0], 0, width - 1)
        vertices[:, 1] = np.clip(vertices[:, 1], 0, height - 1)

        # Create a mask for the ROI
        mask = np.zeros_like(image, dtype=bool)
        rr, cc = polygon(vertices[:, 1], vertices[:, 0], image.shape)
        mask[rr, cc] = True

        # Extract pixel values within the ROI
        roi_pixels = image[mask]

        # Compute statistics
        roi_stats[roi_labels[idx]] = np.mean(roi_pixels)

        if inspect:
            # Overlay the ROI on the image
            polygon_patch = plt.Polygon(
                vertices,
                closed=True,
                edgecolor=roi_colors[idx],
                facecolor="none",
                linewidth=2,
            )
            ax.add_patch(polygon_patch)

            ax.text(
                rect_center_x,
                rect_center_y,
                roi_labels[idx],
                color=roi_colors[idx],
                fontsize=10,
                ha="center",
                va="center",
                fontweight="bold",
                # backgroundcolor='white'  # Optional for better visibility
            )

    if inspect:
        plt.title("DICOM Image with Rotated ROIs")
        plt.axis("off")
        plt.show()

    return roi_stats


def get_roi_stats_for_images_in_dir(dirpath):
    roi_stats_all = []
    dicom_fpaths = list(dirpath.glob("*.dcm"))
    dicom_fpath_count = len(dicom_fpaths)
    for i, dicom_fpath in enumerate(dicom_fpaths, start=1):
        print(f"Analysing DICOM file {i} of {dicom_fpath_count}")
        ds = pydicom.dcmread(dicom_fpath)

        roi_angles = np.array([150, 210, 270, 330, 30])
        if "_A" in dicom_fpath.stem:
            roi_angles -= 180

        roi_stats = get_roi_stats(ds, roi_angles=roi_angles, inspect=True)
        roi_stats["Image"] = dicom_fpath.stem
        roi_stats_all.append(roi_stats)

    return pd.DataFrame(roi_stats_all)


if __name__ == "__main__":
    HERE = pathlib.Path(__file__).parent

    DATA_DIRPATH = HERE / "NWS_OSBURN"

    df = get_roi_stats_for_images_in_dir(DATA_DIRPATH)

    # df.to_csv(HERE / "test_icon_results.csv")
    # df.to_excel(HERE / "test_icon_results.xlsx")
