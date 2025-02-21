# Analyse DR-CS

Analyse DR-CS is a Python script designed to process and analyze DICOM RTImages for the *Dose Rate - Collimator Speed* (DR-CS) quality control test for Varian TrueBeam radiotherapy systems. Predicted and acquired images are supported.

> [!WARNING]
> Licensed under the *Apache License, Version 2.0* (the "License");
> you may not use this software except in compliance with the License.
> You may obtain a copy of the License [here](http://www.apache.org/licenses/LICENSE-2.0).
>
> Unless agreed to in writing, software distributed under the License is
> distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
> ANY KIND, either express or implied. See the License for the specific
> language governing permissions and limitations under the License.
>
> Further to this, please note that this application has only undergone relatively rudimentary testing with a small sample of test images. It is strongly recommended that users perform comprehensive independent validation before use. 

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Command-Line Arguments](#command-line-arguments)
  - [Example Commands](#example-commands)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [Testing](#testing)
- [Continuous Integration](#continuous-integration)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Automated Analysis**: Processes DICOM images to calculate statistics for predefined Regions of Interest (ROIs).
- **Normalization**: Optionally normalizes DR-CS ROI statistics by corresponding open field images based on `AcquisitionDate`.
  - Open field images and DR-CS field images are differentiated by _RTImageLabel_. Users can edit `config.json` to configure supported open and DR-CS _RTImageLabel_ values.
- **Visualization**: Supports live display or saving of images with ROIs overlaid for visual inspection.
- **Flexible Configuration**: Uses a JSON configuration file to define ROI parameters and matching criteria.

## Installation

### Prerequisites

- Python 3.7 or higher
- Pip package manager

### Clone the Repository

```bash
git clone https://github.com/Matthew-Jennings/rad-qa
cd analyse-drcs
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Arguments

```bash
python analyse_drcs.py input_directory [options]
```

`input_directory`: Path to the directory containing DICOM files and `config.json`

#### Optional Arguments:

- `--inspect-live`: Display images with ROIs overlaid during processing.
- `--inspect-save`: Save images with ROIs overlaid to the `output_images` directory.
- `--normalize`: Normalize DR-CS ROI stats by the normalization field stats.
- `--open-csv`: Automatically open the output CSV file after processing.
- `--open-excel`: Automatically open the output Excel file after processing.


### Example Commands

- **Basic Analysis:**

```bash
python analyse_drcs.py path_to_input_directory
```

- **Analysis with Normalization:**

```bash
python analyse_drcs.py path_to_input_directory --normalize
```

- **Save Images with ROIs Overlaid:**

```bash
python analyse_drcs.py path_to_input_directory --inspect-save
```

- **Display Images with ROIs Overlaid:**

```bash
python analyse_drcs.py path_to_input_directory --inspect-live
```

- **Open Output Files After Processing:**

```bash
python analyse_drcs.py path_to_input_directory --open-csv --open-excel
```

## Configuration

The script requires a `config.json` file in the input directory to define ROI configurations and image matching criteria.

**Sample** `config.json`:

```json
{
  "roi_center_offset_from_image_centre_mm": 56,
  "roi_width_mm": 102,
  "roi_height_mm": 25,
  "roi_angles": [0, 90, 180, 270, 30],
  "roi_colors": ["red", "blue", "orange", "purple", "green"],
  "open_rtimage_labels": ["mv_0_1a", "open"],
  "drcs_rtimage_labels": ["mv_182_1a", "rad_qa_hdmlc", "ra2"]
}
```

### Configration Parameters

#### Shared Parameters:

  - `roi_center_offset_from_image_centre_mm`: Offset from the image center to the ROI center.
  - `roi_width_mm`: Width of the ROI.
  - `roi_height_mm`: Height of the ROI.

#### ROI-Specific Parameters:

- `roi_angles`: List of rotation angles for each ROI.
- `roi_colors`: List of colors for each ROI when overlaid on images.

#### Image Matching:

- `open_rtimage_labels`: List of `RTImageLabel`s identifying open field images.
- `drcs_rtimage_labels`: List of `RTImageLabel`s identifying DR-CS images.

**Note:** Ensure that `roi_angles` and `roi_colors` lists have the same length and ROI correspondence.


## Output Files

The script generates the following output files in the input directory:

- `roi_stats_{timestamp}.csv`: A CSV file containing the ROI statistics.
- `roi_stats_{timestamp}.xlsx`: An Excel file containing the ROI statistics.

### Columns in Output Files:

- `File`: Name of the DICOM file.
- `RTImageLabel`: The RTImageLabel from the DICOM metadata.
- `ImageType`: Type of the image (DRCS, OPEN, or NORMALIZED).
- `AcquisitionDate`: The date the image was acquired.
- `A`, `B`, `C`, `D`, `E`: Mean pixel values within each ROI.
- `Average`: Average of the ROI values.
- `Max vs Min`: Ratio of the maximum to minimum ROI values minus one.

**Note**: When normalization is performed, additional rows with `ImageType` set to `NORMALIZED` are added, representing the normalized ROI statistics.


## Testing

### Running Tests Locally

The project includes a test suite using pytest. To run the tests:

```bash
pytest test_analyse_drcs.py -v
```

### Test Data

The tests rely on sample DICOM files and a `config.json` file located in the `test/data` directory.
Ensure that the `config.json` file in `test/data` matches the expected configuration for the tests.

### Continuous Integration

This project uses GitHub Actions for continuous integration. The tests are automatically run on GitHub whenever changes are pushed to the repository or a pull request is opened.
