# E57 File Analyzer for Gaussian Splatting

This C++ tool analyzes `.e57` point cloud files to assess their suitability for Gaussian Splatting reconstruction. It provides a detailed breakdown of each scan, including point count, available data fields (Cartesian coordinates, color, intensity), pose information, and an overall readiness score.

## Features

-   **Detailed Scan Analysis**: Reports on point count, coordinate systems, color data, and more for each scan.
-   **Gaussian Splatting Readiness**: Assesses if a scan has the necessary data (3D coordinates and color/intensity).
-   **Memory Estimation**: Calculates the estimated memory footprint for training a Gaussian Splatting model.
-   **Workflow Recommendations**: Suggests a path for converting the E57 data to a format usable by GS tools.
-   **File Validation**: Checks if the input file exists before processing.

## Installation

This tool requires the E57 Format library (`libe57format-dev`). You can install it on Debian-based systems (like Ubuntu) using `apt`.

```bash
sudo apt update
sudo apt install -y libe57format-dev
```

This will install the necessary headers and shared libraries. The script also uses `g++` for compilation, which is part of the `build-essential` package.

```bash
sudo apt install -y build-essential
```

## Compilation

To compile the `analyze.cpp` script, you need to link against the E57 Format and Xerces-C libraries. The following command will create an executable named `analyze`:

```bash
g++ -std=c++11 -o analyze analyze.cpp -lE57Format -lxerces-c
```

*Note: If the E57 Format library was installed to a custom location, you may need to specify the include (`-I`) and library (`-L`) paths.*

For example, if it was installed in `/usr/local/E57Format-3.3-x86_64-gcc11/`:
```bash
g++ -std=c++11 -I/usr/local/E57Format-3.3-x86_64-gcc11/include -L/usr/local/E57Format-3.3-x86_64-gcc11/lib -o analyze analyze.cpp -lE57Format -lxerces-c
```

## Usage

Run the compiled executable from your terminal, passing the path to your `.e57` file as an argument.

```bash
./analyze "path/to/your/file.e57"
```

**Example:**
```bash
./analyze "BLK360 Outdoor Dataset.e57"
```

## Example Output

The script will produce a detailed report for each scan and a final summary.

```
╔══════════════════════════════════════════════════════════════╗
║         E57 FILE ANALYSIS FOR GAUSSIAN SPLATTING             ║
╚══════════════════════════════════════════════════════════════╝

FILE INFORMATION:
  File: BLK360 Outdoor Dataset.e57
  Format: ASTM E57 3D Imaging Data File
  Version: 1.0
  GUID: {C99B3C72-45D9-48D8-8314-6B40D76DAA4D}
  Total Scans: 23
  2D Images: 138
────────────────────────────────────────────────────────────────


╔══════════════════════════════════════════════════════════════╗
║ SCAN #0                                                      ║
╚══════════════════════════════════════════════════════════════╝

BASIC INFO:
  Name: 1 09
  Point Count: 3149514
  Est. Memory (GS): 450.5 MB

COORDINATE SYSTEM:
  ✓ Cartesian (X, Y, Z)
    - Has invalid state field

...
```
