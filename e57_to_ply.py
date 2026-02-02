#!/usr/bin/env python3
"""
E57 to PLY Converter
Extracts point cloud data from E57 files to PLY format.
Optimized for Gaussian Splatting workflows.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np

try:
    import pye57
except ImportError:
    print("Error: pye57 not installed. Run: pip install pye57")
    sys.exit(1)

try:
    from plyfile import PlyData, PlyElement
except ImportError:
    print("Error: plyfile not installed. Run: pip install plyfile")
    sys.exit(1)


def read_e57_file(filepath):
    """Read E57 file and return E57 object."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"E57 file not found: {filepath}")
    return pye57.E57(filepath)


def extract_scan_data(e57, scan_index=0):
    """
    Extract point cloud data from a specific scan.
    Returns dict with: x, y, z, r, g, b, intensity (all numpy arrays)
    """
    print(f"  Extracting scan {scan_index}...")
    
    # Read scan raw data
    data = e57.read_scan(scan_index)
    
    # Extract coordinates (always present)
    x = data.get("cartesianX", np.array([]))
    y = data.get("cartesianY", np.array([]))
    z = data.get("cartesianZ", np.array([]))
    
    if len(x) == 0 or len(y) == 0 or len(z) == 0:
        raise ValueError(f"Scan {scan_index} has no valid Cartesian coordinates")
    
    # Extract color (may not be present)
    r = data.get("colorRed", np.array([]))
    g = data.get("colorGreen", np.array([]))
    b = data.get("colorBlue", np.array([]))
    
    # If no RGB, try to use intensity for grayscale
    intensity = data.get("intensity", np.array([]))
    
    if len(r) == 0 and len(intensity) > 0:
        # Convert intensity to grayscale RGB
        int_min, int_max = intensity.min(), intensity.max()
        if int_max > int_min:
            norm_int = ((intensity - int_min) / (int_max - int_min) * 255).astype(np.uint8)
        else:
            norm_int = (intensity * 255).astype(np.uint8)
        r = g = b = norm_int
    elif len(r) == 0:
        # No color or intensity - set to white
        r = g = b = np.full_like(x, 255, dtype=np.uint8)
    else:
        # Ensure uint8 format for colors
        r = r.astype(np.uint8)
        g = g.astype(np.uint8)
        b = b.astype(np.uint8)
    
    # Handle intensity if present
    if len(intensity) == 0:
        intensity = np.zeros_like(x, dtype=np.float32)
    else:
        intensity = intensity.astype(np.float32)
    
    point_count = len(x)
    print(f"    Points: {point_count:,}")
    print(f"    Has RGB: {len(data.get('colorRed', [])) > 0}")
    print(f"    Has Intensity: {len(data.get('intensity', [])) > 0}")
    
    return {
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "z": z.astype(np.float32),
        "r": r,
        "g": g,
        "b": b,
        "intensity": intensity,
        "point_count": point_count
    }


def downsample_data(data, target_points=None):
    """
    Downsample point cloud if it exceeds target_points.
    """
    if target_points is None or data["point_count"] <= target_points:
        return data
    
    # Random sampling
    indices = np.random.choice(data["point_count"], target_points, replace=False)
    indices = np.sort(indices)
    
    print(f"  Downsampling from {data['point_count']:,} to {target_points:,} points")
    
    return {
        "x": data["x"][indices],
        "y": data["y"][indices],
        "z": data["z"][indices],
        "r": data["r"][indices],
        "g": data["g"][indices],
        "b": data["b"][indices],
        "intensity": data["intensity"][indices],
        "point_count": target_points
    }


def write_ply(data, output_path, binary=True):
    """
    Write point cloud data to PLY file.
    """
    vertex_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('intensity', 'f4')
    ]
    
    vertices = np.zeros(data["point_count"], dtype=vertex_dtype)
    vertices['x'] = data["x"]
    vertices['y'] = data["y"]
    vertices['z'] = data["z"]
    vertices['red'] = data["r"]
    vertices['green'] = data["g"]
    vertices['blue'] = data["b"]
    vertices['intensity'] = data["intensity"]
    
    el = PlyElement.describe(vertices, 'vertex')
    ply_data = PlyData([el], text=not binary)
    ply_data.write(output_path)
    
    print(f"  Written: {output_path} ({data['point_count']:,} points)")


def convert_single_scan(e57_path, scan_index, output_path, downsample=None):
    """Convert a single scan from E57 to PLY."""
    print(f"\nConverting scan {scan_index} from: {e57_path}")
    
    e57 = read_e57_file(e57_path)
    data = extract_scan_data(e57, scan_index)
    
    if downsample:
        data = downsample_data(data, downsample)
    
    write_ply(data, output_path, binary=True)
    return output_path


def convert_all_scans(e57_path, output_dir, downsample_per_scan=None):
    """Convert all scans from E57 file to separate PLY files."""
    print(f"\nConverting all scans from: {e57_path}")
    
    e57 = read_e57_file(e57_path)
    scan_count = e57.scan_count
    print(f"Found {scan_count} scan(s)")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_files = []
    for i in range(scan_count):
        output_path = output_dir / f"scan_{i:03d}.ply"
        try:
            convert_single_scan(e57_path, i, str(output_path), downsample_per_scan)
            output_files.append(str(output_path))
        except Exception as e:
            print(f"  Error converting scan {i}: {e}")
    
    print(f"\nConverted {len(output_files)}/{scan_count} scans to: {output_dir}")
    return output_files


def merge_scans_to_single_ply(e57_path, output_path, downsample_total=None):
    """
    Merge all scans from E57 file into a single PLY file.
    """
    print(f"\nMerging all scans to single PLY: {output_path}")
    
    e57 = read_e57_file(e57_path)
    scan_count = e57.scan_count
    print(f"Found {scan_count} scan(s)")
    
    all_data = []
    total_points = 0
    
    for i in range(scan_count):
        try:
            data = extract_scan_data(e57, i)
            all_data.append(data)
            total_points += data["point_count"]
        except Exception as e:
            print(f"  Error reading scan {i}: {e}")
    
    print(f"Total points across all scans: {total_points:,}")
    
    if downsample_total and total_points > downsample_total:
        ratio = downsample_total / total_points
        print(f"Downsampling ratio: {ratio:.3f}")
        
        for data in all_data:
            target = int(data["point_count"] * ratio)
            data.update(downsample_data(data, target))
    
    # Merge all data
    merged = {
        "x": np.concatenate([d["x"] for d in all_data]),
        "y": np.concatenate([d["y"] for d in all_data]),
        "z": np.concatenate([d["z"] for d in all_data]),
        "r": np.concatenate([d["r"] for d in all_data]),
        "g": np.concatenate([d["g"] for d in all_data]),
        "b": np.concatenate([d["b"] for d in all_data]),
        "intensity": np.concatenate([d["intensity"] for d in all_data]),
        "point_count": sum(d["point_count"] for d in all_data)
    }
    
    write_ply(merged, output_path, binary=True)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert E57 point cloud files to PLY format"
    )
    parser.add_argument("input", help="Input E57 file path")
    parser.add_argument("-o", "--output", help="Output PLY file path (for single scan)")
    parser.add_argument("-d", "--output-dir", help="Output directory (for multiple scans)")
    parser.add_argument("-s", "--scan", type=int, default=0, help="Scan index to convert (default: 0)")
    parser.add_argument("--all-scans", action="store_true", help="Convert all scans")
    parser.add_argument("--merge", action="store_true", help="Merge all scans to single PLY")
    parser.add_argument("--downsample", type=int, help="Downsample to N points per scan")
    parser.add_argument("--downsample-total", type=int, help="Downsample total merged output to N points")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    if args.merge:
        output = args.output or args.input.replace(".e57", "_merged.ply")
        merge_scans_to_single_ply(args.input, output, args.downsample_total)
    elif args.all_scans:
        output_dir = args.output_dir or args.input.replace(".e57", "_scans")
        convert_all_scans(args.input, output_dir, args.downsample)
    else:
        output = args.output or args.input.replace(".e57", f"_scan{args.scan}.ply")
        convert_single_scan(args.input, args.scan, output, args.downsample)
    
    print("\nConversion complete!")


if __name__ == "__main__":
    main()
