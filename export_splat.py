#!/usr/bin/env python3
"""
Export Gaussian Splatting PLY to compressed .splat or .spz format.
.splat is a simple binary format for WebGL viewers.
.spz is Niantic's compressed format (smaller file size).
"""

import argparse
import os
import sys
import struct
from pathlib import Path
from typing import Optional

import numpy as np
from plyfile import PlyData


def load_gaussian_ply(ply_path: str) -> dict:
    """
    Load Gaussian Splatting PLY file.
    Returns dict with all Gaussian parameters.
    """
    print(f"Loading Gaussian PLY: {ply_path}")
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    
    num_points = len(vertex['x'])
    
    # Extract data
    data = {
        'positions': np.stack([
            np.array(vertex['x']),
            np.array(vertex['y']),
            np.array(vertex['z'])
        ], axis=-1).astype(np.float32),
        'scales': np.stack([
            np.array(vertex['scale_0']),
            np.array(vertex['scale_1']),
            np.array(vertex['scale_2'])
        ], axis=-1).astype(np.float32) if 'scale_0' in vertex else np.ones((num_points, 3), dtype=np.float32) * 0.01,
        'rotations': np.stack([
            np.array(vertex['rot_0']),
            np.array(vertex['rot_1']),
            np.array(vertex['rot_2']),
            np.array(vertex['rot_3'])
        ], axis=-1).astype(np.float32) if 'rot_0' in vertex else np.array([[1, 0, 0, 0]] * num_points, dtype=np.float32),
        'opacities': np.array(vertex['opacity']).astype(np.float32) if 'opacity' in vertex else np.ones(num_points, dtype=np.float32) * 0.5,
        'colors': np.stack([
            np.array(vertex['f_dc_0']),
            np.array(vertex['f_dc_1']),
            np.array(vertex['f_dc_2'])
        ], axis=-1).astype(np.float32) if 'f_dc_0' in vertex else np.zeros((num_points, 3), dtype=np.float32),
    }
    
    print(f"  Loaded {num_points:,} Gaussians")
    return data


def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def sigmoid_inverse(x):
    """Inverse sigmoid (logit)."""
    return np.log(x / (1 - x))


def pack_gaussians_to_splat(data: dict, output_path: str):
    """
    Pack Gaussian parameters into .splat binary format.
    Format per Gaussian (32 bytes):
    - Position (3x float32) = 12 bytes
    - Scale (3x float32) = 12 bytes
    - Color (4x uint8, packed) = 4 bytes (RGB + alpha)
    - Rotation (4x uint8, normalized) = 4 bytes (quaternion)
    """
    print(f"Packing to .splat: {output_path}")
    
    positions = data['positions']
    scales = data['scales']
    rotations = data['rotations']
    opacities = sigmoid(data['opacities'])  # Apply sigmoid
    colors = data['colors']
    
    num_points = len(positions)
    
    # Normalize colors (SH DC component to RGB)
    # DC component needs to be transformed: C = 0.5 + 0.28209 * SH_DC
    rgb = 0.5 + 0.28209 * colors
    rgb = np.clip(rgb, 0, 1)
    
    # Pack to uint8
    rgba = np.zeros((num_points, 4), dtype=np.uint8)
    rgba[:, :3] = (rgb * 255).astype(np.uint8)
    rgba[:, 3] = (opacities * 255).astype(np.uint8)
    
    # Normalize quaternions and pack to uint8
    norm_rot = rotations / (np.linalg.norm(rotations, axis=1, keepdims=True) + 1e-8)
    # Convert from [-1, 1] to [0, 255]
    rot_packed = ((norm_rot * 127.5) + 127.5).astype(np.uint8)
    
    # Write binary file
    with open(output_path, 'wb') as f:
        for i in range(num_points):
            # Position (3 floats)
            f.write(struct.pack('3f', *positions[i]))
            # Scale (3 floats) - convert to log scale
            log_scales = np.log(np.maximum(scales[i], 1e-6))
            f.write(struct.pack('3f', *log_scales))
            # Color (4 bytes: RGBA)
            f.write(struct.pack('4B', *rgba[i]))
            # Rotation (4 bytes: quaternion)
            f.write(struct.pack('4B', *rot_packed[i]))
    
    file_size = os.path.getsize(output_path)
    print(f"  Written {num_points:,} Gaussians ({file_size / 1024 / 1024:.2f} MB)")


def quantize_floats(x: np.ndarray, bits: int = 8) -> tuple:
    """
    Quantize float array to integers.
    Returns: (quantized array, min_val, max_val)
    """
    min_val = x.min()
    max_val = x.max()
    
    if max_val - min_val < 1e-8:
        return np.zeros_like(x, dtype=np.uint8), min_val, max_val
    
    normalized = (x - min_val) / (max_val - min_val)
    quantized = (normalized * ((1 << bits) - 1)).astype(np.uint8)
    
    return quantized, min_val, max_val


def pack_gaussians_to_spz(data: dict, output_path: str):
    """
    Pack Gaussian parameters into .spz compressed format (Niantic format).
    This is a simplified version - full SPZ uses more advanced compression.
    """
    print(f"Packing to .spz: {output_path}")
    print("  Note: This is a simplified SPZ-like format")
    
    positions = data['positions']
    scales = data['scales']
    rotations = data['rotations']
    opacities = sigmoid(data['opacities'])
    colors = data['colors']
    
    num_points = len(positions)
    
    # Transform colors
    rgb = 0.5 + 0.28209 * colors
    rgb = np.clip(rgb, 0, 1)
    
    # Quantize to 8-bit
    pos_q, pos_min, pos_max = quantize_floats(positions.flatten(), 8)
    scale_q, scale_min, scale_max = quantize_floats(np.log(scales.flatten() + 1e-6), 8)
    
    # Pack rotations (normalized to 8-bit)
    norm_rot = rotations / (np.linalg.norm(rotations, axis=1, keepdims=True) + 1e-8)
    rot_q = ((norm_rot + 1) * 127.5).astype(np.uint8)
    
    # Write header + data
    with open(output_path, 'wb') as f:
        # Magic number
        f.write(b'SPZ\x00')
        # Version
        f.write(struct.pack('I', 1))
        # Number of points
        f.write(struct.pack('I', num_points))
        # Bounds for positions
        f.write(struct.pack('6f', pos_min, pos_max, 0, 0, 0, 0))
        # Bounds for scales  
        f.write(struct.pack('6f', scale_min, scale_max, 0, 0, 0, 0))
        
        # Write quantized data
        f.write(pos_q.tobytes())
        f.write(scale_q.tobytes())
        f.write(rot_q.tobytes())
        f.write((opacities * 255).astype(np.uint8).tobytes())
        f.write((rgb * 255).astype(np.uint8).tobytes())
    
    file_size = os.path.getsize(output_path)
    print(f"  Written {num_points:,} Gaussians ({file_size / 1024 / 1024:.2f} MB)")


def convert_ply_to_format(ply_path: str, output_path: str, format_type: str = 'splat'):
    """
    Convert Gaussian PLY to specified format.
    """
    data = load_gaussian_ply(ply_path)
    
    if format_type.lower() == 'splat':
        if not output_path.endswith('.splat'):
            output_path += '.splat'
        pack_gaussians_to_splat(data, output_path)
    elif format_type.lower() == 'spz':
        if not output_path.endswith('.spz'):
            output_path += '.spz'
        pack_gaussians_to_spz(data, output_path)
    else:
        raise ValueError(f"Unknown format: {format_type}. Use 'splat' or 'spz'.")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Export Gaussian Splatting PLY to compressed formats"
    )
    parser.add_argument("input", help="Input Gaussian PLY file path")
    parser.add_argument("-o", "--output", help="Output file path (auto-detects format from extension)")
    parser.add_argument("-f", "--format", choices=['splat', 'spz'], default='splat',
                        help="Output format (default: splat)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Determine output format from extension or argument
    output = args.output or args.input.replace('.ply', f'.{args.format}')
    
    if output.endswith('.spz'):
        fmt = 'spz'
    else:
        fmt = 'splat'
    
    # Convert
    result = convert_ply_to_format(args.input, output, fmt)
    
    print(f"\nConversion complete: {result}")


if __name__ == "__main__":
    main()
