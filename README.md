# treesats

GPU-accelerated satellite constellation simulation using Tensorgator.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the simulation:**
   ```bash
   python simulate.py
   ```

This will simulate 10,000 satellites across 4 orbital bands over 24 hours using GPU acceleration.

## Overview

Uses [Tensorgator](https://github.com/ApoPeri/TensorGator) to efficiently simulate large satellite constellations on GPU. Currently simulates 10k satellites across standard orbital bands:
- 550 km altitude, 53° inclination (Starlink-like)
- 600 km altitude, 97.6° inclination (Sun-Synchronous)
- 500 km altitude, 30° inclination (Low inclination)
- 700 km altitude, 0° inclination (Equatorial)

## Requirements

- NVIDIA GPU with CUDA support
- Python 3.8+
- Tensorgator, NumPy, Matplotlib

## Future Work

- Star tracker data generation (projecting satellite positions as dots in camera frame)
- Visualization tools
- Custom orbital configurations
