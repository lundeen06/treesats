# treesats

GPU-accelerated satellite constellation simulation and onboard algorithms.

## Structure

```
treesats/
├── sim/              # Physics simulation (Tensorgator-based)
│   └── simulate.py   # Constellation propagation
├── sat/              # Satellite onboard systems
│   ├── cv/           # Computer vision algorithms
│   ├── ctrl/         # Attitude control
│   └── sensors/      # Sensor models
└── tests/            # Test suite
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the simulation:**
   ```bash
   python sim/simulate.py
   ```

This will simulate 10,000 satellites across 4 orbital bands over 24 hours using GPU acceleration.

## Simulation (`sim/`)

Uses [Tensorgator](https://github.com/ApoPeri/TensorGator) to efficiently simulate large satellite constellations on GPU. Currently simulates 10k satellites across standard orbital bands:
- 550 km altitude, 53° inclination (Starlink-like)
- 600 km altitude, 97.6° inclination (Sun-Synchronous)
- 500 km altitude, 30° inclination (Low inclination)
- 700 km altitude, 0° inclination (Equatorial)

## Satellite Onboard (`sat/`)

- **`cv/`** - Computer vision algorithms for star tracking and pose estimation
- **`ctrl/`** - Attitude control and guidance algorithms
- **`sensors/`** - Sensor models (star tracker, IMU, etc.)

## Requirements

- NVIDIA GPU with CUDA support
- Python 3.8+
- Tensorgator, NumPy, Matplotlib

## Future Work

- Star tracker image generation (render satellites as dots in camera frame)
- Pose estimation from star tracker data
- Relative navigation algorithms
- Attitude control implementation
