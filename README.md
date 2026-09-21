# TreeSats 🛰️🌲
## Autonomous collision avoidance without a ground link

**TreeSats explores how satellites can detect, characterize, and avoid nearby objects using onboard sensing and compute.**

Its Starguard prototype connects synthetic orbital simulation, star-tracker imagery, computer vision, angles-only navigation, threat assessment, and maneuver optimization into an end-to-end autonomy pipeline designed for GPS- and communications-denied environments.

> **Prototype status:** This is a research and demonstration system, not flight-qualified collision-avoidance software. Its models, scenarios, and performance claims should be evaluated in that context.

---

## The Problem

As orbital traffic grows, centralized screening and ground-in-the-loop response become harder to scale—especially when navigation or communications are disrupted. A spacecraft that can reason locally about nearby objects could preserve valuable response time when the ground link is unavailable.

Satellites are no longer passive—they maneuver unpredictably, creating collision risks with no margin for error.

## The Solution
<p align="left">
  <img src="https://drive.google.com/uc?export=view&id=1lbUgbUyeI4kUeUHzc2t0AbJ29y34Fm_s" alt="Starguard System Diagram" width="250em"/>
</p>

TreeSats' **Starguard** prototype connects three autonomous capabilities:

1. **Collision Detection** - YOLOv8 + BoT-SORT identify satellites and debris in star tracker imagery, UKF estimates trajectories from angles-only measurements
   <p align="center">
     <img src="https://drive.google.com/uc?export=view&id=1xzDJ-i59cHXLY0aJ8GwoAdINu9SL0_rB" alt="Detection" width="40%"/>
     <img src="https://drive.google.com/thumbnail?id=1NW5rMCgtGMTp9HBoaMQg_mX1bjjVmRvC&sz=w1000" alt="Collision Avoidance" width="50%"/>
   </p>

2. **Collision Avoidance** - Convex optimization computes fuel-optimal evasive maneuvers for non-adversarial and adversarial satellites
<p align="center">
  <img src="https://drive.google.com/thumbnail?id=1DY_pNTBg9X7kO-Gb5fsW9xsS55nn07S-&sz=w1000" alt="Collision Avoidance" width="60%"/>
  <img src="https://drive.google.com/thumbnail?id=1s6pmjE2kwMOZi4Kas_SFDr-vOzMF5uRe&sz=w1000" alt="Threat Assessment" width="35%"/>
</p>

3. **Threat Assessment** - NVIDIA Cosmos VLM conducts a satellite threat assessment for satellites in close proximity orbits
<p align="center">
  <img src="https://drive.google.com/thumbnail?id=1VvNbrP6w-B9JxoqLWdP5gTR2MLfsx2OF&sz=w1000" alt="Demo 0" width="80%"/>
  <img src="https://drive.google.com/uc?export=view&id=1BDNoNxncAkacbsRdfgB7F8g53OMLWowd" alt="Demo 2" width="80%"/>
</p>

All processing happens onboard using existing star tracker cameras. No GPS. No ground link required.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic training data (orbital sim + star tracker images)
python main.py --mode train --duration 0.5 --visualize

# Auto-label the generated images
cd sat/computer_vision
python auto_label.py batch data/images/train/ data/labels/train/

# Train YOLOv8 satellite detector
python train.py --epochs 150 --batch 64

# Run full pipeline (simulate → detect → track → avoid)
cd ../..
python main.py --mode pipeline
```

---

## Key Features

- **Communication-Independent**: Operates without GPS or ground contact using only star tracker imagery
- **Autonomous**: Real-time collision detection, avoidance, and threat assessment without ground intervention
- **Scalable simulation**: GPU-accelerated propagation supports 10,000+ simulated satellites
- **Mission-Aware**: Fuel-optimal maneuvers via convex optimization

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Orbital Propagation | Tensorgator (GPU) | High-fidelity Keplerian mechanics for 10k+ satellites |
| Object Detection | YOLOv8 | Detect satellites/debris in star tracker images |
| Object Tracking | BoT-SORT | Persistent IDs across frames for trajectory extraction |
| State Estimation | Unscented Kalman Filter | Fuse measurements with orbital dynamics |
| Maneuver Planning | CVXPY | Convex optimization for collision avoidance |
| Classification | NVIDIA Cosmos VLM | Identify satellite types and assess threats |

---

## Repository Structure

```
treesats/
├── main.py                    # Entry point (train/pipeline modes)
├── sim/                       # GPU orbital simulation
│   ├── simulate.py           # Tensorgator propagation
│   └── star_tracker/         # Camera rendering
├── sat/
│   ├── computer_vision/      # YOLOv8 detection + tracking
│   │   ├── train.py
│   │   ├── auto_label.py
│   │   └── TRAINING.md       # Complete training guide
│   └── control/              # Collision avoidance + GNC
├── treesats_proximity/        # VLM threat assessment
└── flyby_data/               # Test imagery (Cassini, TIE, etc.)
```

---

## Documentation

- **[Training Guide](sat/computer_vision/TRAINING.md)** - YOLOv8 training for satellite detection
- **[Auto-Labeling Guide](sat/computer_vision/AUTO_LABELING.md)** - Automated dataset generation
- **[Presentation Slides](slides/treesats.pdf)** - Full system overview

---

<br>

# Detailed Technical Overview

## How Starguard Works

### 1. Detection Pipeline

**Star trackers** are attitude sensors present on virtually all spacecraft. TreeSats repurposes them for threat detection:

```
Star Tracker Image → YOLOv8 Detection → BoT-SORT Tracking → Bearing Angles
```

- Filters out background stars (known catalog)
- Detects moving objects as bright spots
- Assigns persistent track IDs
- Extracts pixel coordinates → elevation/azimuth angles

### 2. Angles-Only Navigation

From sequential star tracker images, TreeSats reconstructs full 6-DOF state:

| Images | Information Extracted |
|--------|---------------------|
| 1 photo | Bearing angles (el, az) |
| 2 photos | Bearing + depth → **Position vector** |
| 3 photos | Finite difference → **Position + Velocity** |

**Output**: State vector `x = [x, y, z, ẋ, ẏ, ż]` in ECI frame

### 3. State Estimation (UKF)

An **Unscented Kalman Filter** fuses two information sources:

- **Dynamics Model**: Keplerian orbital propagation (`M = E - e sin E`)
- **Measurements**: Position/velocity from star tracker angles

This provides filtered, noise-resistant state estimates for collision prediction.

### 4. Collision Detection

Forward-propagate filtered trajectories to compute:

- **Closest approach distance** (miss distance)
- **Time to closest approach**
- **Probability of collision** (from covariance)

If risk exceeds threshold → trigger autonomous maneuver.

### 5. Evasive Maneuvers

**Convex optimization** (CVXPY) solves for minimum ΔV subject to:

- Collision avoidance constraint (minimum safe distance)
- Fuel budget limit
- Thrust magnitude/direction limits
- Mission constraints (maintain orbit parameters)

**Output**: Time-optimal thrust profile for safe evasion.

---

## GPU Orbital Simulation

TreeSats uses **Tensorgator** (GPU-accelerated Keplerian propagator) to simulate 10,000+ satellites:

```python
# Solve Kepler's equation: M = E - e sin E
# Distribute across 10k satellites on CUDA cores
positions, velocities = propagate_constellation(
    n_sats=10000,
    duration_hours=24,
    dt_seconds=10
)
```

**Performance**: 10,000 satellites × 8,640 timesteps (24h @ 10s) in seconds on GPU vs. hours on CPU.

---

## Star Tracker Rendering

**Pinhole camera model** generates synthetic 256×256 images:

1. Project satellite positions into camera frame (RTN → body frame)
2. Apply perspective projection with field-of-view
3. Render as bright dots on black background
4. Save as training images

**Auto-labeling**: Brightness thresholding automatically generates YOLO bounding boxes from ground truth positions—no manual annotation required.

---

## Computer Vision Pipeline

### Training Workflow

```bash
# 1. Generate synthetic data
python main.py --mode train --duration 0.5 --visualize

# 2. Auto-label (brightness thresholding)
cd sat/computer_vision
python auto_label.py batch data/images/train/ data/labels/train/

# 3. Validate dataset
python prepare_data.py validate

# 4. Train YOLOv8
python train.py --epochs 150 --batch 64 --model yolov8m
```

### Inference

```python
from sat.computer_vision import detect_single, track

# Single image detection
result = detect_single("star_tracker_image.jpg")
print(f"Detected {result['count']} objects")

# Video tracking
results = track("star_tracker_sequence.mp4", tracker="botsort.yaml")
```

**Output**: Bounding boxes + persistent track IDs + pixel coordinates

---

## VLM Threat Assessment

**NVIDIA Cosmos Reason** (served via **vLLM**) classifies satellites from imagery:

```bash
cd treesats_proximity
python assessment.py flyby_data/cassini/cassini.mp4
```

**Identifies**:
- Satellite type (Cassini spacecraft, communications sat, etc.)
- Object class (active satellite vs. debris vs. fairing)
- Threat level based on maneuver capability

**vLLM** provides high-throughput, low-latency inference for the vision-language model, enabling real-time proximity threat assessment onboard satellites.

Supports images, image sequences, and video (mp4/avi/mov).

---

## Demonstration Scenarios

### Chaser-Evader
- **Chaser** actively pursues target
- **Evader** (your satellite) detects and autonomously evades
- Real-time trajectory prediction enables proactive response

### Multi-Object Tracking
- Simultaneous tracking of multiple threats
- Prioritized risk scoring (closest approach, time-to-collision)
- Coordinated maneuvers to avoid multiple objects

### Collision Avoidance
- Forward propagate trajectories
- Predict collisions before they occur
- Compute fuel-optimal avoidance maneuver

---

## Applications

### Commercial Space
- Starlink, OneWeb, and mega-constellations
- Earth observation satellites
- Communications satellites

### National Security
- Reconnaissance satellites in communication-denied regions
- Resilient operations in contested environments
- Autonomous collision detection, avoidance, and threat assessment

### Space Situational Awareness
- Orbital debris tracking
- Traffic management for crowded orbits
- Collision warning systems

---

## Team

- **Sid Anantha** - SpaceX Starship GNC
- **Rahul Ayanampudi** - Northrop Grumman GNC/Physical AI
- **Lundeen Cahilly** - Stanford CS, Incoming GNC @ K2 Space
- **Raj Thapliyal** - Defence Science Org. - Satellite Technology & Research / NUS ML
---

## Roadmap

- [ ] Embedded deployment (spacecraft flight computer)
- [ ] Multi-sensor fusion (star tracker + IMU + magnetometer)
- [ ] Cooperative avoidance (inter-satellite coordination)
- [ ] Hardware-in-the-loop testing
- [ ] Flight software integration

---

<p align="center">
  <strong>TreeSats: Autonomous orbital defense for the modern space domain</strong>
</p>
