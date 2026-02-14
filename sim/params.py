"""
Satellite simulation parameters and configuration.

This file contains all configurable parameters for the satellite simulation,
including orbital elements, star tracker settings, and visualization options.
"""

# ==============================================================================
# SATELLITE ORBITAL PARAMETERS (TreeSat - Satellite of Interest)
# ==============================================================================

SATELLITE = {
    'altitude': 550,        # km above Earth's surface
    'eccentricity': 0.0001, # orbital eccentricity (0 = circular orbit)
    'inclination': 53.0,    # degrees (orbital inclination)
    'omega': 0.0,           # degrees (argument of periapsis)
    'Omega': 0.0,           # degrees (RAAN - right ascension of ascending node)
    'M': 0.0                # degrees (mean anomaly - starting position in orbit)
}

# ==============================================================================
# CONSTELLATION PARAMETERS
# ==============================================================================

CONSTELLATION = {
    'n_satellites': 10000,  # Total number of satellites in simulation
    'orbital_bands': [
        # Define orbital bands for constellation (altitude in km, inclination in degrees)
        {"altitude": 550, "inclination": 53.0},   # Starlink-like
        {"altitude": 600, "inclination": 97.6},   # Sun-synchronous orbit (SSO)
        {"altitude": 500, "inclination": 30.0},   # Low inclination
        {"altitude": 700, "inclination": 0.0},    # Equatorial
    ]
}

# ==============================================================================
# SIMULATION PARAMETERS
# ==============================================================================

SIMULATION = {
    'duration_hours': 100/60,  # Simulation duration in hours (default: 100 minutes)
    'dt_seconds': 5,           # Time step in seconds
    'backend': 'cpu',          # 'cpu' or 'gpu' (if CUDA available)
    'random_seed': 21          # Random seed for reproducibility
}

# ==============================================================================
# STAR TRACKER PARAMETERS
# ==============================================================================

STAR_TRACKER = {
    'fov_deg': 10.0,           # Field of view in degrees
    'image_size': 256,         # Image resolution (pixels, square)
    'observer_index': 0,       # Which satellite carries the star tracker (0 = TreeSat)
    'pointing_direction': 'T'  # 'T' = tangential (along-track), or specify [x, y, z] vector in ECI
}

# ==============================================================================
# VISUALIZATION PARAMETERS
# ==============================================================================

VISUALIZATION = {
    'max_satellites': None,     # Max satellites to show in plots (None = all)
    'earth_resolution': 'medium', # 'low', 'medium', 'high', 'ultra'
    'animation_fps': 30,        # Frames per second for animations
    'animation_max_frames': 200, # Maximum number of frames in animation
    'trail_length': 20,         # Number of timesteps to show in satellite trail
    'view_elevation': 20,       # Camera elevation angle in degrees
    'view_azimuth': 45,         # Camera azimuth angle in degrees
    'treesat_color': '#00FF00', # TreeSat marker color (bright green)
    'treesat_size': 8,          # TreeSat marker size
    'other_sat_color': 'white', # Other satellites color
    'other_sat_size': 5,        # Other satellites marker size
    'other_sat_alpha': 0.5,     # Other satellites transparency
}

# ==============================================================================
# PHYSICAL CONSTANTS
# ==============================================================================

CONSTANTS = {
    'earth_radius_km': 6378.137,    # Earth equatorial radius in km
    'earth_radius_m': 6378136.3,    # Earth equatorial radius in meters
    'mu': 3.986004418e14,           # Earth's gravitational parameter (m^3/s^2)
}

# ==============================================================================
# FILE PATHS
# ==============================================================================

import os

# Get the sim directory
SIM_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SIM_DIR)

PATHS = {
    'sim_dir': SIM_DIR,
    'project_root': PROJECT_ROOT,
    'data_dir': os.path.join(PROJECT_ROOT, 'sat', 'data'),
    'plots_dir': os.path.join(SIM_DIR, 'plots'),
    'earth_texture': os.path.join(SIM_DIR, 'assets', 'earth.jpeg'),
}
