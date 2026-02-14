"""
Core satellite simulation library using Tensorgator.

Provides functions for creating satellite constellations and running
GPU-accelerated orbital propagation simulations.
"""

import numpy as np
import tensorgator as tg
import time
from .params import SATELLITE, CONSTELLATION, SIMULATION, CONSTANTS

# Check if CUDA is available
CUDA_AVAILABLE = False
try:
    from numba import cuda
    if cuda.is_available():
        CUDA_AVAILABLE = True
        print("CUDA GPU detected - will use GPU acceleration")
    else:
        print("CUDA GPU not available - will use CPU backend")
except Exception as e:
    print(f"CUDA not available ({e}) - will use CPU backend")

def create_constellation(n_sats=10000, sat=None):
    """
    Create a constellation of satellites randomly distributed across orbital shells.

    Parameters:
    -----------
    n_sats : int
        Total number of satellites to simulate (default: 10000)
    sat : dict, optional
        Orbital elements for the satellite of interest (will be at index 0).
        If provided, satellite 0 will use these specific orbital parameters.
        Dictionary should contain:
            'altitude': altitude in km above Earth's surface
            'eccentricity': orbital eccentricity (0 for circular)
            'inclination': orbital inclination in degrees
            'omega': argument of periapsis in degrees
            'Omega': RAAN (right ascension of ascending node) in degrees
            'M': mean anomaly in degrees

    Returns:
    --------
    constellation : ndarray
        Array of Keplerian orbital elements [a, e, i, omega, Omega, M] for tensorgator
        Satellite of interest is at index 0, remaining satellites randomly distributed.
    """

    # Get orbital shells configuration from params
    shells = CONSTELLATION['orbital_shells']

    # Earth radius in meters (tensorgator expects meters!)
    earth_radius = CONSTANTS['earth_radius_m']

    # Initialize arrays for Keplerian elements
    a_list = []  # Semi-major axis
    e_list = []  # Eccentricity
    i_list = []  # Inclination
    omega_list = []  # Argument of periapsis
    Omega_list = []  # RAAN (Right Ascension of Ascending Node)
    M_list = []  # Mean anomaly

    # Add the satellite of interest first (index 0)
    sat_index = 0
    if sat is not None:
        # Add satellite of interest with specified orbital elements
        # Convert altitude from km to meters
        a_list.append(earth_radius + sat['altitude'] * 1000)
        e_list.append(sat.get('eccentricity', 0.0))
        i_list.append(np.radians(sat['inclination']))
        omega_list.append(np.radians(sat.get('omega', 0.0)))
        Omega_list.append(np.radians(sat.get('Omega', 0.0)))
        M_list.append(np.radians(sat.get('M', 0.0)))
        sat_index = 1
        print(f"Satellite of interest (index 0) configured:")
        print(f"  Altitude: {sat['altitude']} km")
        print(f"  Inclination: {sat['inclination']}°")
        print(f"  Eccentricity: {sat.get('eccentricity', 0.0)}")

    # Randomly distribute remaining satellites across shells
    remaining_sats = n_sats - sat_index

    print(f"\nRandomly distributing {remaining_sats} satellites across {len(shells)} orbital shells...")

    # Randomly assign each satellite to a shell
    shell_assignments = np.random.choice(len(shells), size=remaining_sats)

    # Count and print satellites per shell
    for i, shell in enumerate(shells):
        count = np.sum(shell_assignments == i)
        if count > 0:
            print(f"  {shell['name']:20s}: {count:5d} sats (~{100*count/remaining_sats:4.1f}%) | "
                  f"Alt: {shell['altitude']}±{shell['altitude_var']}km, Inc: {shell['inclination']}±{shell['inc_var']}°")

    # Create satellites with shell-specific parameters
    for shell_idx in shell_assignments:
        shell = shells[shell_idx]

        # Altitude: shell altitude ± variation
        altitude_km = np.random.normal(shell["altitude"], shell["altitude_var"])
        altitude_km = np.clip(altitude_km, 400, 800)  # Keep within LEO range
        a_list.append(earth_radius + altitude_km * 1000)

        # Eccentricity: 0.0-0.005 (very circular)
        e_list.append(np.random.uniform(0.0, 0.005))

        # Inclination: shell inclination ± variation
        inc_variation = np.random.normal(0, shell["inc_var"])
        inclination = shell["inclination"] + inc_variation
        inclination = np.clip(inclination, 0, 180)
        i_list.append(np.radians(inclination))

        # Argument of periapsis: random (doesn't matter much for circular orbits)
        omega_list.append(np.random.uniform(0, 2 * np.pi))

        # RAAN: uniformly distributed for coverage
        Omega_list.append(np.random.uniform(0, 2 * np.pi))

        # Mean anomaly: uniformly distributed for spread
        M_list.append(np.random.uniform(0, 2 * np.pi))

    # Create constellation array for tensorgator
    # Format: [a, e, i, omega, Omega, M] for each satellite
    constellation = np.column_stack([
        np.array(a_list),
        np.array(e_list),
        np.array(i_list),
        np.array(omega_list),
        np.array(Omega_list),
        np.array(M_list)
    ])

    return constellation


def run_simulation(n_sats=10000, duration_hours=24, dt_seconds=60, sat=None):
    """
    Run GPU-accelerated satellite simulation.

    Parameters:
    -----------
    n_sats : int
        Number of satellites to simulate
    duration_hours : float
        Simulation duration in hours
    dt_seconds : float
        Time step in seconds
    sat : dict, optional
        Orbital elements for satellite of interest (see create_constellation for format)

    Returns:
    --------
    positions : ndarray
        Satellite positions in ECI frame, shape (timesteps, satellites, 3) in meters
    time_array : ndarray
        Time array in seconds
    constellation : ndarray
        Keplerian orbital elements for all satellites
    """

    np.random.seed(SIMULATION['random_seed'])

    print(f"Setting up constellation with {n_sats} satellites...")
    constellation = create_constellation(n_sats, sat=sat)

    # Create time array - TensorGator expects times in seconds since reference epoch
    # Using np.arange to create array from 0 to duration with dt_seconds step size
    time_array = np.arange(0, duration_hours * 3600, dt_seconds)  # Convert hours to seconds
    n_timesteps = len(time_array)

    # Select backend based on CUDA availability
    backend = SIMULATION['backend']

    print(f"Running simulation for {n_timesteps} timesteps over {duration_hours} hours (dt={dt_seconds}s)...")
    print(f"Using backend: {backend.upper()}")

    start_time = time.time()

    # Run simulation
    positions = tg.satellite_positions(
        time_array,
        constellation,
        backend=backend,
        return_frame='eci',  # Earth-Centered Inertial frame
        input_type='kepler'
    )

    elapsed = time.time() - start_time

    # Tensorgator returns shape (satellites, timesteps, xyz), we want (timesteps, satellites, xyz)
    positions = np.transpose(positions, (1, 0, 2))

    print(f"\n✓ Simulation complete!")
    print(f"  Time elapsed: {elapsed:.3f} seconds")
    print(f"  Satellites: {n_sats}")
    print(f"  Timesteps: {n_timesteps}")
    print(f"  Total positions computed: {n_sats * n_timesteps:,}")
    print(f"  Performance: {(n_sats * n_timesteps) / elapsed:,.0f} positions/second")

    print(f"\nPosition array shape: {positions.shape}")
    print(f"Position array shape format: (timesteps, satellites, xyz)")

    return positions, time_array, constellation
