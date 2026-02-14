"""
Simple example of running 10k+ satellite simulations in parallel on GPU using Tensorgator.
"""

import numpy as np
import tensorgator as tg
import time
import sys

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

def create_constellation(n_sats=10000):
    """
    Create a constellation of satellites distributed across standard orbital bands.

    Parameters:
    -----------
    n_sats : int
        Total number of satellites to simulate (default: 10000)

    Returns:
    --------
    constellation : dict
        Dictionary of Keplerian orbital elements for tensorgator
    """

    # Define orbital bands (semi-major axis in km, inclination in degrees)
    bands = [
        {"altitude": 550, "inclination": 53.0},   # Starlink-like
        {"altitude": 600, "inclination": 97.6},   # SSO
        {"altitude": 500, "inclination": 30.0},   # Low inclination
        {"altitude": 700, "inclination": 0.0},    # Equatorial
    ]

    # Distribute satellites across bands
    sats_per_band = n_sats // len(bands)

    # Earth radius in km
    earth_radius = 6371.0

    # Initialize arrays for Keplerian elements
    a_list = []  # Semi-major axis
    e_list = []  # Eccentricity
    i_list = []  # Inclination
    omega_list = []  # Argument of periapsis
    Omega_list = []  # RAAN (Right Ascension of Ascending Node)
    M_list = []  # Mean anomaly

    for band in bands:
        # Semi-major axis (radius in km)
        a = earth_radius + band["altitude"]

        # Create satellites in this band
        for _ in range(sats_per_band):
            a_list.append(a)
            e_list.append(0.0)  # Circular orbits
            i_list.append(np.radians(band["inclination"]))

            # Randomize RAAN and mean anomaly for distribution
            omega_list.append(0.0)  # Circular orbit, so argument of periapsis doesn't matter
            Omega_list.append(np.random.uniform(0, 2 * np.pi))
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


def run_simulation(n_sats=10000, n_timesteps=100, duration_hours=24):
    """
    Run GPU-accelerated satellite simulation.

    Parameters:
    -----------
    n_sats : int
        Number of satellites to simulate
    n_timesteps : int
        Number of time steps
    duration_hours : float
        Simulation duration in hours
    """

    print(f"Setting up constellation with {n_sats} satellites...")
    constellation = create_constellation(n_sats)

    # Create time array
    time_array = np.linspace(0, duration_hours * 3600, n_timesteps)  # Convert hours to seconds

    # Select backend based on CUDA availability
    backend = 'cuda' if CUDA_AVAILABLE else 'cpu'

    print(f"Running simulation for {n_timesteps} timesteps over {duration_hours} hours...")
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

    print(f"\n✓ Simulation complete!")
    print(f"  Time elapsed: {elapsed:.3f} seconds")
    print(f"  Satellites: {n_sats}")
    print(f"  Timesteps: {n_timesteps}")
    print(f"  Total positions computed: {n_sats * n_timesteps:,}")
    print(f"  Performance: {(n_sats * n_timesteps) / elapsed:,.0f} positions/second")

    print(f"\nPosition array shape: {positions.shape}")
    print(f"Position array shape format: (timesteps, satellites, xyz)")

    return positions, time_array, constellation


if __name__ == "__main__":
    # Run simulation with 10k satellites
    positions, times, constellation = run_simulation(
        n_sats=10,
        n_timesteps=100,
        duration_hours=24
    )

    # Example: Access position of first satellite at first timestep
    print(f"\nExample - First satellite position at t=0:")
    print(f"  X: {positions[0, 0, 0]:.2f} km")
    print(f"  Y: {positions[0, 0, 1]:.2f} km")
    print(f"  Z: {positions[0, 0, 2]:.2f} km")

    # TODO: Future extension - generate star tracker data
    # This would involve:
    # 1. Choose observer satellite
    # 2. Transform all other satellites to observer's reference frame
    # 3. Project positions onto 2D image plane
    # 4. Filter by field of view
    # 5. Generate dot pattern representing visible satellites
