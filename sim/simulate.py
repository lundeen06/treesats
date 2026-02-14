"""
Simple example of running 10k+ satellite simulations in parallel on GPU using Tensorgator.
"""

import numpy as np
import tensorgator as tg
import time
import sys
import argparse

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
    Create a constellation of satellites distributed across standard orbital bands.

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
        Satellite of interest is at index 0, remaining satellites fill the constellation.
    """

    # Define orbital bands (altitude in km, inclination in degrees)
    bands = [
        {"altitude": 550, "inclination": 53.0},   # Starlink-like
        {"altitude": 600, "inclination": 97.6},   # SSO
        {"altitude": 500, "inclination": 30.0},   # Low inclination
        {"altitude": 700, "inclination": 0.0},    # Equatorial
    ]

    # Earth radius in meters (tensorgator expects meters!)
    earth_radius = 6378136.3  # meters

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

    # Distribute remaining satellites across bands
    remaining_sats = n_sats - sat_index
    sats_per_band = remaining_sats // len(bands)

    for band in bands:
        # Semi-major axis (radius in meters)
        # Convert altitude from km to meters
        a = earth_radius + band["altitude"] * 1000

        # Create satellites in this band
        for _ in range(sats_per_band):
            a_list.append(a)
            e_list.append(0.0)  # Circular orbits
            # Randomize inclination within ±5 degrees of the band inclination
            i_list.append(np.radians(band["inclination"] + np.random.uniform(-5, 5)))

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
    """

    np.random.seed(21)

    print(f"Setting up constellation with {n_sats} satellites...")
    constellation = create_constellation(n_sats, sat=sat)

    # Create time array - TensorGator expects times in seconds since reference epoch
    # Using np.arange to create array from 0 to duration with dt_seconds step size
    time_array = np.arange(0, duration_hours * 3600, dt_seconds)  # Convert hours to seconds
    n_timesteps = len(time_array)

    # Select backend based on CUDA availability
    backend = 'cuda' if CUDA_AVAILABLE else 'cpu'

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


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run satellite constellation simulation')
    parser.add_argument('--duration', type=float, default=100/60,
                        help='Simulation duration in hours (default: 100 minutes)')
    parser.add_argument('--dt', type=float, default=5,
                        help='Time step in seconds (default: 5)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate 3D visualization of satellite orbits and star tracker images')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save orbit visualization (default: saves to sat/data/)')
    parser.add_argument('--animate', action='store_true',
                        help='Create animated visualization of orbits (requires --visualize)')
    parser.add_argument('--animation-fps', type=int, default=10,
                        help='Frames per second for animation (default: 10)')
    parser.add_argument('--animation-frames', type=int, default=200,
                        help='Maximum number of frames in animation (default: 200)')
    args = parser.parse_args()

    # Define the satellite of interest (index 0 in the constellation)
    sat = {
        'altitude': 550,        # km above Earth's surface
        'eccentricity': 0.0001, # 0 = circular orbit
        'inclination': 53.0,    # degrees
        'omega': 0.0,           # argument of periapsis (degrees)
        'Omega': 0.0,           # RAAN - right ascension of ascending node (degrees)
        'M': 0.0                # mean anomaly (degrees) - starting position in orbit
    }

    # Run simulation
    positions, times, constellation = run_simulation(
        n_sats=10000,
        duration_hours=args.duration,
        dt_seconds=args.dt,
        sat=sat
    )

    # Print detailed position evolution for satellite 0
    print(f"\n{'='*80}")
    print(f"Position Evolution for Satellite 0 (ECI frame)")
    print(f"{'='*80}")
    print(f"{'Time (min)':>12} {'X (km)':>12} {'Y (km)':>12} {'Z (km)':>12} {'R (km)':>12} {'V (km/s)':>12}")
    print(f"{'-'*80}")

    # Expected orbital velocity for satellite 0
    mu = 3.986004418e14  # m^3/s^2
    a0 = constellation[0, 0]  # semi-major axis in meters
    expected_velocity = np.sqrt(mu / a0) / 1000  # km/s
    print(f"Expected orbital velocity: {expected_velocity:.3f} km/s\n")

    # Print every 1 minute (12 timesteps at 5s intervals)
    interval = 12  # 1 minute = 12 * 5 seconds
    for i in range(0, len(times), interval):
        t_min = times[i] / 60
        x, y, z = positions[i, 0, :] / 1000  # Satellite 0, convert to km
        r = np.linalg.norm(positions[i, 0, :]) / 1000  # Distance from Earth center

        # Calculate velocity by finite difference
        if i < len(times) - 1:
            dt = times[i+1] - times[i]
            vel = (positions[i+1, 0, :] - positions[i, 0, :]) / dt  # m/s
            v_mag = np.linalg.norm(vel) / 1000  # km/s
        else:
            v_mag = 0.0

        print(f"{t_min:12.2f} {x:12.2f} {y:12.2f} {z:12.2f} {r:12.2f} {v_mag:12.3f}")

    # Generate star tracker images
    print(f"\nGenerating star tracker images...")
    from star_tracker import render_star_tracker_sequence
    import matplotlib.pyplot as plt
    from PIL import Image
    import os

    # Render image sequence
    images, visible_sats_list, pixel_coords_list = render_star_tracker_sequence(
        positions,
        observer_index=0,
        fov_deg=15.0,
        image_size=256
    )

    print(f"Star tracker images rendered:")
    print(f"  Total timesteps: {len(images)}")
    print(f"  Visible satellites per timestep: {[len(v) for v in visible_sats_list]}")

    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'sat', 'data')
    os.makedirs(output_dir, exist_ok=True)

    # Save raw 256x256 star tracker images
    print(f"\nSaving raw star tracker images to {output_dir}/...")
    for i, img_array in enumerate(images):
        # Convert to 8-bit grayscale (0-255)
        img_uint8 = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(img_uint8, mode='L')
        img_path = os.path.join(output_dir, f'star_tracker_{i:04d}.png')
        img.save(img_path)

    print(f"Saved {len(images)} raw star tracker images (256x256 pixels)")

    # Visualize if requested
    if args.visualize:
        # 1. Visualize 3D orbits around Earth
        print(f"\nGenerating 3D orbit visualization...")
        from visualize_orbits import visualize_orbits, animate_orbits

        # Save orbit visualization in sim folder
        sim_dir = os.path.dirname(__file__)
        orbit_save_path = os.path.join(sim_dir, 'orbit_visualization.png') if not args.save else args.save
        visualize_orbits(
            positions,
            constellation=constellation,
            save_path=orbit_save_path,
            show_trails=False,  # No trails in static image
            title=f'Satellite Constellation Orbits\n{positions.shape[1]} satellites over {args.duration} hours'
        )
        print(f"3D orbit visualization saved to: {orbit_save_path}")

        # Create animation if requested
        if args.animate:
            print(f"\nGenerating orbit animation...")
            anim_save_path = os.path.join(sim_dir, 'orbit_animation.gif')
            animate_orbits(
                positions,
                time_array=times,
                constellation=constellation,
                save_path=anim_save_path,
                fps=args.animation_fps,
                max_frames=args.animation_frames
            )
            print(f"Animation saved! Duration: ~{args.animation_frames/args.animation_fps:.1f} seconds")

        # 2. Visualize star tracker image sequence
        n_plots = min(10, len(images))
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()

        for i in range(n_plots):
            axes[i].imshow(images[i], cmap='gray', origin='lower')
            axes[i].set_title(f't={i} ({len(visible_sats_list[i])} sats)')
            axes[i].axis('off')

        plt.suptitle('Star Tracker Image Sequence (Camera pointing in +x_eci)', fontsize=14)
        plt.tight_layout()

        tracker_save_path = os.path.join(output_dir, 'star_tracker_sequence.png')
        plt.savefig(tracker_save_path, dpi=150, bbox_inches='tight')
        print(f"Star tracker visualization saved to: {tracker_save_path}")
        plt.close()
