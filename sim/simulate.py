"""
Simple example of running 10k+ satellite simulations in parallel on GPU using Tensorgator.
"""

import numpy as np
import tensorgator as tg
import time
import sys
import argparse
from params import SATELLITE, CONSTELLATION, SIMULATION, STAR_TRACKER, VISUALIZATION, CONSTANTS, PATHS

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
    # Tensorgator expects [a, e, inc, Omega, omega, M0] (RAAN before arg of periapsis)
    constellation = np.column_stack([
        np.array(a_list),
        np.array(e_list),
        np.array(i_list),
        np.array(Omega_list),
        np.array(omega_list),
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


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run satellite constellation simulation')
    parser.add_argument('--duration', type=float, default=SIMULATION['duration_hours'],
                        help=f'Simulation duration in hours (default: {SIMULATION["duration_hours"]:.2f} hours)')
    parser.add_argument('--dt', type=float, default=SIMULATION['dt_seconds'],
                        help=f'Time step in seconds (default: {SIMULATION["dt_seconds"]})')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate 3D visualization of satellite orbits and star tracker images')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save orbit visualization (default: saves to sim/plots/)')
    parser.add_argument('--animate', action='store_true',
                        help='Create animated visualization of orbits (requires --visualize)')
    parser.add_argument('--animation-fps', type=int, default=VISUALIZATION['animation_fps'],
                        help=f'Frames per second for animation (default: {VISUALIZATION["animation_fps"]})')
    parser.add_argument('--animation-frames', type=int, default=VISUALIZATION['animation_max_frames'],
                        help=f'Maximum number of frames in animation (default: {VISUALIZATION["animation_max_frames"]})')
    parser.add_argument('--max-satellites', type=int, default=VISUALIZATION['max_satellites'],
                        help='Maximum number of satellites to show in plots (default: all)')
    parser.add_argument('--earth-resolution', type=str, default=VISUALIZATION['earth_resolution'],
                        choices=['low', 'medium', 'high', 'ultra'],
                        help=f'Earth texture resolution: low, medium, high, ultra (default: {VISUALIZATION["earth_resolution"]})')
    args = parser.parse_args()

    # Run simulation
    positions, times, constellation = run_simulation(
        n_sats=CONSTELLATION['n_satellites'],
        duration_hours=args.duration,
        dt_seconds=args.dt,
        sat=SATELLITE
    )

    # Print detailed position evolution for satellite 0
    print(f"\n{'='*80}")
    print(f"Position Evolution for Satellite 0 (ECI frame)")
    print(f"{'='*80}")
    print(f"{'Time (min)':>12} {'X (km)':>12} {'Y (km)':>12} {'Z (km)':>12} {'R (km)':>12} {'V (km/s)':>12}")
    print(f"{'-'*80}")

    # Expected orbital velocity for satellite 0
    mu = CONSTANTS['mu']
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
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from sat.control.rtn_to_eci_propagate import eci_to_rtn_basis

    # Calculate observer velocity for RTN frame (using finite difference)
    # Positions are in meters, convert to km for RTN calculation
    observer_pos_km = positions[:, 0, :] / 1000  # Shape: (n_timesteps, 3)

    # Calculate velocity using central differences
    dt = times[1] - times[0]  # timestep in seconds
    observer_vel_km_s = np.zeros_like(observer_pos_km)
    observer_vel_km_s[0] = (observer_pos_km[1] - observer_pos_km[0]) / dt
    observer_vel_km_s[-1] = (observer_pos_km[-1] - observer_pos_km[-2]) / dt
    observer_vel_km_s[1:-1] = (observer_pos_km[2:] - observer_pos_km[:-2]) / (2 * dt)

    # Get RTN basis at first timestep to determine T direction
    basis_rtn = eci_to_rtn_basis(observer_pos_km[0], observer_vel_km_s[0])
    # basis_rtn has rows [R, T, N], so T direction is row 1
    t_direction_eci = basis_rtn[1, :]  # Tangential (along-track) direction in ECI

    print(f"Star tracker pointing in T (tangential) direction: {t_direction_eci}")

    # Render image sequence with T-direction pointing
    images, visible_sats_list, pixel_coords_list = render_star_tracker_sequence(
        positions,
        observer_index=STAR_TRACKER['observer_index'],
        fov_deg=STAR_TRACKER['fov_deg'],
        image_size=STAR_TRACKER['image_size'],
        pointing_direction=t_direction_eci
    )

    print(f"Star tracker images rendered:")
    print(f"  Total timesteps: {len(images)}")
    print(f"  Visible satellites per timestep: {[len(v) for v in visible_sats_list]}")

    # Create output directory
    output_dir = PATHS['data_dir']
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

    # Generate MP4 video from star tracker images
    print(f"\nGenerating MP4 video from star tracker sequence...")
    try:
        import cv2

        # Video parameters - 30 fps by default for smooth playback
        video_fps = 30

        video_path = os.path.join(output_dir, 'star_tracker_sequence.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, video_fps, (256, 256), isColor=False)

        for img_array in images:
            # Convert to 8-bit grayscale (0-255)
            img_uint8 = (img_array * 255).astype(np.uint8)
            video_writer.write(img_uint8)

        video_writer.release()
        print(f"Star tracker MP4 saved to: {video_path}")
        print(f"  Duration: {len(images)/video_fps:.1f} seconds at {video_fps} fps")
    except ImportError:
        print("Warning: opencv-python not installed. Skipping MP4 generation.")
        print("Install with: pip install opencv-python")

    # Visualize if requested
    if args.visualize:
        # 1. Visualize 3D orbits around Earth
        print(f"\nGenerating 3D orbit visualization...")
        from plots import plot_orbits_static, animate_orbits_fast

        # Create plots output directory
        plots_dir = PATHS['plots_dir']
        os.makedirs(plots_dir, exist_ok=True)

        # Earth texture path
        earth_texture_path = PATHS['earth_texture']

        # Save orbit visualization in sim/plots folder
        orbit_save_path = os.path.join(plots_dir, 'orbit_visualization.png') if not args.save else args.save
        plot_orbits_static(
            positions,
            save_path=orbit_save_path,
            title=f'Satellite Positions\n{positions.shape[1]} satellites over {args.duration*60:.0f} minutes',
            max_satellites=args.max_satellites,
            earth_texture_path=earth_texture_path
        )
        print(f"3D orbit visualization saved to: {orbit_save_path}")

        # Create animation if requested
        if args.animate:
            print(f"\nGenerating fast orbit animation...")
            anim_save_path = os.path.join(plots_dir, 'orbit_animation.gif')
            animate_orbits_fast(
                positions,
                times=times,
                save_path=anim_save_path,
                fps=args.animation_fps,
                max_frames=args.animation_frames,
                max_satellites=args.max_satellites,
                earth_texture_path=earth_texture_path
            )
            print(f"Animation complete!")

        # 2. Visualize star tracker image sequence
        n_plots = min(10, len(images))
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()

        for i in range(n_plots):
            axes[i].imshow(images[i], cmap='gray', origin='lower')
            axes[i].set_title(f't={i} ({len(visible_sats_list[i])} sats)')
            axes[i].axis('off')

        plt.suptitle('Star Tracker Image Sequence (Camera pointing in T direction)', fontsize=14)
        plt.tight_layout()

        tracker_save_path = os.path.join(output_dir, 'star_tracker_sequence.png')
        plt.savefig(tracker_save_path, dpi=150, bbox_inches='tight')
        print(f"Star tracker visualization saved to: {tracker_save_path}")
        plt.close()
