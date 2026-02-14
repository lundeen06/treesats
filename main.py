"""
Main entry point for TreeSats satellite simulation and navigation system.

Modes:
- train: Generate training data (simulation + star tracker images)
- pipeline: Run full system (simulation → autolabel → angles-only nav)
"""

import numpy as np
import argparse
import os
import sys

# Import from sim module
from sim.simulate import run_simulation
from sim.params import SATELLITE, CONSTELLATION, SIMULATION, STAR_TRACKER, VISUALIZATION, CONSTANTS, PATHS


def generate_training_data(args):
    """
    Generate training data: run full simulation and save star tracker images.

    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    """
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

    # Print every 1 minute (based on dt)
    interval = max(1, int(60 / args.dt))  # Print every ~1 minute
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
    from sim.star_tracker import render_star_tracker_sequence
    from sat.control.rtn_to_eci_propagate import eci_to_rtn_basis
    import matplotlib.pyplot as plt
    from PIL import Image

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
        from sim.plots import plot_orbits_static, animate_orbits_fast

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

    return positions, times, constellation, images


def run_pipeline(args):
    """
    Run full pipeline: sequential time-stepping with image → autolabel → angles-only nav.

    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    """
    from sim.star_tracker.star_tracker import render_star_tracker_image
    from sat.control.rtn_to_eci_propagate import eci_to_rtn_basis

    print("="*80)
    print("RUNNING FULL PIPELINE (Sequential Time-Stepping)")
    print("="*80)

    # Step 1: Propagate all orbits upfront
    print("\n[1/4] Propagating all satellite orbits...")
    positions, times, constellation = run_simulation(
        n_sats=CONSTELLATION['n_satellites'],
        duration_hours=args.duration,
        dt_seconds=args.dt,
        sat=SATELLITE
    )
    n_timesteps = len(times)
    n_satellites = positions.shape[1]
    observer_index = STAR_TRACKER['observer_index']

    print(f"  Propagated {n_satellites} satellites for {n_timesteps} timesteps")
    print(f"  Time step: {args.dt}s, Duration: {args.duration*60:.1f} min")

    # Step 2: Calculate RTN basis for camera pointing (do this once)
    print("\n[2/4] Setting up camera pointing (T direction in RTN frame)...")
    observer_pos_km = positions[:, observer_index, :] / 1000
    dt = times[1] - times[0]
    observer_vel_km_s = np.zeros_like(observer_pos_km)
    observer_vel_km_s[0] = (observer_pos_km[1] - observer_pos_km[0]) / dt
    observer_vel_km_s[-1] = (observer_pos_km[-1] - observer_pos_km[-2]) / dt
    observer_vel_km_s[1:-1] = (observer_pos_km[2:] - observer_pos_km[:-2]) / (2 * dt)

    # Get T direction for camera pointing (tangential direction)
    basis_rtn = eci_to_rtn_basis(observer_pos_km[0], observer_vel_km_s[0])
    t_direction_eci = basis_rtn[1, :]
    print(f"  Camera boresight: T-direction = {t_direction_eci}")

    # Step 3: Sequential time-stepping loop
    print("\n[3/4] Starting sequential time-stepping loop...")
    print(f"{'Time':>8} {'Step':>6} {'Visible':>8} {'Status':>20}")
    print("-" * 50)

    for t_idx in range(n_timesteps):
        current_time = times[t_idx]
        current_positions_m = positions[t_idx]  # Shape: (n_satellites, 3) in meters
        current_positions_km = current_positions_m / 1000  # Convert to km

        # Observer position at this timestep
        observer_pos = current_positions_km[observer_index]

        # Get positions of all other satellites (exclude observer)
        other_indices = np.arange(n_satellites) != observer_index
        other_positions = current_positions_km[other_indices]

        # Generate star tracker image for THIS timestep only
        image, visible_sats, pixel_coords = render_star_tracker_image(
            observer_pos=observer_pos,
            satellite_positions=other_positions,
            fov_deg=STAR_TRACKER['fov_deg'],
            image_size=STAR_TRACKER['image_size'],
            pointing_direction=t_direction_eci
        )

        # TODO: Feed image to autolabeler model
        # predictions = autolabel_model(image)

        # TODO: Use predictions for angles-only navigation
        # nav_update = angles_only_nav(predictions, current_state)

        # Progress output (print every 10 timesteps or first/last)
        if t_idx % 10 == 0 or t_idx == n_timesteps - 1:
            print(f"{current_time:8.1f}s {t_idx:6d} {len(visible_sats):8d} {'Image generated':>20}")

    # Step 4: Summary
    print("\n[4/4] Pipeline summary...")
    print("  → Orbital propagation: ✓ Complete")
    print("  → Star tracker images: ✓ Generated sequentially")
    print("  → Autolabeler: TODO")
    print("  → Angles-only nav: TODO")

    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='TreeSats: Satellite Simulation and Navigation System')

    # Mode selection
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'pipeline'],
                        help='Mode: train (generate training data) or pipeline (run full system)')

    # Simulation parameters
    parser.add_argument('--duration', type=float, default=SIMULATION['duration_hours'],
                        help=f'Simulation duration in hours (default: {SIMULATION["duration_hours"]:.2f} hours)')
    parser.add_argument('--dt', type=float, default=SIMULATION['dt_seconds'],
                        help=f'Time step in seconds (default: {SIMULATION["dt_seconds"]})')

    # Visualization parameters (for train mode)
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

    # Run appropriate mode
    if args.mode == 'train':
        print("Running in TRAINING DATA GENERATION mode\n")
        generate_training_data(args)
    elif args.mode == 'pipeline':
        print("Running in FULL PIPELINE mode\n")
        run_pipeline(args)
