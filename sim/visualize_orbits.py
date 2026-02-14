"""
3D visualization of satellite orbits around Earth in ECI frame.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from mpl_toolkits.mplot3d import Axes3D
import os
from PIL import Image


def create_earth_sphere(ax, earth_image_path=None, radius=6371.0, resolution='high'):
    """
    Create a textured Earth sphere in the 3D plot.

    Parameters:
    -----------
    ax : matplotlib Axes3D
        The 3D axes to plot on
    earth_image_path : str, optional
        Path to Earth texture image. If None, uses a placeholder.
    radius : float
        Earth radius in km (default: 6371.0)
    resolution : str
        'high' for static images, 'low' for animations (default: 'high')
    """
    # Create sphere coordinates - adjust resolution based on use case
    if resolution == 'low':
        # Much lower resolution for faster animation rendering
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, 2 * np.pi, 20)
    else:
        # Higher resolution for static images
        u = np.linspace(0, 2 * np.pi, 500)
        v = np.linspace(0, np.pi, 350)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))

    # Load Earth texture if available
    if earth_image_path and os.path.exists(earth_image_path):
        try:
            img = Image.open(earth_image_path)
            img_array = np.array(img)

            # Normalize to [0, 1] range if needed
            if img_array.dtype == np.uint8:
                img_array = img_array / 255.0

            # Get dimensions
            img_height, img_width = img_array.shape[:2]

            # Create color array that matches the sphere mesh
            # Map UV coordinates to image coordinates
            colors = np.zeros((len(u), len(v), 3))

            for i in range(len(u)):
                for j in range(len(v)):
                    # Map u (longitude) to image width
                    img_x = int((u[i] / (2 * np.pi)) * img_width) % img_width
                    # Map v (latitude) to image height
                    img_y = int((v[j] / np.pi) * img_height) % img_height

                    # Get RGB color from image
                    if len(img_array.shape) == 3:  # RGB image
                        colors[i, j] = img_array[img_y, img_x, :3]
                    else:  # Grayscale
                        colors[i, j] = img_array[img_y, img_x]

            # Plot textured sphere
            ax.plot_surface(x, y, z, rstride=1, cstride=1,
                          facecolors=colors,
                          alpha=1.0, shade=False, antialiased=True)
            print(f"Successfully loaded Earth texture from: {earth_image_path}")
        except Exception as e:
            print(f"Warning: Could not load Earth texture ({e}), using solid color")
            ax.plot_surface(x, y, z, rstride=4, cstride=4, color='steelblue',
                          alpha=0.6, shade=True)
    else:
        # Use solid color as fallback
        print(f"Earth texture not found at {earth_image_path}, using solid color")
        ax.plot_surface(x, y, z, rstride=4, cstride=4, color='steelblue',
                      alpha=0.6, shade=True)


def visualize_orbits(positions, constellation=None, timestep_indices=None,
                     earth_image_path=None, save_path=None, show_trails=True,
                     trail_length=20, earth_radius=6371.0, title=None):
    """
    Visualize satellite orbits around Earth in ECI frame.

    Parameters:
    -----------
    positions : ndarray
        Position array with shape (timesteps, satellites, xyz) in km
    constellation : ndarray, optional
        Constellation orbital elements for reference
    timestep_indices : list, optional
        Specific timestep indices to plot. If None, plots all timesteps as trails.
    earth_image_path : str, optional
        Path to Earth texture image. If None, looks for default location.
    save_path : str, optional
        Path to save the figure. If None, displays interactively.
    show_trails : bool
        Whether to show orbital trails (default: True)
    trail_length : int
        Number of timesteps to include in trails (default: 20)
    earth_radius : float
        Earth radius in km (default: 6371.0)
    title : str, optional
        Custom title for the plot

    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """

    # Determine default Earth image path if not provided
    if earth_image_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Try earth.jpeg first, then fall back to earth_texture.png
        earth_jpeg_path = os.path.join(script_dir, 'assets', 'earth.jpeg')
        earth_png_path = os.path.join(script_dir, 'assets', 'earth_texture.png')
        if os.path.exists(earth_jpeg_path):
            earth_image_path = earth_jpeg_path
        else:
            earth_image_path = earth_png_path

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create Earth sphere
    create_earth_sphere(ax, earth_image_path, radius=earth_radius)

    # Get shape info
    n_timesteps, n_sats, _ = positions.shape

    # Determine which timesteps to plot
    if timestep_indices is None:
        # Use the last timestep for satellite positions
        timestep_indices = [n_timesteps - 1]

    # Plot satellite positions at specified timesteps
    for tidx in timestep_indices:
        # Plot satellite 0 (of interest) in red
        ax.scatter(positions[tidx, 0, 0],
                  positions[tidx, 0, 1],
                  positions[tidx, 0, 2],
                  c='red', s=50, marker='o', label='Satellite 0' if tidx == timestep_indices[0] else '',
                  edgecolors='darkred', linewidth=1.5, alpha=0.9)

        # Plot all other satellites in white
        if n_sats > 1:
            ax.scatter(positions[tidx, 1:, 0],
                      positions[tidx, 1:, 1],
                      positions[tidx, 1:, 2],
                      c='white', s=20, marker='o',
                      label='Other satellites' if tidx == timestep_indices[0] else '',
                      edgecolors='gray', linewidth=0.5, alpha=0.7)

    # Plot orbital trails if requested
    if show_trails and n_timesteps > 1:
        # Show complete orbit trail for satellite 0
        # Sample points evenly to avoid overcrowding while showing full orbit
        n_trail_points = min(500, n_timesteps)  # Limit trail points for performance
        trail_indices = np.linspace(0, n_timesteps-1, n_trail_points, dtype=int)

        trail_0 = positions[trail_indices, 0, :]
        ax.plot(trail_0[:, 0], trail_0[:, 1], trail_0[:, 2],
               'r-', linewidth=2, alpha=0.7, label='Sat 0 trail')

        # Plot trails for a few other satellites
        if n_sats > 1:
            # Sample a few satellites to show their trails
            n_sample_sats = min(8, n_sats-1)
            sample_sats = np.linspace(1, n_sats-1, n_sample_sats, dtype=int)
            for sat_idx in sample_sats:
                trail = positions[trail_indices, sat_idx, :]
                ax.plot(trail[:, 0], trail[:, 1], trail[:, 2],
                       'w-', linewidth=0.5, alpha=0.3)

    # Set axis properties
    max_range = np.max(np.abs(positions)) * 1.1
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    ax.set_xlabel('X (km) - ECI', fontsize=10)
    ax.set_ylabel('Y (km) - ECI', fontsize=10)
    ax.set_zlabel('Z (km) - ECI', fontsize=10)

    # Set title
    if title is None:
        title = f'Satellite Orbits in ECI Frame\n{n_sats} satellites, {n_timesteps} timesteps'
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add legend
    ax.legend(loc='upper right', fontsize=9)

    # Set viewing angle
    ax.view_init(elev=20, azim=45)

    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    # Add grid
    ax.grid(True, alpha=0.3)

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Orbit visualization saved to: {save_path}")
    else:
        plt.show()

    return fig, ax


def animate_orbits(positions, time_array=None, constellation=None, earth_image_path=None,
                   save_path=None, fps=10, trail_length=20, earth_radius=6371.0,
                   max_frames=500):
    """
    Create an animated visualization of satellite orbits.

    Parameters:
    -----------
    positions : ndarray
        Position array with shape (timesteps, satellites, xyz) in km
    time_array : ndarray, optional
        Time values for each timestep in seconds
    constellation : ndarray, optional
        Constellation orbital elements for reference
    earth_image_path : str, optional
        Path to Earth texture image
    save_path : str, optional
        Path to save the animation (must end in .mp4 or .gif)
    fps : int
        Frames per second for animation (default: 10)
    trail_length : int
        Number of previous positions to show as trails (default: 20)
    earth_radius : float
        Earth radius in km (default: 6371.0)
    max_frames : int
        Maximum number of frames in animation (subsamples if needed, default: 500)

    Returns:
    --------
    animation : matplotlib animation object
    """
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

    # Determine default Earth image path if not provided
    if earth_image_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Try earth.jpeg first, then fall back to earth_texture.png
        earth_jpeg_path = os.path.join(script_dir, 'assets', 'earth.jpeg')
        earth_png_path = os.path.join(script_dir, 'assets', 'earth_texture.png')
        if os.path.exists(earth_jpeg_path):
            earth_image_path = earth_jpeg_path
        else:
            earth_image_path = earth_png_path

    n_timesteps, n_sats, _ = positions.shape

    # Subsample frames if needed for reasonable animation length
    if n_timesteps > max_frames:
        frame_indices = np.linspace(0, n_timesteps-1, max_frames, dtype=int)
        print(f"Subsampling {n_timesteps} timesteps to {max_frames} frames for animation")
    else:
        frame_indices = np.arange(n_timesteps)
        max_frames = n_timesteps

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create Earth sphere (static) with lower resolution for faster rendering
    create_earth_sphere(ax, earth_image_path, radius=earth_radius, resolution='low')

    # Initialize empty plots for satellites (no trails)
    sat0_scatter = ax.scatter([], [], [], c='red', s=50, marker='o',
                             edgecolors='darkred', linewidth=1.5, alpha=0.9,
                             label='Satellite 0')
    other_sats_scatter = ax.scatter([], [], [], c='white', s=20, marker='o',
                                   edgecolors='gray', linewidth=0.5, alpha=0.7,
                                   label='Other satellites')

    # Set axis properties
    max_range = np.max(np.abs(positions)) * 1.1
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    ax.set_xlabel('X (km) - ECI', fontsize=10)
    ax.set_ylabel('Y (km) - ECI', fontsize=10)
    ax.set_zlabel('Z (km) - ECI', fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.view_init(elev=20, azim=45)
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True, alpha=0.3)

    title_text = ax.set_title('', fontsize=14, fontweight='bold')

    def update(frame_idx):
        # Get actual timestep index
        actual_frame = frame_indices[frame_idx]

        # Update satellite positions
        sat0_scatter._offsets3d = (positions[actual_frame, 0:1, 0],
                                   positions[actual_frame, 0:1, 1],
                                   positions[actual_frame, 0:1, 2])

        if n_sats > 1:
            other_sats_scatter._offsets3d = (positions[actual_frame, 1:, 0],
                                            positions[actual_frame, 1:, 1],
                                            positions[actual_frame, 1:, 2])

        # Update title with time information
        if time_array is not None:
            t_minutes = time_array[actual_frame]  # time_array is in minutes
            t_hours = t_minutes / 60
            title_text.set_text(f'Satellite Orbits in ECI Frame\nt = {t_hours:.2f} hours ({t_minutes:.1f} min)')
        else:
            title_text.set_text(f'Satellite Orbits in ECI Frame\nFrame {frame_idx}/{max_frames-1}')

        return sat0_scatter, other_sats_scatter, title_text

    # Create animation
    print(f"Creating animation with {max_frames} frames at {fps} fps...")
    anim = FuncAnimation(fig, update, frames=max_frames, interval=1000/fps, blit=False)

    # Save or show with progress
    if save_path:
        print(f"Saving animation to {save_path}...")
        print(f"Rendering {max_frames} frames (this may take a few minutes)...")

        try:
            from tqdm import tqdm

            if save_path.endswith('.gif'):
                writer = PillowWriter(fps=fps)
                with tqdm(total=max_frames, desc="Rendering", unit="frame") as pbar:
                    anim.save(save_path, writer=writer, progress_callback=lambda i, n: pbar.update(1))
            elif save_path.endswith('.mp4'):
                writer = FFMpegWriter(fps=fps, bitrate=1800)
                with tqdm(total=max_frames, desc="Rendering", unit="frame") as pbar:
                    anim.save(save_path, writer=writer, progress_callback=lambda i, n: pbar.update(1))
            else:
                raise ValueError("save_path must end with .gif or .mp4")
        except ImportError:
            # Fallback if tqdm not available
            print("(Install tqdm for progress bar: pip install tqdm)")
            if save_path.endswith('.gif'):
                writer = PillowWriter(fps=fps)
                anim.save(save_path, writer=writer)
            elif save_path.endswith('.mp4'):
                writer = FFMpegWriter(fps=fps, bitrate=1800)
                anim.save(save_path, writer=writer)
            else:
                raise ValueError("save_path must end with .gif or .mp4")

        print(f"✓ Animation saved to: {save_path}")
    else:
        plt.show()

    return anim
