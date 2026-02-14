"""
Fast orbit visualization and animation for satellite simulations.

Optimized for performance with simplified rendering.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import os


def create_simple_earth(ax, radius_km=6378.137, earth_texture_path=None, resolution='medium'):
    """
    Create Earth sphere with optional texture.

    Parameters:
    -----------
    ax : matplotlib Axes3D
        The 3D axes to plot on
    radius_km : float
        Earth radius in km (default: 6378.137)
    earth_texture_path : str, optional
        Path to Earth texture image
    resolution : str
        'low' (30x20), 'medium' (60x40), 'high' (120x80), 'ultra' (240x160)
    """

    # Choose resolution
    res_map = {
        'low': (30, 20),
        'medium': (60, 40),
        'high': (120, 80),
        'ultra': (240, 160)
    }
    u_res, v_res = res_map.get(resolution, (60, 40))

    u = np.linspace(0, 2 * np.pi, u_res)
    v = np.linspace(0, np.pi, v_res)
    x = radius_km * np.outer(np.cos(u), np.sin(v))
    y = radius_km * np.outer(np.sin(u), np.sin(v))
    z = radius_km * np.outer(np.ones(np.size(u)), np.cos(v))

    # Try to load and apply texture
    if earth_texture_path and os.path.exists(earth_texture_path):
        try:
            from PIL import Image
            img = Image.open(earth_texture_path)
            img = img.resize((len(u), len(v)))  # Resize to match sphere resolution
            img_array = np.array(img) / 255.0  # Normalize to [0, 1]

            # Apply texture
            ax.plot_surface(x, y, z, rstride=1, cstride=1,
                          facecolors=plt.cm.colors.ListedColormap(
                              img_array.reshape(-1, img_array.shape[-1])
                          )(np.linspace(0, 1, x.size)).reshape(x.shape + (4,))[:,:,:3],
                          shade=False, antialiased=False)

            # Simpler approach - just map the texture colors directly
            # Get RGB values for each mesh point
            colors = np.zeros(x.shape + (3,))
            for i in range(len(u)):
                for j in range(len(v)):
                    colors[i, j] = img_array[j, i, :3] if img_array.ndim == 3 else [img_array[j, i]]*3

            ax.plot_surface(x, y, z, rstride=1, cstride=1,
                          facecolors=colors, shade=False, antialiased=False)

        except Exception as e:
            print(f"Could not load Earth texture: {e}")
            # Fallback to simple blue sphere
            ax.plot_surface(x, y, z, color='steelblue', alpha=0.6, shade=True)
    else:
        # Simple blue sphere - slightly transparent but solid enough to occlude
        ax.plot_surface(x, y, z, color='steelblue', alpha=0.7, shade=True)


def is_visible_from_view(positions, elev_deg, azim_deg, earth_radius_km):
    """
    Check which satellites are visible (not occluded by Earth) from viewing angle.

    Parameters:
    -----------
    positions : ndarray
        Satellite positions in km, shape (n_satellites, 3)
    elev_deg : float
        Elevation angle in degrees
    azim_deg : float
        Azimuthal angle in degrees
    earth_radius_km : float
        Earth radius in km

    Returns:
    --------
    visible_mask : ndarray
        Boolean array indicating which satellites are visible
    """
    # Calculate view direction from elev/azim
    elev_rad = np.radians(elev_deg)
    azim_rad = np.radians(azim_deg)

    view_direction = np.array([
        np.cos(elev_rad) * np.cos(azim_rad),
        np.cos(elev_rad) * np.sin(azim_rad),
        np.sin(elev_rad)
    ])

    # For each satellite, check if it's on the visible hemisphere
    # A satellite is visible if the angle between its position and view direction
    # is less than 90 degrees, accounting for Earth's curvature

    visible_mask = np.zeros(len(positions), dtype=bool)

    for i, pos in enumerate(positions):
        r = np.linalg.norm(pos)
        if r < earth_radius_km * 1.01:  # Too close to or inside Earth
            visible_mask[i] = False
            continue

        pos_normalized = pos / r

        # Dot product tells us if satellite is on front or back hemisphere
        dot_product = np.dot(pos_normalized, view_direction)

        # Calculate Earth's angular radius as seen from satellite distance
        # sin(angle) = earth_radius / satellite_distance
        sin_earth_angle = earth_radius_km / r
        cos_limb_angle = np.sqrt(max(0, 1 - sin_earth_angle**2))

        # Satellite is visible if it's beyond Earth's limb from viewer's perspective
        visible_mask[i] = dot_product > cos_limb_angle

    return visible_mask


def plot_orbits_static(positions, save_path=None, title=None, max_satellites=None,
                       earth_radius_km=6378.137, earth_texture_path=None, earth_resolution='medium'):
    """
    Fast static plot of satellite positions (no trajectories).

    Parameters:
    -----------
    positions : ndarray
        Position array with shape (timesteps, satellites, 3) in METERS
    save_path : str, optional
        Path to save the figure
    title : str, optional
        Custom title
    max_satellites : int, optional
        Maximum number of satellites to show (default: None = all satellites)
    earth_radius_km : float
        Earth radius in km (default: 6378.137)
    earth_texture_path : str, optional
        Path to Earth texture image
    earth_resolution : str
        Earth mesh resolution: 'low', 'medium', 'high', 'ultra' (default: 'medium')

    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    # Convert positions from meters to km
    positions_km = positions / 1000.0

    n_timesteps, n_sats, _ = positions_km.shape

    # Determine which satellites to show
    if max_satellites is not None and max_satellites < n_sats:
        # Show satellite 0 plus a sample of others
        other_indices = np.linspace(1, n_sats - 1, max_satellites - 1, dtype=int)
        show_indices = np.concatenate([[0], other_indices])
    else:
        show_indices = np.arange(n_sats)

    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Set view angle
    elev, azim = 20, 45
    ax.view_init(elev=elev, azim=azim)

    # Earth with texture - slightly transparent to show it's solid
    create_simple_earth(ax, radius_km=earth_radius_km, earth_texture_path=earth_texture_path,
                       resolution=earth_resolution)

    # Show satellite positions (last timestep, no trails)
    last_pos = positions_km[-1]

    # Filter satellites based on visibility (not occluded by Earth)
    if len(show_indices) > 1:
        other_pos = last_pos[show_indices[1:]]
        # Check which satellites are visible from current view angle
        visible_mask = is_visible_from_view(other_pos, elev, azim, earth_radius_km)
        visible_other_pos = other_pos[visible_mask]

        if len(visible_other_pos) > 0:
            ax.scatter(visible_other_pos[:, 0], visible_other_pos[:, 1], visible_other_pos[:, 2],
                      c='white', s=5, marker='.', alpha=0.5, linewidths=0,
                      label=f'Other satellites ({len(visible_other_pos)})')

    # Satellite 0 with short trail - only if visible
    sat0_visible = is_visible_from_view(np.array([last_pos[0]]), elev, azim, earth_radius_km)[0]

    if sat0_visible:
        trail_length = min(20, n_timesteps)
        trail_start = max(0, n_timesteps - trail_length)
        trail_pos = positions_km[trail_start:, 0, :]

        # Plot trail as a line - bright green
        ax.plot(trail_pos[:, 0], trail_pos[:, 1], trail_pos[:, 2],
                color='#00FF00', linewidth=1, alpha=0.6, zorder=999)

        # Plot current position using plot() instead of scatter() - bright green, no shading
        ax.plot([last_pos[0, 0]], [last_pos[0, 1]], [last_pos[0, 2]],
                marker='o', markersize=8, color='#00FF00', markeredgecolor='none',
                markerfacecolor='#00FF00', alpha=1.0, label='TreeSat', zorder=1000)

    # Set equal aspect and limits
    max_range = np.max(np.abs(positions_km)) * 1.1
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    ax.set_xlabel('X (km)', fontsize=10)
    ax.set_ylabel('Y (km)', fontsize=10)
    ax.set_zlabel('Z (km)', fontsize=10)

    if title is None:
        title = f'Satellite Orbits (ECI Frame)\n{n_sats} satellites, {n_timesteps} timesteps'
    ax.set_title(title, fontsize=12, fontweight='bold')

    ax.legend(loc='upper right', fontsize=9)
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True, alpha=0.2)

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"✓ Orbit plot saved to: {save_path}")

    return fig, ax


def animate_orbits_fast(positions, times=None, save_path=None, fps=20,
                        max_frames=200, max_satellites=None,
                        earth_radius_km=6378.137, earth_texture_path=None, earth_resolution='medium'):
    """
    Fast orbit animation (no trails, just satellite positions).

    Parameters:
    -----------
    positions : ndarray
        Position array with shape (timesteps, satellites, 3) in METERS
    times : ndarray, optional
        Time array in seconds
    save_path : str, optional
        Path to save (must end in .gif)
    fps : int
        Frames per second (default: 20)
    max_frames : int
        Maximum number of frames (default: 200)
    max_satellites : int, optional
        Maximum number of satellites to show (default: None = all satellites)
    earth_radius_km : float
        Earth radius in km (default: 6378.137)
    earth_texture_path : str, optional
        Path to Earth texture image
    earth_resolution : str
        Earth mesh resolution: 'low', 'medium', 'high', 'ultra' (default: 'medium')

    Returns:
    --------
    anim : matplotlib animation object
    """
    # Convert to km
    positions_km = positions / 1000.0

    n_timesteps, n_sats, _ = positions_km.shape

    # Determine which satellites to show
    if max_satellites is not None and max_satellites < n_sats:
        # Always show satellite 0 plus a sample of others
        other_indices = np.linspace(1, n_sats - 1, max_satellites - 1, dtype=int)
        show_sat_indices = other_indices
        n_show_sats = len(show_sat_indices)
    else:
        show_sat_indices = np.arange(1, n_sats)
        n_show_sats = n_sats - 1

    print(f"Showing {n_show_sats + 1} satellites (1 + {n_show_sats} others)")

    # Subsample frames if needed
    if n_timesteps > max_frames:
        frame_indices = np.linspace(0, n_timesteps - 1, max_frames, dtype=int)
        print(f"Subsampling {n_timesteps} timesteps -> {max_frames} frames")
    else:
        frame_indices = np.arange(n_timesteps)
        max_frames = n_timesteps

    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create Earth once (static)
    # TEST: Use 6000 km radius to see if clipping is from Earth or other satellites
    create_simple_earth(ax, earth_texture_path=earth_texture_path, resolution=earth_resolution)

    # Set up axis limits (static)
    max_range = np.max(np.abs(positions_km)) * 1.1
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    ax.set_xlabel('X (km)', fontsize=9)
    ax.set_ylabel('Y (km)', fontsize=9)
    ax.set_zlabel('Z (km)', fontsize=9)

    # Set view angle
    elev, azim = 20, 45
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True, alpha=0.2)

    # Initialize scatter plots - 1 pixel dots
    # Other satellites first - match static plot
    other_sats_scatter = ax.scatter([], [], [], c='white', s=5,
                                   marker='.', alpha=0.5, linewidths=0)

    # Satellite 0 - will be redrawn fresh each frame to ensure ALWAYS on top
    sat0_scatter = None
    sat0_trail = None  # Trail line

    title_text = ax.set_title('', fontsize=11)

    # Trail length for satellite 0
    trail_length = 0

    def update(frame_num):
        """Update function for animation."""
        nonlocal sat0_scatter, sat0_trail
        idx = frame_indices[frame_num]

        # Update other satellites first - filter for visibility
        if n_show_sats > 0:
            pos_others = positions_km[idx, show_sat_indices, :]
            # Only show satellites visible from current view (not behind Earth)
            visible_mask = is_visible_from_view(pos_others, elev, azim, earth_radius_km)
            visible_pos = pos_others[visible_mask]

            if len(visible_pos) > 0:
                other_sats_scatter._offsets3d = (visible_pos[:, 0],
                                                 visible_pos[:, 1],
                                                 visible_pos[:, 2])
            else:
                other_sats_scatter._offsets3d = ([], [], [])

        # Get satellite 0 position
        pos0_x = positions_km[idx, 0, 0]
        pos0_y = positions_km[idx, 0, 1]
        pos0_z = positions_km[idx, 0, 2]
        pos0 = np.array([[pos0_x, pos0_y, pos0_z]])

        # Debug: Check if position is valid
        if frame_num % 10 == 0:  # Print every 10th frame
            r = np.sqrt(pos0_x**2 + pos0_y**2 + pos0_z**2)
            print(f"Frame {frame_num}: Sat0 @ ({pos0_x:.1f}, {pos0_y:.1f}, {pos0_z:.1f}), r={r:.1f} km")

        # REMOVE old satellite 0 and trail, redraw fresh - ensures it's ALWAYS last/on top
        if sat0_scatter is not None:
            sat0_scatter.remove()
            sat0_scatter = None
        if sat0_trail is not None:
            sat0_trail.remove()
            sat0_trail = None

        # Check if satellite 0 is visible (not behind Earth)
        sat0_visible = is_visible_from_view(pos0, elev, azim, earth_radius_km)[0]

        if sat0_visible:
            # Draw trail for satellite 0 - bright green
            trail_start_idx = max(0, idx - trail_length)
            trail_positions = positions_km[trail_start_idx:idx+1, 0, :]
            if len(trail_positions) > 1:
                sat0_trail, = ax.plot(trail_positions[:, 0], trail_positions[:, 1], trail_positions[:, 2],
                                     color='#00FF00', linewidth=1, alpha=0.6, zorder=999)

            # Draw satellite 0 FRESH each frame using plot() - bright green, no shading
            sat0_scatter, = ax.plot([pos0_x], [pos0_y], [pos0_z],
                                   marker='o', markersize=8, color='#00FF00',
                                   markeredgecolor='none', markerfacecolor='#00FF00',
                                   alpha=1.0, zorder=1000)

        # Update title
        if times is not None:
            t_min = times[idx] / 60
            title_text.set_text(f'Satellite Positions\nTime: {t_min:.1f} min')
        else:
            title_text.set_text(f'Satellite Positions\nFrame {frame_num}/{max_frames}')

        # Return in order: other sats first, then trail, then sat0 on top
        artists = [other_sats_scatter, title_text]
        if sat0_trail is not None:
            artists.append(sat0_trail)
        if sat0_scatter is not None:
            artists.append(sat0_scatter)
        return artists

    # Create animation
    print(f"Creating animation: {max_frames} frames @ {fps} fps")
    anim = FuncAnimation(fig, update, frames=max_frames,
                        interval=1000/fps, blit=False)

    # Save
    if save_path:
        print(f"Saving to {save_path}...")

        # Try to use tqdm for progress
        try:
            from tqdm import tqdm
            writer = PillowWriter(fps=fps)
            with tqdm(total=max_frames, desc="Rendering", unit="frame") as pbar:
                anim.save(save_path, writer=writer,
                         progress_callback=lambda i, n: pbar.update(1))
        except ImportError:
            print("(Install tqdm for progress bar: pip install tqdm)")
            writer = PillowWriter(fps=fps)
            anim.save(save_path, writer=writer)

        print(f"✓ Animation saved: {save_path}")
        print(f"  Duration: {max_frames/fps:.1f}s at {fps} fps")

    return anim
