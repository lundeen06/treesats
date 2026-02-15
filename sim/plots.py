"""
Fast orbit visualization and animation for satellite simulations.

Optimized for performance with simplified rendering.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import os


def create_simple_earth(ax, radius_km=6378.137, earth_texture_path=None, resolution='medium', center=(0, 0, 0)):
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
    center : tuple of float, optional
        (x, y, z) center of sphere in same units as radius_km. Default (0, 0, 0).
    """
    cx, cy, cz = center[0], center[1], center[2]

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
    x = cx + radius_km * np.outer(np.cos(u), np.sin(v))
    y = cy + radius_km * np.outer(np.sin(u), np.sin(v))
    z = cz + radius_km * np.outer(np.ones(np.size(u)), np.cos(v))

    surfaces = []
    # Try to load and apply texture
    if earth_texture_path and os.path.exists(earth_texture_path):
        try:
            from PIL import Image
            img = Image.open(earth_texture_path)
            img = img.resize((len(u), len(v)))  # Resize to match sphere resolution
            img_array = np.array(img) / 255.0  # Normalize to [0, 1]

            # Apply texture
            surfaces.append(ax.plot_surface(x, y, z, rstride=1, cstride=1,
                          facecolors=plt.cm.colors.ListedColormap(
                              img_array.reshape(-1, img_array.shape[-1])
                          )(np.linspace(0, 1, x.size)).reshape(x.shape + (4,))[:,:,:3],
                          shade=False, antialiased=False))

            # Simpler approach - just map the texture colors directly
            # Get RGB values for each mesh point
            colors = np.zeros(x.shape + (3,))
            for i in range(len(u)):
                for j in range(len(v)):
                    colors[i, j] = img_array[j, i, :3] if img_array.ndim == 3 else [img_array[j, i]]*3

            surfaces.append(ax.plot_surface(x, y, z, rstride=1, cstride=1,
                          facecolors=colors, shade=False, antialiased=False))

        except Exception as e:
            print(f"Could not load Earth texture: {e}")
            # Fallback to simple blue sphere
            surfaces.append(ax.plot_surface(x, y, z, color='steelblue', alpha=0.6, shade=True))
    else:
        # Simple blue sphere - slightly transparent but solid enough to occlude
        surfaces.append(ax.plot_surface(x, y, z, color='steelblue', alpha=0.7, shade=True))
    return surfaces


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
                      label=f'Other satellites')

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
        print(f"Orbit plot saved to: {save_path}")

    return fig, ax


def animate_orbits_fast(positions, times=None, save_path=None, fps=20,
                        max_frames=200, max_satellites=None,
                        earth_radius_km=6378.137, earth_texture_path=None, earth_resolution='medium',
                        other_sat_color='white', adversary_index=None, constellation_start_index=None,
                        constellation_color='gray'):
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
    other_sat_color : str
        Color for "other" satellites (default: 'white'). When adversary_index=1, this is adversary (red).
    adversary_index : int, optional
        If set (e.g. 1), this index is drawn as single large marker with other_sat_color (adversary).
    constellation_start_index : int, optional
        If set (e.g. 2), indices >= this are drawn as small constellation_color dots (constellation).
    constellation_color : str
        Color for constellation dots when constellation_start_index is set (default: 'gray').

    Returns:
    --------
    anim : matplotlib animation object
    """
    # Convert to km
    positions_km = positions / 1000.0

    n_timesteps, n_sats, _ = positions_km.shape
    use_adversary_constellation = (adversary_index is not None and constellation_start_index is not None
                                    and constellation_start_index <= n_sats)

    # Determine which satellites to show
    if use_adversary_constellation:
        adversary_indices = np.array([adversary_index]) if adversary_index < n_sats else np.array([], dtype=int)
        constellation_indices = np.arange(constellation_start_index, n_sats) if constellation_start_index < n_sats else np.array([], dtype=int)
        n_show_sats = 1 + len(adversary_indices) + len(constellation_indices)
        print(f"Showing chief (green), adversary (red), constellation (gray): 1 + 1 + {len(constellation_indices)} sats")
    elif max_satellites is not None and max_satellites < n_sats:
        other_indices = np.linspace(1, n_sats - 1, max_satellites - 1, dtype=int)
        show_sat_indices = other_indices
        n_show_sats = len(show_sat_indices)
        adversary_indices = np.array([], dtype=int)
        constellation_indices = np.array([], dtype=int)
    else:
        show_sat_indices = np.arange(1, n_sats)
        n_show_sats = n_sats - 1
        adversary_indices = np.array([], dtype=int)
        constellation_indices = np.array([], dtype=int)

    if not use_adversary_constellation:
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
    create_simple_earth(ax, radius_km=earth_radius_km, earth_texture_path=earth_texture_path,
                        resolution=earth_resolution)

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

    # When adversary+constellation mode: adversary (red, large), constellation (gray, small)
    # Otherwise: single "other" scatter
    if use_adversary_constellation:
        adversary_scatter = ax.scatter([], [], [], c=other_sat_color, s=80, marker='o',
                                       alpha=0.95, linewidths=0, label='Adversary')
        constellation_scatter = ax.scatter([], [], [], c=constellation_color, s=5, marker='.',
                                           alpha=0.6, linewidths=0, label='Constellation')
        other_sats_scatter = None
    else:
        other_marker_size = 80 if n_sats <= 3 else 5
        other_alpha = 0.9 if n_sats <= 3 else 0.5
        other_sats_scatter = ax.scatter([], [], [], c=other_sat_color, s=other_marker_size,
                                       marker='o', alpha=other_alpha, linewidths=0, label='Other sats')
        adversary_scatter = None
        constellation_scatter = None

    # Legend: Chief is redrawn each frame so use a proxy; include all satellite types
    proxy_chief = Line2D([0], [0], linestyle='none', marker='o', markerfacecolor='#00FF00',
                         markeredgecolor='none', markersize=8, label='Chief (TreeSat)')
    if use_adversary_constellation:
        ax.legend(handles=[proxy_chief, adversary_scatter, constellation_scatter],
                  labels=['Chief (TreeSat)', 'Adversary', 'Constellation'],
                  loc='upper left', fontsize=9)
    else:
        ax.legend(handles=[proxy_chief, other_sats_scatter],
                  labels=['Chief (TreeSat)', 'Other sats'],
                  loc='upper left', fontsize=9)

    # Satellite 0 - will be redrawn fresh each frame to ensure ALWAYS on top
    sat0_scatter = None
    sat0_trail = None  # Trail line

    title_text = ax.set_title('', fontsize=11)

    # Trail length for satellite 0 (show trail in adversary scenario)
    trail_length = min(40, max_frames // 2) if n_sats <= 3 else 0

    def update(frame_num):
        """Update function for animation."""
        nonlocal sat0_scatter, sat0_trail
        idx = frame_indices[frame_num]

        # Update other satellites: either adversary+constellation or single "others" scatter
        if use_adversary_constellation:
            if len(adversary_indices) > 0:
                pos_adv = positions_km[idx, adversary_indices[0], :]
                adv_visible = is_visible_from_view(pos_adv.reshape(1, 3), elev, azim, earth_radius_km)[0]
                if adv_visible:
                    adversary_scatter._offsets3d = (pos_adv[0:1], pos_adv[1:2], pos_adv[2:3])
                else:
                    adversary_scatter._offsets3d = ([], [], [])
            if len(constellation_indices) > 0:
                pos_const = positions_km[idx, constellation_indices, :]
                visible_mask = is_visible_from_view(pos_const, elev, azim, earth_radius_km)
                visible_pos = pos_const[visible_mask]
                if len(visible_pos) > 0:
                    constellation_scatter._offsets3d = (visible_pos[:, 0], visible_pos[:, 1], visible_pos[:, 2])
                else:
                    constellation_scatter._offsets3d = ([], [], [])
        elif other_sats_scatter is not None and n_show_sats > 0:
            pos_others = positions_km[idx, show_sat_indices, :]
            visible_mask = is_visible_from_view(pos_others, elev, azim, earth_radius_km)
            visible_pos = pos_others[visible_mask]
            if len(visible_pos) > 0:
                other_sats_scatter._offsets3d = (visible_pos[:, 0], visible_pos[:, 1], visible_pos[:, 2])
            else:
                other_sats_scatter._offsets3d = ([], [], [])

        # Get satellite 0 position
        pos0_x = positions_km[idx, 0, 0]
        pos0_y = positions_km[idx, 0, 1]
        pos0_z = positions_km[idx, 0, 2]
        pos0 = np.array([[pos0_x, pos0_y, pos0_z]])

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

        # Return artists: other/adversary/constellation, then trail, then sat0 on top
        artists = [title_text]
        if use_adversary_constellation:
            artists.append(adversary_scatter)
            artists.append(constellation_scatter)
        elif other_sats_scatter is not None:
            artists.append(other_sats_scatter)
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

        print(f"Animation saved: {save_path}")
        print(f"  Duration: {max_frames/fps:.1f}s at {fps} fps")

    return anim


def plot_collision_scenario(
    save_path=None,
    earth_radius_km=6378.137,
    earth_texture_path=None,
    earth_resolution='medium',
    show_evasion=True,
    params=None,
):
    """
    Plot chief vs deputy (Hohmann transfer) collision scenario: chief on circular orbit,
    deputy on Hohmann that rendezvouses at half transfer. Optionally show evaded chief
    trajectory after collision-avoidance delta-v.

    Parameters
    ----------
    save_path : str, optional
        Path to save the figure (e.g. PATHS['plots_dir'] + '/collision_scenario.png').
    earth_radius_km, earth_texture_path, earth_resolution
        Passed to create_simple_earth.
    show_evasion : bool
        If True, compute delta-v and plot evaded chief trajectory.
    params : dict, optional
        If provided, must have keys: SATELLITE, CONSTELLATION, CONSTANTS, PATHS, VISUALIZATION.
        If None, imports from params (when run from sim/).

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    if params is None:
        try:
            from params import SATELLITE, CONSTELLATION, CONSTANTS, PATHS, VISUALIZATION
        except ImportError:
            import os
            _sim_dir = os.path.dirname(os.path.abspath(__file__))
            import sys
            if _sim_dir not in sys.path:
                sys.path.insert(0, _sim_dir)
            from params import SATELLITE, CONSTELLATION, CONSTANTS, PATHS, VISUALIZATION
    else:
        SATELLITE = params['SATELLITE']
        CONSTELLATION = params['CONSTELLATION']
        CONSTANTS = params['CONSTANTS']
        PATHS = params.get('PATHS', {})
        VISUALIZATION = params.get('VISUALIZATION', {})

    import sys
    project_root = PATHS.get('project_root', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from sat.control.rtn_to_eci_propagate import (
        propagate_chief,
        propagate_constellation,
        eci_to_relative_rtn,
        deputy_state_for_rendezvous,
    )
    from sat.control.collision_avoidance import (
        apply_impulse_eci,
        collision_avoidance_delta_v,
        DEFAULT_RADIUS_KM,
    )

    alt_chief = float(SATELLITE['altitude'])
    inc_deg = float(SATELLITE['inclination'])
    earth_km = CONSTANTS['earth_radius_km']
    mu_km = CONSTANTS['mu'] / 1e9
    r_mag = earth_km + alt_chief
    v_mag = np.sqrt(mu_km / r_mag)
    inc = np.radians(inc_deg)
    r_chief = np.array([r_mag, 0.0, 0.0])
    v_chief = np.array([0.0, v_mag * np.cos(inc), v_mag * np.sin(inc)])

    r_deputy, v_deputy, r_meet, v_meet, T_rendezvous = deputy_state_for_rendezvous(
        r_chief, v_chief, T_rendezvous=None, backend='cpu',
    )
    constellation_rtn = eci_to_relative_rtn(r_chief, v_chief, r_deputy, v_deputy)
    constellation_rtn = np.asarray(constellation_rtn, dtype=float).reshape(1, 6)

    T_orbit = 2 * np.pi * np.sqrt(((earth_km + alt_chief) ** 3) / mu_km)
    time_array = np.linspace(0, T_orbit, 150)

    pos_chief = propagate_chief(r_chief, v_chief, time_array, backend='cpu')
    pos_deputy = propagate_constellation(
        r_chief, v_chief, constellation_rtn, time_array, backend='cpu', return_velocity=False
    )
    if pos_deputy.ndim == 2:
        pos_deputy = pos_deputy[:, np.newaxis, :]
    pos_deputy = pos_deputy[:, 0, :]

    treesat_color = VISUALIZATION.get('treesat_color', '#00FF00')
    other_color = VISUALIZATION.get('other_sat_color', 'white')
    elev = VISUALIZATION.get('view_elevation', 20)
    azim = VISUALIZATION.get('view_azimuth', 45)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim)
    create_simple_earth(ax, radius_km=earth_radius_km, earth_texture_path=earth_texture_path,
                       resolution=earth_resolution)

    ax.plot(pos_chief[:, 0], pos_chief[:, 1], pos_chief[:, 2],
            color=treesat_color, linewidth=1.5, alpha=0.9, label='Chief (TreeSat)')
    ax.plot(pos_deputy[:, 0], pos_deputy[:, 1], pos_deputy[:, 2],
            color=other_color, linewidth=1.2, alpha=0.8, label='Deputy (rendezvous)')
    ax.plot([pos_chief[-1, 0]], [pos_chief[-1, 1]], [pos_chief[-1, 2]],
            marker='o', markersize=8, color=treesat_color, markeredgecolor='none',
            linestyle='')
    ax.plot([pos_deputy[-1, 0]], [pos_deputy[-1, 1]], [pos_deputy[-1, 2]],
            marker='o', markersize=6, color=other_color, alpha=0.9, linestyle='')

    if show_evasion:
        delta_v = collision_avoidance_delta_v(
            r_chief, v_chief, constellation_rtn,
            backend='cpu', radius_km=DEFAULT_RADIUS_KM, n_times_orbit=80,
        )
        r_ev, v_ev = apply_impulse_eci(r_chief, v_chief, delta_v)
        pos_evaded = propagate_chief(r_ev, v_ev, time_array, backend='cpu')
        ax.plot(pos_evaded[:, 0], pos_evaded[:, 1], pos_evaded[:, 2],
                color='cyan', linewidth=1.2, alpha=0.7, linestyle='--', label='Chief (evaded)')

    max_range = np.max(np.abs(np.vstack([pos_chief, pos_deputy]))) * 1.1
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    ax.set_xlabel('X (km)', fontsize=10)
    ax.set_ylabel('Y (km)', fontsize=10)
    ax.set_zlabel('Z (km)', fontsize=10)
    ax.set_title('Collision scenario: Chief vs deputy (rendezvous at T/2)\n' +
                 f'Chief {alt_chief} km, i={inc_deg}°',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True, alpha=0.2)

    if save_path is None and PATHS:
        plots_dir = PATHS.get('plots_dir')
        if plots_dir:
            os.makedirs(plots_dir, exist_ok=True)
            save_path = os.path.join(plots_dir, 'collision_scenario.png')
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"Collision scenario plot saved to: {save_path}")

    return fig, ax


def animate_adversary_transfer_rtn(
    positions_rtn_km,
    times,
    save_path,
    fps=20,
    max_frames=200,
    earth_radius_km=6378.137,
    chief_orbital_radius_km=None,
    earth_texture_path=None,
    earth_resolution='low',
):
    """
    Animate adversary transfer scenario in RTN frame (3D): Earth in -R direction (correct
    radius and center for RTN), chief at origin, deputy and constellation with trails.
    View from tangential–normal side (looking along R from -R so T–N plane faces viewer).

    In RTN, R points radially outward; chief is at origin. Earth center is along -R at
    distance chief_orbital_radius_km, with radius earth_radius_km (same as PNG).

    Parameters
    ----------
    positions_rtn_km : ndarray
        Shape (n_times, n_sats, 3). Index 0 = chief (always 0,0,0), 1 = adversary, 2+ = constellation.
        Columns are R, T, N (km).
    times : ndarray
        Time array in seconds.
    save_path : str
        Path to save GIF.
    fps, max_frames : int
        Animation frame rate and max frames.
    earth_radius_km : float
        Earth radius in km (same as adversary_transfer_scenario.png).
    chief_orbital_radius_km : float, optional
        Distance from Earth center to chief (km). If None, uses earth_radius_km + 550.
    earth_texture_path : str, optional
        Path to Earth texture image.
    earth_resolution : str
        'low', 'medium', 'high', 'ultra' for Earth mesh.
    """
    if chief_orbital_radius_km is None:
        chief_orbital_radius_km = earth_radius_km + 550.0
    n_times, n_sats, _ = positions_rtn_km.shape
    if n_times > max_frames:
        frame_indices = np.linspace(0, n_times - 1, max_frames, dtype=int)
    else:
        frame_indices = np.arange(n_times)
    n_frames = len(frame_indices)

    # RTN: Earth center is at -chief_orbital_radius along R (chief at origin)
    earth_center_R = -chief_orbital_radius_km
    view_radius_km = 1000.0

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    ax.set_xlim([-view_radius_km, view_radius_km])
    ax.set_ylim([-view_radius_km, view_radius_km])
    ax.set_zlim([-view_radius_km, view_radius_km])
    ax.set_xlabel('R (km)', fontsize=9)
    ax.set_ylabel('T (km)', fontsize=9)
    ax.set_zlabel('N (km)', fontsize=9)
    ax.view_init(elev=20, azim=0)
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True, alpha=0.2)

    # Earth: same radius and texture as PNG; center on -R (may be outside ±1000 km zoom)
    earth_surfaces = create_simple_earth(
        ax,
        radius_km=earth_radius_km,
        earth_texture_path=earth_texture_path,
        resolution=earth_resolution,
        center=(earth_center_R, 0.0, 0.0),
    )
    for s in earth_surfaces:
        s.set_zorder(0)

    marker_size = 18
    trail_length = min(40, max(15, n_frames // 3))
    chief_trail_line = None
    adv_trail_line = None
    chief_scatter = ax.scatter([0], [0], [0], c='#00FF00', s=marker_size, marker='o', label='Chief',
                               edgecolors='black', linewidths=0.5, depthshade=False, zorder=10)
    adversary_scatter = ax.scatter([], [], [], c='red', s=marker_size, marker='o', label='Deputy',
                                   alpha=0.95, edgecolors='black', linewidths=0.5, depthshade=False, zorder=10)
    constellation_scatter = ax.scatter([], [], [], c='orange', s=marker_size, marker='o', label='Constellation',
                                       alpha=0.85, depthshade=False, zorder=10)
    title_text = ax.set_title('', fontsize=10)
    ax.legend(loc='upper left', fontsize=9)

    def update(frame_num):
        nonlocal chief_trail_line, adv_trail_line
        idx = frame_indices[frame_num]
        t = times[idx] if times is not None else idx
        pos_adv = positions_rtn_km[idx, 1, :]
        adversary_scatter._offsets3d = (pos_adv[0:1], pos_adv[1:2], pos_adv[2:3])
        if n_sats > 2:
            pos_const = positions_rtn_km[idx, 2:, :]
            constellation_scatter._offsets3d = (pos_const[:, 0], pos_const[:, 1], pos_const[:, 2])
        if chief_trail_line is not None:
            chief_trail_line.remove()
        if adv_trail_line is not None:
            adv_trail_line.remove()
        trail_start = max(0, idx - trail_length)
        chief_trail_pts = positions_rtn_km[trail_start:idx + 1, 0, :]
        if len(chief_trail_pts) >= 1:
            chief_trail_line, = ax.plot(chief_trail_pts[:, 0], chief_trail_pts[:, 1], chief_trail_pts[:, 2],
                                        color='#00FF00', linewidth=1.2, alpha=0.7, zorder=10)
        trail_pts = positions_rtn_km[trail_start:idx + 1, 1, :]
        if len(trail_pts) >= 1:
            adv_trail_line, = ax.plot(trail_pts[:, 0], trail_pts[:, 1], trail_pts[:, 2],
                                      color='red', linewidth=1.2, alpha=0.7, zorder=10)
        if times is not None:
            title_text.set_text(f'RTN frame  t = {t:.0f} s ({t / 60:.1f} min)')
        else:
            title_text.set_text(f'RTN frame  Frame {frame_num + 1}/{n_frames}')
        return [adversary_scatter, constellation_scatter, title_text]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    if save_path:
        try:
            from tqdm import tqdm
            writer = PillowWriter(fps=fps)
            with tqdm(total=n_frames, desc="RTN GIF", unit="frame") as pbar:
                anim.save(save_path, writer=writer, progress_callback=lambda i, n: pbar.update(1))
        except ImportError:
            writer = PillowWriter(fps=fps)
            anim.save(save_path, writer=writer)
        plt.close(fig)
    return anim


def animate_adversary_transfer_eci(
    positions_eci_km,
    times,
    save_path,
    fps=20,
    max_frames=200,
    earth_radius_km=6378.137,
    view_radius_km=1000.0,
    earth_texture_path=None,
    earth_resolution='low',
    orbit_period_sec=None,
    camera_follows=True,
):
    """
    ECI version of adversary transfer animation: same style as RTN (chief green, deputy red,
    small markers, ±view_radius_km zoom centered on chief, trails for chief and deputy).
    Earth at origin. Axes and labels clearly visible. Optionally rotate camera around Earth
    to follow the satellites (one revolution per orbit).

    Parameters
    ----------
    positions_eci_km : ndarray
        Shape (n_times, n_sats, 3). Index 0 = chief, 1 = deputy, 2+ = constellation. ECI km.
    times : ndarray
        Time array in seconds.
    save_path : str
        Path to save GIF.
    fps, max_frames : int
        Animation frame rate and max frames.
    earth_radius_km : float
        Earth radius in km.
    view_radius_km : float
        Half-extent of view box (km); limits = chief ± view_radius_km each axis.
    earth_texture_path, earth_resolution
        Passed to create_simple_earth.
    orbit_period_sec : float, optional
        Orbit period in seconds; used to rotate camera one full turn per orbit. If None, from times.
    camera_follows : bool
        If True, rotate the camera around Earth (azim) so we follow the satellites; one revolution per orbit.
    """
    n_times, n_sats, _ = positions_eci_km.shape
    if n_times > max_frames:
        frame_indices = np.linspace(0, n_times - 1, max_frames, dtype=int)
    else:
        frame_indices = np.arange(n_times)
    n_frames = len(frame_indices)
    if orbit_period_sec is None and times is not None and len(times) > 1:
        orbit_period_sec = float(times[-1] - times[0])
    if orbit_period_sec is None or orbit_period_sec <= 0:
        orbit_period_sec = 5400.0

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    ax.view_init(elev=20, azim=0)
    ax.set_box_aspect([1, 1, 1])
    ax.grid(False)
    # Hide graph axes (labels, panes, ticks) for a clean space view
    ax.set_axis_off()

    earth_surfaces = create_simple_earth(
        ax,
        radius_km=earth_radius_km,
        earth_texture_path=earth_texture_path,
        resolution=earth_resolution,
        center=(0.0, 0.0, 0.0),
    )
    for s in earth_surfaces:
        s.set_zorder(0)

    marker_size = 18
    trail_length = min(40, max(15, n_frames // 3))
    chief_trail_line = None
    adv_trail_line = None
    chief_scatter = ax.scatter([], [], [], c='#00FF00', s=marker_size, marker='o', label='Chief',
                               edgecolors='black', linewidths=0.5, depthshade=False, zorder=10)
    adversary_scatter = ax.scatter([], [], [], c='red', s=marker_size, marker='o', label='Deputy',
                                   alpha=0.95, edgecolors='black', linewidths=0.5, depthshade=False, zorder=10)
    constellation_scatter = ax.scatter([], [], [], c='orange', s=marker_size, marker='o', label='Constellation',
                                       alpha=0.85, depthshade=False, zorder=10)
    title_text = ax.set_title('', fontsize=10)
    ax.legend(loc='upper left', fontsize=9)

    def update(frame_num):
        nonlocal chief_trail_line, adv_trail_line
        idx = frame_indices[frame_num]
        t = times[idx] if times is not None else idx
        center = positions_eci_km[idx, 0, :]
        ax.set_xlim([center[0] - view_radius_km, center[0] + view_radius_km])
        ax.set_ylim([center[1] - view_radius_km, center[1] + view_radius_km])
        ax.set_zlim([center[2] - view_radius_km, center[2] + view_radius_km])
        pos_chief = positions_eci_km[idx, 0, :]
        pos_adv = positions_eci_km[idx, 1, :]
        chief_scatter._offsets3d = (pos_chief[0:1], pos_chief[1:2], pos_chief[2:3])
        adversary_scatter._offsets3d = (pos_adv[0:1], pos_adv[1:2], pos_adv[2:3])
        if n_sats > 2:
            pos_const = positions_eci_km[idx, 2:, :]
            constellation_scatter._offsets3d = (pos_const[:, 0], pos_const[:, 1], pos_const[:, 2])
        if chief_trail_line is not None:
            chief_trail_line.remove()
        if adv_trail_line is not None:
            adv_trail_line.remove()
        trail_start = max(0, idx - trail_length)
        chief_trail_pts = positions_eci_km[trail_start:idx + 1, 0, :]
        if len(chief_trail_pts) >= 1:
            chief_trail_line, = ax.plot(chief_trail_pts[:, 0], chief_trail_pts[:, 1], chief_trail_pts[:, 2],
                                        color='#00FF00', linewidth=1.2, alpha=0.7, zorder=10)
        trail_pts = positions_eci_km[trail_start:idx + 1, 1, :]
        if len(trail_pts) >= 1:
            adv_trail_line, = ax.plot(trail_pts[:, 0], trail_pts[:, 1], trail_pts[:, 2],
                                      color='red', linewidth=1.2, alpha=0.7, zorder=10)
        if camera_follows:
            azim = (t / orbit_period_sec) * 360.0
            ax.view_init(elev=20, azim=azim)
        if times is not None:
            title_text.set_text(f'ECI  t = {t:.0f} s ({t / 60:.1f} min)')
        else:
            title_text.set_text(f'ECI  Frame {frame_num + 1}/{n_frames}')
        return [adversary_scatter, constellation_scatter, title_text]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    if save_path:
        try:
            from tqdm import tqdm
            writer = PillowWriter(fps=fps)
            with tqdm(total=n_frames, desc="ECI GIF", unit="frame") as pbar:
                anim.save(save_path, writer=writer, progress_callback=lambda i, n: pbar.update(1))
        except ImportError:
            writer = PillowWriter(fps=fps)
            anim.save(save_path, writer=writer)
        plt.close(fig)
    return anim


def plot_adversary_transfer_scenario(
    save_path=None,
    save_animation_path=None,
    save_animation_rtn_path=None,
    save_animation_eci_path=None,
    earth_radius_km=None,
    earth_texture_path=None,
    earth_resolution=None,
    fps=None,
    max_frames=None,
    show_evasion=True,
):
    """
    Visualize the collision-avoidance test: chief from params (SATELLITE, e=0),
    adversary from ADVERSARY_TRANSFER. Builds a constellation of 2 (chief + adversary),
    then uses the same pipeline as the rest of the sim: plot_orbits_static and
    animate_orbits_fast with Earth texture, VISUALIZATION fps/resolution.

    Parameters
    ----------
    save_path : str, optional
        Path to save the static figure. If None, uses PATHS['plots_dir']/adversary_transfer_scenario.png.
    save_animation_path : str, optional
        If set (e.g. .gif path), save an animation at VISUALIZATION fps.
    save_animation_rtn_path : str, optional
        If set, save RTN-frame animation (chief at origin, ±1000 km zoom).
    save_animation_eci_path : str, optional
        If set, save ECI-frame animation (same style as RTN: zoom ±1000 km, trails, axes visible).
    earth_radius_km, earth_texture_path, earth_resolution
        If None, taken from CONSTANTS and PATHS and VISUALIZATION.
    fps : int, optional
        Animation FPS. If None, uses VISUALIZATION['animation_fps'].
    max_frames : int, optional
        Max animation frames. If None, uses VISUALIZATION['animation_max_frames'].
    show_evasion : bool
        If True, compute collision_avoidance_delta_v and add evaded chief as 3rd trajectory.

    Returns
    -------
    fig, ax : matplotlib figure and axes (from static plot)
    """
    import sys
    try:
        from params import SATELLITE, CONSTANTS, CONSTELLATION, ADVERSARY_TRANSFER, PATHS, VISUALIZATION
    except ImportError:
        _sim_dir = os.path.dirname(os.path.abspath(__file__))
        if _sim_dir not in sys.path:
            sys.path.insert(0, _sim_dir)
        from params import SATELLITE, CONSTANTS, CONSTELLATION, ADVERSARY_TRANSFER, PATHS, VISUALIZATION

    project_root = PATHS.get('project_root', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from sat.control.rtn_to_eci_propagate import (
        propagate_chief,
        propagate_constellation,
        eci_to_relative_rtn,
        eci_to_rtn_basis,
        kepler_to_rv,
    )
    from sat.control.collision_avoidance import (
        apply_impulse_eci,
        collision_avoidance_delta_v,
        detect_near_miss,
        DEFAULT_RADIUS_KM,
    )

    mu_km = CONSTANTS['mu'] / 1e9
    earth_km = CONSTANTS['earth_radius_km']
    if earth_radius_km is None:
        earth_radius_km = earth_km
    if earth_texture_path is None:
        earth_texture_path = PATHS.get('earth_texture')
    if earth_resolution is None:
        earth_resolution = VISUALIZATION.get('earth_resolution', 'medium')
    if fps is None:
        fps = VISUALIZATION.get('animation_fps', 30)
    if max_frames is None:
        max_frames = VISUALIZATION.get('animation_max_frames', 200)

    # Chief from SATELLITE (e=0)
    a_c = earth_km + float(SATELLITE['altitude'])
    e_c = float(SATELLITE['eccentricity'])
    i_c = np.radians(float(SATELLITE['inclination']))
    om_c = np.radians(float(SATELLITE['omega']))
    Om_c = np.radians(float(SATELLITE['Omega']))
    M_c = np.radians(float(SATELLITE['M']))
    r_chief, v_chief = kepler_to_rv(a_c, e_c, i_c, om_c, Om_c, M_c, mu=mu_km)

    # Adversary from ADVERSARY_TRANSFER -> constellation_rtn for propagation
    a_a = float(ADVERSARY_TRANSFER['semi_major_axis'])
    e_a = float(ADVERSARY_TRANSFER['eccentricity'])
    i_a = np.radians(float(ADVERSARY_TRANSFER['inclination']))
    om_a = np.radians(float(ADVERSARY_TRANSFER['omega']))
    Om_a = np.radians(float(ADVERSARY_TRANSFER['Omega']))
    M_a = np.radians(float(ADVERSARY_TRANSFER['M']))
    r_adv, v_adv = kepler_to_rv(a_a, e_a, i_a, om_a, Om_a, M_a, mu=mu_km)

    constellation_rtn = np.asarray(
        eci_to_relative_rtn(r_chief, v_chief, r_adv, v_adv), dtype=float
    ).reshape(1, 6)

    # Build constellation (gray dots) from CONSTELLATION orbital bands
    np.random.seed(42)
    bands = CONSTELLATION.get('orbital_bands', [
        {"altitude": 550, "inclination": 53.0},
        {"altitude": 600, "inclination": 97.6},
        {"altitude": 500, "inclination": 30.0},
        {"altitude": 700, "inclination": 0.0},
    ])
    n_const_sats = min(120, max(30, len(bands) * 25))
    sats_per_band = max(1, n_const_sats // len(bands))
    constellation_rtn_list = []
    for band in bands:
        alt = float(band['altitude'])
        inc_deg = float(band['inclination'])
        a_b = earth_km + alt
        e_b = 0.0
        i_b = np.radians(inc_deg)
        om_b = 0.0
        for _ in range(sats_per_band):
            Om_b = np.random.uniform(0, 2 * np.pi)
            M_b = np.random.uniform(0, 2 * np.pi)
            r_b, v_b = kepler_to_rv(a_b, e_b, i_b, om_b, Om_b, M_b, mu=mu_km)
            rtn = eci_to_relative_rtn(r_chief, v_chief, r_b, v_b)
            constellation_rtn_list.append(rtn)
    constellation_rtn_extra = np.asarray(constellation_rtn_list, dtype=float)

    T_orbit = 2 * np.pi * np.sqrt((a_c ** 3) / mu_km)
    n_times = max(150, min(400, int(T_orbit / 15.0)))
    time_array = np.linspace(0, T_orbit, n_times)

    # Propagate chief, adversary, and constellation
    pos_chief_km = propagate_chief(r_chief, v_chief, time_array, backend='cpu')
    pos_adv_result = propagate_constellation(
        r_chief, v_chief, constellation_rtn, time_array, backend='cpu', return_velocity=False
    )
    if pos_adv_result.ndim == 2:
        pos_adv_km = pos_adv_result
    else:
        pos_adv_km = pos_adv_result[:, 0, :]

    pos_const_result = propagate_constellation(
        r_chief, v_chief, constellation_rtn_extra, time_array, backend='cpu', return_velocity=False
    )
    if pos_const_result.ndim == 2:
        pos_const_km = pos_const_result[:, np.newaxis, :]
    else:
        pos_const_km = pos_const_result
    n_const = pos_const_km.shape[1]

    # Test results for plot text: collision detected, miss distance, delta_v
    pos_adv_2 = pos_adv_km[:, np.newaxis, :] if pos_adv_km.ndim == 2 else pos_adv_km
    has_near, min_dist_km, t_idx_closest, _ = detect_near_miss(
        pos_chief_km, pos_adv_2, radius_km=DEFAULT_RADIUS_KM
    )
    delta_v_evaded = collision_avoidance_delta_v(
        r_chief, v_chief, constellation_rtn,
        backend='cpu', radius_km=DEFAULT_RADIUS_KM, n_times_orbit=80,
    ) if show_evasion else np.zeros(3)
    dv_norm = np.linalg.norm(delta_v_evaded)
    min_dist_after_km = None
    if show_evasion and dv_norm > 1e-6:
        r_ev, v_ev = apply_impulse_eci(r_chief, v_chief, delta_v_evaded)
        pos_evaded_km = propagate_chief(r_ev, v_ev, time_array, backend='cpu')
        _, min_dist_after_km, _, _ = detect_near_miss(
            pos_evaded_km, pos_adv_2, radius_km=DEFAULT_RADIUS_KM
        )
    else:
        pos_evaded_km = None

    # Full positions: chief (0), adversary (1), [evaded (2)], constellation (2+ or 3+)
    positions_m = np.zeros((n_times, 2 + n_const, 3), dtype=float)
    positions_m[:, 0, :] = pos_chief_km * 1000.0
    positions_m[:, 1, :] = pos_adv_km * 1000.0
    positions_m[:, 2:2 + n_const, :] = pos_const_km * 1000.0
    if pos_evaded_km is not None:
        positions_m = np.concatenate([
            positions_m[:, :2, :],
            (pos_evaded_km * 1000.0)[:, np.newaxis, :],
            positions_m[:, 2:, :],
        ], axis=1)

    plots_dir = PATHS.get('plots_dir')
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
    if save_path is None and plots_dir:
        save_path = os.path.join(plots_dir, 'adversary_transfer_scenario.png')

    treesat_color = VISUALIZATION.get('treesat_color', '#00FF00')
    adversary_color = 'red'

    # Static plot: draw full orbits (chief, adversary, evaded) so both are visible
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    elev = VISUALIZATION.get('view_elevation', 20)
    azim = VISUALIZATION.get('view_azimuth', 45)
    ax.view_init(elev=elev, azim=azim)
    create_simple_earth(ax, radius_km=earth_radius_km, earth_texture_path=earth_texture_path,
                        resolution=earth_resolution)

    ax.plot(pos_chief_km[:, 0], pos_chief_km[:, 1], pos_chief_km[:, 2],
            color=treesat_color, linewidth=1.5, alpha=0.9, label='Chief (TreeSat)')
    ax.plot(pos_adv_km[:, 0], pos_adv_km[:, 1], pos_adv_km[:, 2],
            color=adversary_color, linewidth=1.5, alpha=0.95, label='Adversary')
    ax.plot([pos_chief_km[-1, 0]], [pos_chief_km[-1, 1]], [pos_chief_km[-1, 2]],
            marker='o', markersize=8, color=treesat_color, markeredgecolor='none', linestyle='')
    ax.plot([pos_adv_km[-1, 0]], [pos_adv_km[-1, 1]], [pos_adv_km[-1, 2]],
            marker='o', markersize=8, color=adversary_color, alpha=0.95, linestyle='')

    # Constellation: gray dots at last timestep
    last_const_km = pos_const_km[-1]
    if last_const_km.size > 0:
        ax.scatter(last_const_km[:, 0], last_const_km[:, 1], last_const_km[:, 2],
                   c='gray', s=4, alpha=0.7, label='Constellation', linewidths=0)

    if pos_evaded_km is not None:
        ax.plot(pos_evaded_km[:, 0], pos_evaded_km[:, 1], pos_evaded_km[:, 2],
                color='cyan', linewidth=1.2, alpha=0.7, linestyle='--', label='Chief (evaded)')
        ax.plot([pos_evaded_km[-1, 0]], [pos_evaded_km[-1, 1]], [pos_evaded_km[-1, 2]],
                marker='s', markersize=5, color='cyan', alpha=0.8, linestyle='')

    all_pos_km = [pos_chief_km, pos_adv_km, pos_const_km.reshape(-1, 3)]
    if pos_evaded_km is not None:
        all_pos_km.append(pos_evaded_km)
    max_range = np.max(np.abs(np.vstack(all_pos_km))) * 1.1
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    ax.set_xlabel('X (km)', fontsize=10)
    ax.set_ylabel('Y (km)', fontsize=10)
    ax.set_zlabel('Z (km)', fontsize=10)
    title = (
        'Adversary transfer: Chief (green), Adversary (red), Constellation (gray)\n'
        f'Chief a={a_c:.0f} km, Adversary a={a_a:.0f} km e={e_a:.4f}'
    )
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True, alpha=0.2)

    # Results text box: collision detected, miss distance, delta_v
    results_lines = [
        'Test results',
        '----------------------------',
        f'Collision (near miss): {"Yes" if has_near else "No"}',
        f'Min separation: {min_dist_km:.4f} km',
        f'|delta_v|: {dv_norm:.6f} km/s',
        f'delta_v: [{delta_v_evaded[0]:.5f}, {delta_v_evaded[1]:.5f}, {delta_v_evaded[2]:.5f}]',
    ]
    if min_dist_after_km is not None:
        results_lines.append(f'Min sep. after evasion: {min_dist_after_km:.4f} km')
    results_text = '\n'.join(results_lines)
    ax.text2D(0.02, 0.98, results_text, transform=ax.transAxes, fontsize=9,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"Adversary transfer scenario plot saved to: {save_path}")

    if save_animation_path:
        constellation_start = 3 if pos_evaded_km is not None else 2
        animate_orbits_fast(
            positions_m,
            times=time_array,
            save_path=save_animation_path,
            fps=fps,
            max_frames=max_frames,
            max_satellites=None,
            earth_radius_km=earth_radius_km,
            earth_texture_path=earth_texture_path,
            earth_resolution=earth_resolution,
            other_sat_color=adversary_color,
            adversary_index=1,
            constellation_start_index=constellation_start,
            constellation_color='gray',
        )
        print(f"Adversary transfer animation saved to: {save_animation_path}")

    if save_animation_rtn_path:
        # Chief velocity by finite difference (km/s) for RTN basis at each time
        dt = float(time_array[1] - time_array[0]) if len(time_array) > 1 else 1.0
        v_chief_km = np.zeros_like(pos_chief_km)
        v_chief_km[0] = (pos_chief_km[1] - pos_chief_km[0]) / dt
        v_chief_km[-1] = (pos_chief_km[-1] - pos_chief_km[-2]) / dt
        v_chief_km[1:-1] = (pos_chief_km[2:] - pos_chief_km[:-2]) / (2.0 * dt)
        n_t = len(time_array)
        positions_rtn_km = np.zeros((n_t, 1 + 1 + n_const, 3), dtype=float)
        positions_rtn_km[:, 0, :] = 0.0
        for t in range(n_t):
            basis = eci_to_rtn_basis(pos_chief_km[t], v_chief_km[t])
            dr_adv = pos_adv_km[t] - pos_chief_km[t]
            positions_rtn_km[t, 1, :] = basis @ dr_adv
            for k in range(n_const):
                dr_k = pos_const_km[t, k, :] - pos_chief_km[t]
                positions_rtn_km[t, 2 + k, :] = basis @ dr_k
        animate_adversary_transfer_rtn(
            positions_rtn_km,
            times=time_array,
            save_path=save_animation_rtn_path,
            fps=fps,
            max_frames=max_frames,
            earth_radius_km=earth_radius_km,
            chief_orbital_radius_km=a_c,
            earth_texture_path=earth_texture_path,
            earth_resolution=earth_resolution,
        )
        print(f"Adversary transfer RTN animation saved to: {save_animation_rtn_path}")

    if save_animation_eci_path:
        positions_eci_km = np.zeros((n_times, 1 + 1 + n_const, 3), dtype=float)
        positions_eci_km[:, 0, :] = pos_chief_km
        positions_eci_km[:, 1, :] = pos_adv_km
        positions_eci_km[:, 2:, :] = pos_const_km
        animate_adversary_transfer_eci(
            positions_eci_km,
            times=time_array,
            save_path=save_animation_eci_path,
            fps=fps,
            max_frames=max_frames,
            earth_radius_km=earth_radius_km,
            view_radius_km=1000.0,
            earth_texture_path=earth_texture_path,
            earth_resolution=earth_resolution,
            orbit_period_sec=T_orbit,
            camera_follows=True,
        )
        print(f"Adversary transfer ECI animation saved to: {save_animation_eci_path}")

    return fig, ax


def _compute_maneuver_scenario_trajectories(
    n_orbits=5,
    n_per_orbit=200,
    adv_maneuver_delay_steps=10,
    dv_tangential_ms=300.0,
    dv_normal_ms=1500.0,
    inclination_deg=-45.0,
):
    """
    Compute chief and adversary trajectories over n_orbits with periodic maneuvers.

    Chief: at the start of each orbit applies delta-v alternating 300 m/s tangential
    and 1500 m/s normal (hardcoded). Adversary: same maneuver delayed by 10 steps.
    For this scenario only, chief and deputy initial inclinations are hardcoded to -45 deg.

    Returns
    -------
    time_array : ndarray (n_total,) seconds
    pos_chief_km : ndarray (n_total, 3) ECI km
    pos_adv_km : ndarray (n_total, 3) ECI km
    maneuver_indices : list of int (chief maneuver step indices)
    maneuver_dv_eci_km : list of (3,) arrays (delta-v in ECI km/s at each maneuver)
    """
    import sys
    try:
        from params import SATELLITE, CONSTANTS, ADVERSARY_TRANSFER
    except ImportError:
        _sim_dir = os.path.dirname(os.path.abspath(__file__))
        if _sim_dir not in sys.path:
            sys.path.insert(0, _sim_dir)
        from params import SATELLITE, CONSTANTS, ADVERSARY_TRANSFER

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from sat.control.rtn_to_eci_propagate import (
        propagate_chief,
        eci_to_rtn_basis,
        kepler_to_rv,
    )

    mu_km = CONSTANTS['mu'] / 1e9
    earth_km = CONSTANTS['earth_radius_km']
    inc_rad = np.radians(inclination_deg)
    a_c = earth_km + float(SATELLITE['altitude'])
    # Maneuver scenario: hardcoded inclination -45 deg for chief and deputy
    r_chief, v_chief = kepler_to_rv(
        a_c, float(SATELLITE['eccentricity']),
        inc_rad,
        np.radians(float(SATELLITE['omega'])), np.radians(float(SATELLITE['Omega'])),
        np.radians(float(SATELLITE['M'])), mu=mu_km,
    )
    r_adv, v_adv = kepler_to_rv(
        float(ADVERSARY_TRANSFER['semi_major_axis']), float(ADVERSARY_TRANSFER['eccentricity']),
        inc_rad,
        np.radians(float(ADVERSARY_TRANSFER['omega'])), np.radians(float(ADVERSARY_TRANSFER['Omega'])),
        np.radians(float(ADVERSARY_TRANSFER['M'])), mu=mu_km,
    )

    T_orbit = 2 * np.pi * np.sqrt((a_c ** 3) / mu_km)
    n_total = n_orbits * n_per_orbit
    time_full = np.linspace(0, n_orbits * T_orbit, n_total, endpoint=False)
    dt = float(time_full[1] - time_full[0]) if n_total > 1 else T_orbit / n_per_orbit

    # Delta-v magnitudes in km/s (300 m/s tangential, 1500 m/s normal)
    dv_tang_km = dv_tangential_ms / 1000.0
    dv_norm_km = dv_normal_ms / 1000.0

    def dv_eci_tangential(r, v):
        basis = eci_to_rtn_basis(r, v)
        return (basis.T @ np.array([0.0, dv_tang_km, 0.0])).reshape(3)

    def dv_eci_normal(r, v):
        basis = eci_to_rtn_basis(r, v)
        return (basis.T @ np.array([0.0, 0.0, dv_norm_km])).reshape(3)

    pos_chief_km = np.zeros((n_total, 3), dtype=float)
    r_c, v_c = r_chief.copy(), v_chief.copy()
    maneuver_indices = []
    maneuver_dv_eci_km = []

    for orbit in range(n_orbits):
        i_start = orbit * n_per_orbit
        t_seg = time_full[i_start:i_start + n_per_orbit] - time_full[i_start]
        if orbit % 2 == 0:
            dv = dv_eci_tangential(r_c, v_c)
        else:
            dv = dv_eci_normal(r_c, v_c)
        maneuver_indices.append(i_start)
        maneuver_dv_eci_km.append(dv.copy())
        v_c = v_c + dv
        pos_seg = propagate_chief(r_c, v_c, t_seg)
        pos_chief_km[i_start:i_start + n_per_orbit] = pos_seg
        r_c = pos_seg[-1]
        if len(pos_seg) >= 2:
            v_c = (pos_seg[-1] - pos_seg[-2]) / dt
        else:
            v_c = v_c  # keep

    # Adversary: same maneuvers delayed by adv_maneuver_delay_steps
    pos_adv_km = np.zeros((n_total, 3), dtype=float)
    r_a, v_a = r_adv.copy(), v_adv.copy()
    delay = adv_maneuver_delay_steps

    # Segment boundaries for adversary: [0, delay), [delay, n_per_orbit+delay), [n_per_orbit+delay, 2*n_per_orbit+delay), ...
    i_starts_adv = [0, delay]
    for orbit in range(1, n_orbits):
        i_starts_adv.append(orbit * n_per_orbit + delay)
    i_starts_adv.append(n_total)  # end

    for seg in range(len(i_starts_adv) - 1):
        i_start = i_starts_adv[seg]
        i_end = i_starts_adv[seg + 1]
        n_seg = i_end - i_start
        if n_seg <= 0:
            continue
        t_seg = time_full[i_start:i_end] - time_full[i_start]
        if seg == 0:
            # No maneuver for first segment (0 to delay)
            pos_seg = propagate_chief(r_a, v_a, t_seg)
        else:
            # Apply same maneuver chief applied at orbit (seg-1): tangential for even orbit, normal for odd
            orbit_idx = seg - 1
            if orbit_idx % 2 == 0:
                dv = dv_eci_tangential(r_a, v_a)
            else:
                dv = dv_eci_normal(r_a, v_a)
            v_a = v_a + dv
            pos_seg = propagate_chief(r_a, v_a, t_seg)
        pos_adv_km[i_start:i_end] = pos_seg
        r_a = pos_seg[-1]
        if n_seg >= 2:
            v_a = (pos_seg[-1] - pos_seg[-2]) / dt
        else:
            v_a = v_a

    return time_full, pos_chief_km, pos_adv_km, maneuver_indices, maneuver_dv_eci_km


def plot_maneuver_scenario_2d(save_path=None):
    """2D ECI X-Y plot of 5-orbit maneuver scenario: full chief and adversary paths."""
    import sys
    try:
        from params import PATHS
    except ImportError:
        _sim_dir = os.path.dirname(os.path.abspath(__file__))
        if _sim_dir not in sys.path:
            sys.path.insert(0, _sim_dir)
        from params import PATHS

    time_array, pos_chief_km, pos_adv_km, maneuver_indices, maneuver_dv_eci_km = _compute_maneuver_scenario_trajectories()

    def trail_lc_2d(xy, color_name, lw=1.5):
        pts = xy.reshape(-1, 1, 2)
        segs = np.hstack([pts[:-1], pts[1:]])
        alphas = np.linspace(0.15, 1.0, len(segs))
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'fade', [mcolors.to_rgba(color_name, 0.0), mcolors.to_rgba(color_name, 1.0)]
        )
        return LineCollection(segs, array=alphas, cmap=cmap, linewidth=lw)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_aspect('equal')
    # Maneuver scenario: adversary (trajectory 2) = green labeled Chief; chief (trajectory 1) = red labeled Adversary
    ax.add_collection(trail_lc_2d(pos_chief_km[:, :2], 'red'))
    ax.add_collection(trail_lc_2d(pos_adv_km[:, :2], 'green'))
    ax.scatter([pos_chief_km[0, 0]], [pos_chief_km[0, 1]], c='red', s=80, marker='o', edgecolors='black', zorder=5, label='Adversary start')
    ax.scatter([pos_adv_km[0, 0]], [pos_adv_km[0, 1]], c='green', s=80, marker='o', edgecolors='black', zorder=5, label='Chief start')
    ax.scatter([pos_chief_km[-1, 0]], [pos_chief_km[-1, 1]], c='red', s=80, marker='s', edgecolors='black', zorder=5)
    ax.scatter([pos_adv_km[-1, 0]], [pos_adv_km[-1, 1]], c='green', s=80, marker='s', edgecolors='black', zorder=5)

    stack = np.vstack([pos_chief_km[:, :2], pos_adv_km[:, :2]])
    margin = 80.0
    x_min, x_max = stack[:, 0].min(), stack[:, 0].max()
    y_min, y_max = stack[:, 1].min(), stack[:, 1].max()
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_xlabel('X (ECI), km', fontsize=11)
    ax.set_ylabel('Y (ECI), km', fontsize=11)
    ax.set_title('Maneuver scenario (5 orbits): Chief 300 m/s tangential / 1500 m/s normal alternate; Adversary same, 10 steps delayed (incl -45°)', fontsize=10)
    ax.legend(handles=[Line2D([0], [0], color='red', lw=2, label='Adversary'), Line2D([0], [0], color='green', lw=2, label='Chief'),
                       Line2D([0], [0], linestyle='none', marker='o', markersize=8, markerfacecolor='red', markeredgecolor='black', label='Adversary start'),
                       Line2D([0], [0], linestyle='none', marker='o', markersize=8, markerfacecolor='green', markeredgecolor='black', label='Chief start')],
              loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    if save_path is None and PATHS.get('plots_dir'):
        os.makedirs(PATHS['plots_dir'], exist_ok=True)
        save_path = os.path.join(PATHS['plots_dir'], 'maneuver_scenario_2d.png')
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"Maneuver scenario 2D plot saved to: {save_path}")
    return fig, ax


def animate_maneuver_scenario_2d(save_path=None, max_frames=150, fps=15, pause_seconds_per_orbit=3):
    """2D GIF: permanent trail with alpha decay, pause after each orbit."""
    import sys
    try:
        from params import PATHS
    except ImportError:
        _sim_dir = os.path.dirname(os.path.abspath(__file__))
        if _sim_dir not in sys.path:
            sys.path.insert(0, _sim_dir)
        from params import PATHS

    time_array, pos_chief_km, pos_adv_km, maneuver_indices, maneuver_dv_eci_km = _compute_maneuver_scenario_trajectories()
    n_total = len(time_array)
    n_orbits = 5
    n_per_orbit = n_total // n_orbits
    pause_frames = int(pause_seconds_per_orbit * fps)

    if n_total > max_frames:
        frame_indices = np.linspace(0, n_total - 1, max_frames, dtype=int)
    else:
        frame_indices = np.arange(n_total)
    # Build frame list: after we first reach each orbit end, append 3s pause
    orbit_ends = [orbit * n_per_orbit - 1 for orbit in range(1, n_orbits + 1)]
    frame_to_data = []
    paused_for = set()
    for idx in frame_indices:
        frame_to_data.append(idx)
        for oi, oe in enumerate(orbit_ends):
            if oi in paused_for:
                continue
            if idx >= oe:
                frame_to_data.extend([oe] * pause_frames)
                paused_for.add(oi)
                break
    n_frames_total = len(frame_to_data)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_aspect('equal')
    # Trail as LineCollection (updated each frame; alpha decay)
    lc_chief = LineCollection([], linewidth=1.2)
    lc_adv = LineCollection([], linewidth=1.2)
    ax.add_collection(lc_chief)
    ax.add_collection(lc_adv)
    # Maneuver scenario: chief trajectory = red/Adversary, adversary trajectory = green/Chief
    chief_scatter = ax.scatter([], [], c='red', s=100, marker='o', edgecolors='black', zorder=10)
    adv_scatter = ax.scatter([], [], c='green', s=100, marker='o', edgecolors='black', zorder=10)
    title_text = ax.set_title('', fontsize=11)

    stack = np.vstack([pos_chief_km[:, :2], pos_adv_km[:, :2]])
    margin = 80.0
    x_min, x_max = stack[:, 0].min(), stack[:, 0].max()
    y_min, y_max = stack[:, 1].min(), stack[:, 1].max()
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_xlabel('X (ECI), km', fontsize=10)
    ax.set_ylabel('Y (ECI), km', fontsize=10)
    ax.legend([Line2D([0], [0], color='red', lw=2), Line2D([0], [0], color='green', lw=2)], ['Adversary', 'Chief'], loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    cmap_red = mcolors.LinearSegmentedColormap.from_list('fr', [mcolors.to_rgba('red', 0.0), mcolors.to_rgba('red', 1.0)])
    cmap_green = mcolors.LinearSegmentedColormap.from_list('fg', [mcolors.to_rgba('green', 0.0), mcolors.to_rgba('green', 1.0)])

    def update(i):
        idx = frame_to_data[i]
        t = time_array[idx]
        end = idx + 1
        # Trail with alpha decay (old -> transparent, new -> opaque)
        if end >= 2:
            pts_c = pos_chief_km[:end, :2].reshape(-1, 1, 2)
            segs_c = np.hstack([pts_c[:-1], pts_c[1:]])
            alphas_c = np.linspace(0.15, 1.0, len(segs_c))
            lc_chief.set_segments(segs_c)
            lc_chief.set_array(alphas_c)
            lc_chief.set_cmap(cmap_red)
            pts_a = pos_adv_km[:end, :2].reshape(-1, 1, 2)
            segs_a = np.hstack([pts_a[:-1], pts_a[1:]])
            alphas_a = np.linspace(0.15, 1.0, len(segs_a))
            lc_adv.set_segments(segs_a)
            lc_adv.set_array(alphas_a)
            lc_adv.set_cmap(cmap_green)
        else:
            lc_chief.set_segments([])
            lc_adv.set_segments([])
        chief_scatter.set_offsets(pos_chief_km[idx, :2].reshape(1, 2))
        adv_scatter.set_offsets(pos_adv_km[idx, :2].reshape(1, 2))
        orbit_num = min(5, 1 + int(5 * idx / n_total))
        title_text.set_text(f'Maneuver scenario  t = {t:.0f} s ({t / 60:.1f} min)  orbit {orbit_num}/5')
        return [lc_chief, lc_adv, chief_scatter, adv_scatter, title_text]

    anim = FuncAnimation(fig, update, frames=n_frames_total, interval=1000 / fps, blit=False)
    if save_path is None and PATHS.get('plots_dir'):
        os.makedirs(PATHS['plots_dir'], exist_ok=True)
        save_path = os.path.join(PATHS['plots_dir'], 'maneuver_scenario_2d.gif')
    if save_path:
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=100)
        plt.close(fig)
        print(f"Maneuver scenario 2D animation saved to: {save_path}")
    return anim


def plot_maneuver_scenario_3d(
    save_path=None,
    earth_radius_km=6378.137,
    earth_texture_path=None,
    earth_resolution='low',
):
    """3D ECI static plot: full chief and adversary paths, fixed view (no zoom/rotate)."""
    import sys
    try:
        from params import PATHS
    except ImportError:
        _sim_dir = os.path.dirname(os.path.abspath(__file__))
        if _sim_dir not in sys.path:
            sys.path.insert(0, _sim_dir)
        from params import PATHS

    time_array, pos_chief_km, pos_adv_km, maneuver_indices, maneuver_dv_eci_km = _compute_maneuver_scenario_trajectories()

    def trail_3d_segments(xyz, color, seg_len=25):
        """Draw 3D path as segments with alpha decay (old -> transparent)."""
        n = len(xyz)
        for i in range(0, max(1, n - 1), seg_len):
            j = min(i + seg_len + 1, n)
            alpha = 0.15 + 0.85 * (i / max(1, n - 1))
            ax.plot(xyz[i:j, 0], xyz[i:j, 1], xyz[i:j, 2], color=color, lw=1.2, alpha=alpha)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    create_simple_earth(ax, radius_km=earth_radius_km, earth_texture_path=earth_texture_path, resolution=earth_resolution)
    # Maneuver scenario: chief traj = red/Adversary, adversary traj = green/Chief
    trail_3d_segments(pos_chief_km, 'red')
    trail_3d_segments(pos_adv_km, '#00FF00')
    ax.plot([pos_chief_km[-1, 0]], [pos_chief_km[-1, 1]], [pos_chief_km[-1, 2]], marker='o', markersize=8, color='red', linestyle='')
    ax.plot([pos_adv_km[-1, 0]], [pos_adv_km[-1, 1]], [pos_adv_km[-1, 2]], marker='o', markersize=8, color='#00FF00', linestyle='')

    all_km = np.vstack([pos_chief_km, pos_adv_km])
    max_range = np.max(np.abs(all_km)) * 1.1
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    ax.set_xlabel('X (km)', fontsize=10)
    ax.set_ylabel('Y (km)', fontsize=10)
    ax.set_zlabel('Z (km)', fontsize=10)
    ax.set_title('Maneuver scenario (5 orbits): Chief 300 / 1500 m/s T/N, incl -45° (ECI)', fontsize=11)
    ax.view_init(elev=20, azim=45)
    ax.set_box_aspect([1, 1, 1])
    ax.legend(handles=[Line2D([0], [0], color='red', lw=2, label='Adversary'), Line2D([0], [0], color='#00FF00', lw=2, label='Chief')], loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.2)

    if save_path is None and PATHS.get('plots_dir'):
        os.makedirs(PATHS['plots_dir'], exist_ok=True)
        save_path = os.path.join(PATHS['plots_dir'], 'maneuver_scenario_3d.png')
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"Maneuver scenario 3D plot saved to: {save_path}")
    return fig, ax


def animate_maneuver_scenario_3d(
    save_path=None,
    max_frames=200,
    fps=20,
    earth_radius_km=6378.137,
    earth_texture_path=None,
    earth_resolution='low',
    pause_seconds_per_orbit=3,
):
    """3D ECI GIF: permanent trail with alpha decay, pause after each orbit; fixed camera."""
    import sys
    try:
        from params import PATHS
    except ImportError:
        _sim_dir = os.path.dirname(os.path.abspath(__file__))
        if _sim_dir not in sys.path:
            sys.path.insert(0, _sim_dir)
        from params import PATHS

    time_array, pos_chief_km, pos_adv_km, maneuver_indices, maneuver_dv_eci_km = _compute_maneuver_scenario_trajectories()
    n_total = len(time_array)
    n_orbits = 5
    n_per_orbit = n_total // n_orbits
    pause_frames = int(pause_seconds_per_orbit * fps)

    if n_total > max_frames:
        frame_indices = np.linspace(0, n_total - 1, max_frames, dtype=int)
    else:
        frame_indices = np.arange(n_total)
    orbit_ends = [o * n_per_orbit - 1 for o in range(1, n_orbits + 1)]
    frame_to_data = []
    paused_for = set()
    for idx in frame_indices:
        frame_to_data.append(idx)
        for oi, oe in enumerate(orbit_ends):
            if oi in paused_for:
                continue
            if idx >= oe:
                frame_to_data.extend([oe] * pause_frames)
                paused_for.add(oi)
                break
    n_frames_total = len(frame_to_data)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    ax.view_init(elev=20, azim=45)
    ax.set_box_aspect([1, 1, 1])
    ax.grid(False)
    ax.set_axis_off()

    earth_surfaces = create_simple_earth(ax, radius_km=earth_radius_km, earth_texture_path=earth_texture_path, resolution=earth_resolution, center=(0, 0, 0))
    for s in earth_surfaces:
        s.set_zorder(0)

    # Trail: segment-based with alpha decay (we update segment lines each frame)
    seg_len = 20
    max_segs = (n_total // seg_len) + 1
    trail_chief_lines = []
    trail_adv_lines = []
    # Maneuver scenario: chief traj = red/Adversary, adversary traj = green/Chief
    for _ in range(max_segs):
        lc, = ax.plot([], [], [], color='red', linewidth=1.0, zorder=10)
        la, = ax.plot([], [], [], color='#00FF00', linewidth=1.0, zorder=10)
        trail_chief_lines.append(lc)
        trail_adv_lines.append(la)
    chief_scatter = ax.scatter([], [], [], c='red', s=60, marker='o', edgecolors='black', depthshade=False, zorder=10)
    adv_scatter = ax.scatter([], [], [], c='#00FF00', s=60, marker='o', edgecolors='black', depthshade=False, zorder=10)
    title_text = ax.set_title('', fontsize=10)

    leg_handles = [Line2D([0], [0], color='red', lw=2), Line2D([0], [0], color='#00FF00', lw=2)]
    ax.legend(handles=leg_handles, labels=['Adversary', 'Chief'], loc='upper left', fontsize=9)

    all_km = np.vstack([pos_chief_km, pos_adv_km])
    max_range = np.max(np.abs(all_km)) * 1.1
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    def update(frame_num):
        idx = frame_to_data[frame_num]
        t = time_array[idx]
        end = idx + 1
        # Trail with alpha decay: draw segments
        for seg_idx, line in enumerate(trail_chief_lines):
            i, j = seg_idx * seg_len, min((seg_idx + 1) * seg_len + 1, end)
            if j > i and i < end:
                alpha = 0.15 + 0.85 * (i / max(1, end - 1))
                line.set_data(pos_chief_km[i:j, 0], pos_chief_km[i:j, 1])
                line.set_3d_properties(pos_chief_km[i:j, 2])
                line.set_alpha(alpha)
                line.set_visible(True)
            else:
                line.set_visible(False)
        for seg_idx, line in enumerate(trail_adv_lines):
            i, j = seg_idx * seg_len, min((seg_idx + 1) * seg_len + 1, end)
            if j > i and i < end:
                alpha = 0.15 + 0.85 * (i / max(1, end - 1))
                line.set_data(pos_adv_km[i:j, 0], pos_adv_km[i:j, 1])
                line.set_3d_properties(pos_adv_km[i:j, 2])
                line.set_alpha(alpha)
                line.set_visible(True)
            else:
                line.set_visible(False)
        chief_scatter._offsets3d = (pos_chief_km[idx, 0:1], pos_chief_km[idx, 1:2], pos_chief_km[idx, 2:3])
        adv_scatter._offsets3d = (pos_adv_km[idx, 0:1], pos_adv_km[idx, 1:2], pos_adv_km[idx, 2:3])
        title_text.set_text(f'ECI  t = {t:.0f} s ({t / 60:.1f} min)')
        return trail_chief_lines + trail_adv_lines + [chief_scatter, adv_scatter, title_text]

    anim = FuncAnimation(fig, update, frames=n_frames_total, interval=1000 / fps, blit=False)
    if save_path is None and PATHS.get('plots_dir'):
        os.makedirs(PATHS['plots_dir'], exist_ok=True)
        save_path = os.path.join(PATHS['plots_dir'], 'maneuver_scenario_3d.gif')
    if save_path:
        try:
            from tqdm import tqdm
            writer = PillowWriter(fps=fps)
            with tqdm(total=n_frames_total, desc="Maneuver 3D GIF", unit="frame") as pbar:
                anim.save(save_path, writer=writer, progress_callback=lambda i, n: pbar.update(1))
        except ImportError:
            writer = PillowWriter(fps=fps)
            anim.save(save_path, writer=writer)
        plt.close(fig)
        print(f"Maneuver scenario 3D animation saved to: {save_path}")
    return anim


def plot_evasion_2d(
    save_path=None,
    radius_km=None,
    earth_radius_km=None,
    show_evasion=True,
):
    """
    2D plot (chief orbital plane) of collision avoidance: original chief, adversary,
    min-miss constraint violation, delta-v application, and evaded chief trajectory.

    All trajectories are projected onto the chief's R-T plane (radial vs along-track
    at t=0). Overlays original and evaded chief so you see the maneuver effect.

    Parameters
    ----------
    save_path : str, optional
        Path to save the figure. If None, uses PATHS['plots_dir']/evasion_2d.png.
    radius_km : float, optional
        Near-miss radius (km). If None, uses collision_avoidance.DEFAULT_RADIUS_KM.
    earth_radius_km : float, optional
        Earth radius for the circle. If None, from CONSTANTS.
    show_evasion : bool
        If True, compute and plot evaded trajectory; otherwise only original + adversary.

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    import sys
    try:
        from params import SATELLITE, CONSTANTS, ADVERSARY_TRANSFER, PATHS
    except ImportError:
        _sim_dir = os.path.dirname(os.path.abspath(__file__))
        if _sim_dir not in sys.path:
            sys.path.insert(0, _sim_dir)
        from params import SATELLITE, CONSTANTS, ADVERSARY_TRANSFER, PATHS

    project_root = PATHS.get('project_root', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from sat.control.rtn_to_eci_propagate import (
        propagate_chief,
        propagate_constellation,
        eci_to_relative_rtn,
        kepler_to_rv,
    )
    from sat.control.collision_avoidance import (
        apply_impulse_eci,
        collision_avoidance_delta_v,
        detect_near_miss,
        DEFAULT_RADIUS_KM,
    )

    if radius_km is None:
        radius_km = DEFAULT_RADIUS_KM
    mu_km = CONSTANTS['mu'] / 1e9
    earth_km = CONSTANTS['earth_radius_km']
    if earth_radius_km is None:
        earth_radius_km = earth_km

    # Chief and adversary from params
    a_c = earth_km + float(SATELLITE['altitude'])
    e_c = float(SATELLITE['eccentricity'])
    i_c = np.radians(float(SATELLITE['inclination']))
    om_c = np.radians(float(SATELLITE['omega']))
    Om_c = np.radians(float(SATELLITE['Omega']))
    M_c = np.radians(float(SATELLITE['M']))
    r_chief, v_chief = kepler_to_rv(a_c, e_c, i_c, om_c, Om_c, M_c, mu=mu_km)

    a_a = float(ADVERSARY_TRANSFER['semi_major_axis'])
    e_a = float(ADVERSARY_TRANSFER['eccentricity'])
    i_a = np.radians(float(ADVERSARY_TRANSFER['inclination']))
    om_a = np.radians(float(ADVERSARY_TRANSFER['omega']))
    Om_a = np.radians(float(ADVERSARY_TRANSFER['Omega']))
    M_a = np.radians(float(ADVERSARY_TRANSFER['M']))
    r_adv, v_adv = kepler_to_rv(a_a, e_a, i_a, om_a, Om_a, M_a, mu=mu_km)

    constellation_rtn = np.asarray(
        eci_to_relative_rtn(r_chief, v_chief, r_adv, v_adv), dtype=float
    ).reshape(1, 6)

    T_orbit = 2 * np.pi * np.sqrt((a_c ** 3) / mu_km)
    n_times = max(200, int(T_orbit / 10.0))
    time_array = np.linspace(0, T_orbit, n_times)

    pos_chief = propagate_chief(r_chief, v_chief, time_array, backend='cpu')
    pos_adv = propagate_constellation(
        r_chief, v_chief, constellation_rtn, time_array, backend='cpu', return_velocity=False
    )
    if pos_adv.ndim == 2:
        pos_adv = pos_adv[:, np.newaxis, :]
    pos_adv = pos_adv[:, 0, :]

    has_near, min_dist_km, t_idx_closest, _ = detect_near_miss(
        pos_chief, pos_adv[:, np.newaxis, :], radius_km=radius_km
    )
    n_times_orbit = max(200, n_times)
    delta_v = collision_avoidance_delta_v(
        r_chief, v_chief, constellation_rtn,
        backend='cpu', radius_km=radius_km, n_times_orbit=n_times_orbit,
    )
    dv_norm = np.linalg.norm(delta_v)
    pos_evaded = None
    min_dist_after_km = None
    if show_evasion and dv_norm > 1e-9:
        r_ev, v_ev = apply_impulse_eci(r_chief, v_chief, delta_v)
        pos_evaded = propagate_chief(r_ev, v_ev, time_array, backend='cpu')
        _, min_dist_after_km, _, _ = detect_near_miss(
            pos_evaded, pos_adv[:, np.newaxis, :], radius_km=radius_km
        )

    # 2D plot in ECI: X and Y (equatorial plane). Scale from data so orbit change is visible.
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_aspect('equal')

    # Chief (original)
    ax.plot(pos_chief[:, 0], pos_chief[:, 1], color='green', lw=2, alpha=0.9, label='Chief (original)')
    # Delta-v at t=0
    ax.scatter([r_chief[0]], [r_chief[1]], c='green', s=120, marker='o', edgecolors='black', linewidths=2,
               label='$\\Delta v$ applied (t=0)', zorder=5)
    ax.annotate('$\\Delta v$', (r_chief[0], r_chief[1]), textcoords='offset points', xytext=(8, 8),
                fontsize=11, fontweight='bold')

    # Adversary
    ax.plot(pos_adv[:, 0], pos_adv[:, 1], color='red', lw=2, alpha=0.9, label='Adversary')

    # Min-miss point (markers only; caption is below the figure)
    if t_idx_closest >= 0:
        ax.scatter([pos_chief[t_idx_closest, 0]], [pos_chief[t_idx_closest, 1]], c='green', s=80,
                   marker='s', edgecolors='black', zorder=5)
        ax.scatter([pos_adv[t_idx_closest, 0]], [pos_adv[t_idx_closest, 1]], c='red', s=80,
                   marker='s', edgecolors='black', zorder=5)

    # Evaded chief
    if pos_evaded is not None:
        ax.plot(pos_evaded[:, 0], pos_evaded[:, 1], color='cyan', lw=2, linestyle='--', alpha=0.9,
                label='Chief (evaded)')

    # Scale: bounding box of all trajectories + margin so orbit change is visible
    all_xy = [pos_chief[:, :2], pos_adv[:, :2]]
    if pos_evaded is not None:
        all_xy.append(pos_evaded[:, :2])
    stack = np.vstack(all_xy)
    x_min, x_max = stack[:, 0].min(), stack[:, 0].max()
    y_min, y_max = stack[:, 1].min(), stack[:, 1].max()
    margin = 80.0  # km
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)

    ax.set_xlabel('X (ECI), km', fontsize=11)
    ax.set_ylabel('Y (ECI), km', fontsize=11)
    ax.set_title('Collision avoidance: original and evaded trajectories (ECI X-Y)', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Results text below the figure (not on the plot)
    lines = [
        f'Min separation: {min_dist_km:.4f} km',
        f'Radius constraint: {radius_km} km',
        f'Near miss: {"Yes" if has_near else "No"}',
        f'|Delta-v|: {dv_norm * 1000:.4f} m/s',
    ]
    if min_dist_after_km is not None:
        lines.append(f'Min sep. after evasion: {min_dist_after_km:.4f} km')
    if t_idx_closest >= 0:
        lines.append(f'Min miss: sep = {min_dist_km:.4f} km, radius = {radius_km} km')
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.5, 0.06, '\n'.join(lines), ha='center', va='bottom', fontsize=9,
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.95))

    if save_path is None and PATHS.get('plots_dir'):
        os.makedirs(PATHS['plots_dir'], exist_ok=True)
        save_path = os.path.join(PATHS['plots_dir'], 'evasion_2d.png')
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"Evasion 2D plot saved to: {save_path}")

    return fig, ax


def animate_evasion_2d(
    save_path=None,
    radius_km=None,
    max_frames=120,
    fps=15,
):
    """
    Animate the 2D evasion plot: trajectories grow over time, current positions move,
    delta-v at t=0 and min-miss point shown. Saves as GIF.

    Parameters
    ----------
    save_path : str, optional
        Path for the GIF. If None, uses PATHS['plots_dir']/evasion_2d.gif.
    radius_km : float, optional
        Near-miss radius (km). If None, uses DEFAULT_RADIUS_KM.
    max_frames : int
        Number of animation frames (subsampled from time steps if needed).
    fps : int
        Frames per second for the GIF.
    """
    import sys
    try:
        from params import SATELLITE, CONSTANTS, ADVERSARY_TRANSFER, PATHS
    except ImportError:
        _sim_dir = os.path.dirname(os.path.abspath(__file__))
        if _sim_dir not in sys.path:
            sys.path.insert(0, _sim_dir)
        from params import SATELLITE, CONSTANTS, ADVERSARY_TRANSFER, PATHS

    project_root = PATHS.get('project_root', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from sat.control.rtn_to_eci_propagate import (
        propagate_chief,
        propagate_constellation,
        eci_to_relative_rtn,
        kepler_to_rv,
    )
    from sat.control.collision_avoidance import (
        apply_impulse_eci,
        collision_avoidance_delta_v,
        detect_near_miss,
        DEFAULT_RADIUS_KM,
    )

    if radius_km is None:
        radius_km = DEFAULT_RADIUS_KM
    mu_km = CONSTANTS['mu'] / 1e9
    earth_km = CONSTANTS['earth_radius_km']
    a_c = earth_km + float(SATELLITE['altitude'])
    r_chief, v_chief = kepler_to_rv(
        a_c, float(SATELLITE['eccentricity']),
        np.radians(float(SATELLITE['inclination'])),
        np.radians(float(SATELLITE['omega'])), np.radians(float(SATELLITE['Omega'])),
        np.radians(float(SATELLITE['M'])), mu=mu_km,
    )
    r_adv, v_adv = kepler_to_rv(
        float(ADVERSARY_TRANSFER['semi_major_axis']), float(ADVERSARY_TRANSFER['eccentricity']),
        np.radians(float(ADVERSARY_TRANSFER['inclination'])),
        np.radians(float(ADVERSARY_TRANSFER['omega'])), np.radians(float(ADVERSARY_TRANSFER['Omega'])),
        np.radians(float(ADVERSARY_TRANSFER['M'])), mu=mu_km,
    )
    constellation_rtn = np.asarray(eci_to_relative_rtn(r_chief, v_chief, r_adv, v_adv), dtype=float).reshape(1, 6)

    T_orbit = 2 * np.pi * np.sqrt((a_c ** 3) / mu_km)
    n_times = max(200, int(T_orbit / 10.0))
    time_array = np.linspace(0, T_orbit, n_times)

    pos_chief = propagate_chief(r_chief, v_chief, time_array, backend='cpu')
    pos_adv = propagate_constellation(
        r_chief, v_chief, constellation_rtn, time_array, backend='cpu', return_velocity=False
    )
    if pos_adv.ndim == 3:
        pos_adv = pos_adv[:, 0, :]
    # else already (n_times, 3)

    has_near, min_dist_km, t_idx_closest, _ = detect_near_miss(
        pos_chief, pos_adv[:, np.newaxis, :], radius_km=radius_km
    )
    delta_v = collision_avoidance_delta_v(
        r_chief, v_chief, constellation_rtn,
        backend='cpu', radius_km=radius_km, n_times_orbit=max(200, n_times),
    )
    dv_norm = np.linalg.norm(delta_v)
    pos_evaded = None
    min_dist_after_km = None
    if dv_norm > 1e-9:
        r_ev, v_ev = apply_impulse_eci(r_chief, v_chief, delta_v)
        pos_evaded = propagate_chief(r_ev, v_ev, time_array, backend='cpu')
        _, min_dist_after_km, _, _ = detect_near_miss(
            pos_evaded, pos_adv[:, np.newaxis, :], radius_km=radius_km
        )

    # Subsample for animation
    idx = np.linspace(0, n_times - 1, max_frames).astype(int)
    time_anim = time_array[idx]
    pos_chief_a = pos_chief[idx]
    pos_adv_a = pos_adv[idx]
    pos_evaded_a = pos_evaded[idx] if pos_evaded is not None else None
    n_f = len(idx)
    t_idx_closest_anim = int(np.argmin(np.abs(idx - t_idx_closest))) if t_idx_closest >= 0 else -1
    # Pause at min-miss for 3 seconds: duplicate that frame index (3 * fps) times
    pause_frames = int(3 * fps) if t_idx_closest_anim >= 0 else 0
    if pause_frames > 0:
        frame_to_data = np.arange(n_f, dtype=int)
        frame_to_data = np.insert(
            frame_to_data,
            t_idx_closest_anim + 1,
            np.full(pause_frames, t_idx_closest_anim)
        )
    else:
        frame_to_data = np.arange(n_f, dtype=int)
    n_frames_total = len(frame_to_data)

    margin_dynamic_km = 10.0
    margin_static_km = 80.0
    all_xy = [pos_chief[:, :2], pos_adv[:, :2]]
    if pos_evaded is not None:
        all_xy.append(pos_evaded[:, :2])
    stack_full = np.vstack(all_xy)
    x_min_s, x_max_s = stack_full[:, 0].min(), stack_full[:, 0].max()
    y_min_s, y_max_s = stack_full[:, 1].min(), stack_full[:, 1].max()
    xlim_static = (x_min_s - margin_static_km, x_max_s + margin_static_km)
    ylim_static = (y_min_s - margin_static_km, y_max_s + margin_static_km)

    fig, (ax_static, ax_dynamic) = plt.subplots(1, 2, figsize=(16, 8))
    for ax in (ax_static, ax_dynamic):
        ax.set_aspect('equal')
        ax.set_xlabel('X (ECI), km', fontsize=10)
        ax.set_ylabel('Y (ECI), km', fontsize=10)
        ax.grid(True, alpha=0.3)
    ax_static.set_xlim(xlim_static)
    ax_static.set_ylim(ylim_static)
    ax_static.set_title('ECI X–Y: full trajectory (static view)', fontsize=11)
    ax_dynamic.set_title('ECI X–Y: zoom on current positions (dynamic view)', fontsize=11)

    def add_artists(ax):
        lc, = ax.plot([], [], color='green', lw=2, alpha=0.9, label='Chief (original)')
        la, = ax.plot([], [], color='red', lw=2, alpha=0.9, label='Adversary')
        le = None
        if pos_evaded_a is not None:
            le, = ax.plot([], [], color='cyan', lw=2, linestyle='--', alpha=0.9, label='Chief (evaded)')
        ax.scatter([r_chief[0]], [r_chief[1]], c='green', s=100, marker='o', edgecolors='black', linewidths=2,
                   zorder=6, label='$\\Delta v$ (t=0)')
        ax.annotate('$\\Delta v$', (r_chief[0], r_chief[1]), textcoords='offset points', xytext=(6, 6),
                    fontsize=10, fontweight='bold', zorder=6)
        sc_c = ax.scatter([], [], c='green', s=60, marker='o', edgecolors='black', zorder=7)
        sc_a = ax.scatter([], [], c='red', s=60, marker='o', edgecolors='black', zorder=7)
        sc_e = ax.scatter([], [], c='cyan', s=60, marker='s', edgecolors='black', zorder=7) if pos_evaded_a is not None else None
        sc_mc = ax.scatter([], [], c='green', s=80, marker='s', edgecolors='black', zorder=8)
        sc_ma = ax.scatter([], [], c='red', s=80, marker='s', edgecolors='black', zorder=8)
        ann = ax.annotate('', xy=(0, 0), fontsize=8, ha='left', visible=False,
                         xytext=(0, 0), textcoords='data',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.9))
        return lc, la, le, sc_c, sc_a, sc_e, sc_mc, sc_ma, ann

    (line_chief_s, line_adv_s, line_evaded_s, sc_chief_s, sc_adv_s, sc_evaded_s,
     sc_miss_chief_s, sc_miss_adv_s, miss_annot_s) = add_artists(ax_static)
    (line_chief_d, line_adv_d, line_evaded_d, sc_chief_d, sc_adv_d, sc_evaded_d,
     sc_miss_chief_d, sc_miss_adv_d, miss_annot_d) = add_artists(ax_dynamic)

    lines_text = [
        f'Min separation: {min_dist_km:.4f} km',
        f'Radius: {radius_km} km  Near miss: {"Yes" if has_near else "No"}',
        f'|Delta-v|: {dv_norm * 1000:.4f} m/s',
    ]
    if min_dist_after_km is not None:
        lines_text.append(f'Min sep. after evasion: {min_dist_after_km:.4f} km')
    lines_text.append(f'Min miss: sep = {min_dist_km:.4f} km, radius = {radius_km} km')
    # Leave room at top (legend, time) and bottom (caption); keep legend/time closer to plots
    fig.subplots_adjust(top=0.88, bottom=0.20)
    time_text = fig.text(0.5, 0.94, '', fontsize=11, ha='center')
    caption_text = fig.text(0.5, 0.06, '\n'.join(lines_text), fontsize=8, ha='center', va='bottom',
                            fontfamily='monospace',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.95))
    # Single shared legend (handles from left panel), placed just above the plots
    legend_handles = [line_chief_s, line_adv_s]
    legend_labels = ['Chief (original)', 'Adversary']
    if line_evaded_s is not None:
        legend_handles.append(line_evaded_s)
        legend_labels.append('Chief (evaded)')
    proxy_dv = Line2D([0], [0], linestyle='none', marker='o', markersize=8, markerfacecolor='green',
                      markeredgecolor='black', label='$\\Delta v$ (t=0)')
    proxy_miss = Line2D([0], [0], linestyle='none', marker='s', markersize=8, markerfacecolor='gray',
                        markeredgecolor='black', label='Min miss')
    legend_handles.extend([proxy_dv, proxy_miss])
    legend_labels.extend(['$\\Delta v$ (t=0)', 'Min miss'])
    fig.legend(legend_handles, legend_labels, loc='upper center', ncol=len(legend_handles), fontsize=8, bbox_to_anchor=(0.5, 0.91))

    def init():
        for ax, lc, la, le, sc_c, sc_a, sc_e, sc_mc, sc_ma, ann in [
            (ax_static, line_chief_s, line_adv_s, line_evaded_s, sc_chief_s, sc_adv_s, sc_evaded_s,
             sc_miss_chief_s, sc_miss_adv_s, miss_annot_s),
            (ax_dynamic, line_chief_d, line_adv_d, line_evaded_d, sc_chief_d, sc_adv_d, sc_evaded_d,
             sc_miss_chief_d, sc_miss_adv_d, miss_annot_d),
        ]:
            lc.set_data([], [])
            la.set_data([], [])
            if le is not None:
                le.set_data([], [])
            sc_c.set_offsets(np.empty((0, 2)))
            sc_a.set_offsets(np.empty((0, 2)))
            if sc_e is not None:
                sc_e.set_offsets(np.empty((0, 2)))
            sc_mc.set_offsets(np.empty((0, 2)))
            sc_ma.set_offsets(np.empty((0, 2)))
            ann.set_visible(False)
        return ()

    def update(i):
        j = frame_to_data[i]
        t_s = time_anim[j]
        time_text.set_text(f't = {t_s:.0f} s ({t_s / 60:.1f} min)')
        for lc, la, le, sc_c, sc_a, sc_e, sc_mc, sc_ma, ann in [
            (line_chief_s, line_adv_s, line_evaded_s, sc_chief_s, sc_adv_s, sc_evaded_s,
             sc_miss_chief_s, sc_miss_adv_s, miss_annot_s),
            (line_chief_d, line_adv_d, line_evaded_d, sc_chief_d, sc_adv_d, sc_evaded_d,
             sc_miss_chief_d, sc_miss_adv_d, miss_annot_d),
        ]:
            lc.set_data(pos_chief_a[: j + 1, 0], pos_chief_a[: j + 1, 1])
            la.set_data(pos_adv_a[: j + 1, 0], pos_adv_a[: j + 1, 1])
            if le is not None and pos_evaded_a is not None:
                le.set_data(pos_evaded_a[: j + 1, 0], pos_evaded_a[: j + 1, 1])
            sc_c.set_offsets(pos_chief_a[j, :2].reshape(1, 2))
            sc_a.set_offsets(pos_adv_a[j, :2].reshape(1, 2))
            if sc_e is not None:
                sc_e.set_offsets(pos_evaded_a[j, :2].reshape(1, 2))
            if t_idx_closest_anim >= 0 and j >= t_idx_closest_anim:
                sc_mc.set_offsets(pos_chief_a[t_idx_closest_anim, :2].reshape(1, 2))
                sc_ma.set_offsets(pos_adv_a[t_idx_closest_anim, :2].reshape(1, 2))
                # Min-miss / min-separation text is in caption below figure; no on-plot annotation
                ann.set_visible(False)
            else:
                sc_mc.set_offsets(np.empty((0, 2)))
                sc_ma.set_offsets(np.empty((0, 2)))
                ann.set_visible(False)
        # Dynamic axes only: current positions + margin
        box_xy = [pos_chief_a[j, :2].reshape(1, 2), pos_adv_a[j, :2].reshape(1, 2)]
        if pos_evaded_a is not None:
            box_xy.append(pos_evaded_a[j, :2].reshape(1, 2))
        stack = np.vstack(box_xy)
        x_min, x_max = stack[:, 0].min(), stack[:, 0].max()
        y_min, y_max = stack[:, 1].min(), stack[:, 1].max()
        ax_dynamic.set_xlim(x_min - margin_dynamic_km, x_max + margin_dynamic_km)
        ax_dynamic.set_ylim(y_min - margin_dynamic_km, y_max + margin_dynamic_km)
        return ()

    anim = FuncAnimation(fig, update, init_func=init, frames=n_frames_total, interval=1000 / fps, blit=False)

    if save_path is None and PATHS.get('plots_dir'):
        os.makedirs(PATHS['plots_dir'], exist_ok=True)
        save_path = os.path.join(PATHS['plots_dir'], 'evasion_2d.gif')
    if save_path:
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=100)
        plt.close(fig)
        print(f"Evasion 2D animation saved to: {save_path}")
    return anim


def animate_evasion_2d_dynamic_only(
    save_path=None,
    radius_km=None,
    max_frames=120,
    fps=15,
):
    """
    Evasion 2D GIF with only the dynamic view (zoom on current positions).
    Same data and behavior as the right panel of animate_evasion_2d.
    """
    import sys
    try:
        from params import SATELLITE, CONSTANTS, ADVERSARY_TRANSFER, PATHS
    except ImportError:
        _sim_dir = os.path.dirname(os.path.abspath(__file__))
        if _sim_dir not in sys.path:
            sys.path.insert(0, _sim_dir)
        from params import SATELLITE, CONSTANTS, ADVERSARY_TRANSFER, PATHS

    project_root = PATHS.get('project_root', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from sat.control.rtn_to_eci_propagate import (
        propagate_chief,
        propagate_constellation,
        eci_to_relative_rtn,
        kepler_to_rv,
    )
    from sat.control.collision_avoidance import (
        apply_impulse_eci,
        collision_avoidance_delta_v,
        detect_near_miss,
        DEFAULT_RADIUS_KM,
    )

    if radius_km is None:
        radius_km = DEFAULT_RADIUS_KM
    mu_km = CONSTANTS['mu'] / 1e9
    earth_km = CONSTANTS['earth_radius_km']
    a_c = earth_km + float(SATELLITE['altitude'])
    r_chief, v_chief = kepler_to_rv(
        a_c, float(SATELLITE['eccentricity']),
        np.radians(float(SATELLITE['inclination'])),
        np.radians(float(SATELLITE['omega'])), np.radians(float(SATELLITE['Omega'])),
        np.radians(float(SATELLITE['M'])), mu=mu_km,
    )
    r_adv, v_adv = kepler_to_rv(
        float(ADVERSARY_TRANSFER['semi_major_axis']), float(ADVERSARY_TRANSFER['eccentricity']),
        np.radians(float(ADVERSARY_TRANSFER['inclination'])),
        np.radians(float(ADVERSARY_TRANSFER['omega'])), np.radians(float(ADVERSARY_TRANSFER['Omega'])),
        np.radians(float(ADVERSARY_TRANSFER['M'])), mu=mu_km,
    )
    constellation_rtn = np.asarray(eci_to_relative_rtn(r_chief, v_chief, r_adv, v_adv), dtype=float).reshape(1, 6)

    T_orbit = 2 * np.pi * np.sqrt((a_c ** 3) / mu_km)
    n_times = max(200, int(T_orbit / 10.0))
    time_array = np.linspace(0, T_orbit, n_times)

    pos_chief = propagate_chief(r_chief, v_chief, time_array, backend='cpu')
    pos_adv = propagate_constellation(
        r_chief, v_chief, constellation_rtn, time_array, backend='cpu', return_velocity=False
    )
    if pos_adv.ndim == 3:
        pos_adv = pos_adv[:, 0, :]

    has_near, min_dist_km, t_idx_closest, _ = detect_near_miss(
        pos_chief, pos_adv[:, np.newaxis, :], radius_km=radius_km
    )
    delta_v = collision_avoidance_delta_v(
        r_chief, v_chief, constellation_rtn,
        backend='cpu', radius_km=radius_km, n_times_orbit=max(200, n_times),
    )
    dv_norm = np.linalg.norm(delta_v)
    pos_evaded = None
    min_dist_after_km = None
    if dv_norm > 1e-9:
        r_ev, v_ev = apply_impulse_eci(r_chief, v_chief, delta_v)
        pos_evaded = propagate_chief(r_ev, v_ev, time_array, backend='cpu')
        _, min_dist_after_km, _, _ = detect_near_miss(
            pos_evaded, pos_adv[:, np.newaxis, :], radius_km=radius_km
        )

    idx = np.linspace(0, n_times - 1, max_frames).astype(int)
    time_anim = time_array[idx]
    pos_chief_a = pos_chief[idx]
    pos_adv_a = pos_adv[idx]
    pos_evaded_a = pos_evaded[idx] if pos_evaded is not None else None
    n_f = len(idx)
    t_idx_closest_anim = int(np.argmin(np.abs(idx - t_idx_closest))) if t_idx_closest >= 0 else -1
    pause_frames = int(3 * fps) if t_idx_closest_anim >= 0 else 0
    if pause_frames > 0:
        frame_to_data = np.insert(
            np.arange(n_f, dtype=int), t_idx_closest_anim + 1,
            np.full(pause_frames, t_idx_closest_anim)
        )
    else:
        frame_to_data = np.arange(n_f, dtype=int)
    n_frames_total = len(frame_to_data)

    margin_dynamic_km = 10.0

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xlabel('X (ECI), km', fontsize=10)
    ax.set_ylabel('Y (ECI), km', fontsize=10)
    ax.set_title('ECI X–Y: zoom on current positions (dynamic view)', fontsize=11)
    ax.grid(True, alpha=0.3)

    def add_artists(ax):
        lc, = ax.plot([], [], color='green', lw=2, alpha=0.9, label='Chief (original)')
        la, = ax.plot([], [], color='red', lw=2, alpha=0.9, label='Adversary')
        le = None
        if pos_evaded_a is not None:
            le, = ax.plot([], [], color='cyan', lw=2, linestyle='--', alpha=0.9, label='Chief (evaded)')
        ax.scatter([r_chief[0]], [r_chief[1]], c='green', s=100, marker='o', edgecolors='black', linewidths=2, zorder=6, label='$\\Delta v$ (t=0)')
        ax.annotate('$\\Delta v$', (r_chief[0], r_chief[1]), textcoords='offset points', xytext=(6, 6), fontsize=10, fontweight='bold', zorder=6)
        sc_c = ax.scatter([], [], c='green', s=60, marker='o', edgecolors='black', zorder=7)
        sc_a = ax.scatter([], [], c='red', s=60, marker='o', edgecolors='black', zorder=7)
        sc_e = ax.scatter([], [], c='cyan', s=60, marker='s', edgecolors='black', zorder=7) if pos_evaded_a is not None else None
        sc_mc = ax.scatter([], [], c='green', s=80, marker='s', edgecolors='black', zorder=8)
        sc_ma = ax.scatter([], [], c='red', s=80, marker='s', edgecolors='black', zorder=8)
        ann = ax.annotate('', xy=(0, 0), fontsize=8, ha='left', visible=False, xytext=(0, 0), textcoords='data',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.9))
        return lc, la, le, sc_c, sc_a, sc_e, sc_mc, sc_ma, ann

    (line_chief, line_adv, line_evaded, sc_chief, sc_adv, sc_evaded, sc_miss_chief, sc_miss_adv, miss_annot) = add_artists(ax)

    fig.subplots_adjust(top=0.90, bottom=0.12)
    time_text = fig.text(0.5, 0.96, '', fontsize=11, ha='center')
    ax.legend(loc='upper right', fontsize=9)

    def init():
        line_chief.set_data([], [])
        line_adv.set_data([], [])
        if line_evaded is not None:
            line_evaded.set_data([], [])
        sc_chief.set_offsets(np.empty((0, 2)))
        sc_adv.set_offsets(np.empty((0, 2)))
        if sc_evaded is not None:
            sc_evaded.set_offsets(np.empty((0, 2)))
        sc_miss_chief.set_offsets(np.empty((0, 2)))
        sc_miss_adv.set_offsets(np.empty((0, 2)))
        miss_annot.set_visible(False)
        return ()

    def update(i):
        j = frame_to_data[i]
        t_s = time_anim[j]
        time_text.set_text(f't = {t_s:.0f} s ({t_s / 60:.1f} min)')
        line_chief.set_data(pos_chief_a[: j + 1, 0], pos_chief_a[: j + 1, 1])
        line_adv.set_data(pos_adv_a[: j + 1, 0], pos_adv_a[: j + 1, 1])
        if line_evaded is not None and pos_evaded_a is not None:
            line_evaded.set_data(pos_evaded_a[: j + 1, 0], pos_evaded_a[: j + 1, 1])
        sc_chief.set_offsets(pos_chief_a[j, :2].reshape(1, 2))
        sc_adv.set_offsets(pos_adv_a[j, :2].reshape(1, 2))
        if sc_evaded is not None:
            sc_evaded.set_offsets(pos_evaded_a[j, :2].reshape(1, 2))
        if t_idx_closest_anim >= 0 and j >= t_idx_closest_anim:
            sc_miss_chief.set_offsets(pos_chief_a[t_idx_closest_anim, :2].reshape(1, 2))
            sc_miss_adv.set_offsets(pos_adv_a[t_idx_closest_anim, :2].reshape(1, 2))
        else:
            sc_miss_chief.set_offsets(np.empty((0, 2)))
            sc_miss_adv.set_offsets(np.empty((0, 2)))
        miss_annot.set_visible(False)
        box_xy = [pos_chief_a[j, :2].reshape(1, 2), pos_adv_a[j, :2].reshape(1, 2)]
        if pos_evaded_a is not None:
            box_xy.append(pos_evaded_a[j, :2].reshape(1, 2))
        stack = np.vstack(box_xy)
        x_min, x_max = stack[:, 0].min(), stack[:, 0].max()
        y_min, y_max = stack[:, 1].min(), stack[:, 1].max()
        ax.set_xlim(x_min - margin_dynamic_km, x_max + margin_dynamic_km)
        ax.set_ylim(y_min - margin_dynamic_km, y_max + margin_dynamic_km)
        return ()

    anim = FuncAnimation(fig, update, init_func=init, frames=n_frames_total, interval=1000 / fps, blit=False)

    if save_path is None and PATHS.get('plots_dir'):
        os.makedirs(PATHS['plots_dir'], exist_ok=True)
        save_path = os.path.join(PATHS['plots_dir'], 'evasion_2d_dynamic.gif')
    if save_path:
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=100)
        plt.close(fig)
        print(f"Evasion 2D dynamic-view animation saved to: {save_path}")
    return anim


if __name__ == '__main__':
    import sys
    _sim_dir = os.path.dirname(os.path.abspath(__file__))
    if _sim_dir not in sys.path:
        sys.path.insert(0, _sim_dir)
    if len(sys.argv) > 1 and sys.argv[1] == 'evasion_2d':
        plot_evasion_2d()
        animate_evasion_2d()
        animate_evasion_2d_dynamic_only()
    elif len(sys.argv) > 1 and sys.argv[1] == 'evasion_2d_anim':
        animate_evasion_2d()
    elif len(sys.argv) > 1 and sys.argv[1] == 'evasion_2d_dynamic':
        animate_evasion_2d_dynamic_only()
    elif len(sys.argv) > 1 and sys.argv[1] == 'adversary_transfer':
        from params import PATHS
        plots_dir = PATHS.get('plots_dir')
        anim_path = os.path.join(plots_dir, 'adversary_transfer_scenario.gif') if plots_dir else None
        anim_rtn_path = os.path.join(plots_dir, 'adversary_transfer_scenario_rtn.gif') if plots_dir else None
        anim_eci_path = os.path.join(plots_dir, 'adversary_transfer_scenario_eci.gif') if plots_dir else None
        plot_adversary_transfer_scenario(
            show_evasion=True,
            save_animation_path=anim_path,
            save_animation_rtn_path=anim_rtn_path,
            save_animation_eci_path=anim_eci_path,
        )
        print('Adversary transfer: static PNG + ECI GIF + RTN GIF + ECI zoom GIF')
    elif len(sys.argv) > 1 and sys.argv[1] == 'maneuver_scenario':
        from params import PATHS, VISUALIZATION
        plots_dir = PATHS.get('plots_dir')
        earth_texture = PATHS.get('earth_texture') if PATHS else None
        earth_res = (VISUALIZATION.get('earth_resolution', 'low') if VISUALIZATION else 'low')
        plot_maneuver_scenario_2d(save_path=os.path.join(plots_dir, 'maneuver_scenario_2d.png') if plots_dir else None)
        animate_maneuver_scenario_2d(save_path=os.path.join(plots_dir, 'maneuver_scenario_2d.gif') if plots_dir else None)
        plot_maneuver_scenario_3d(
            save_path=os.path.join(plots_dir, 'maneuver_scenario_3d.png') if plots_dir else None,
            earth_texture_path=earth_texture,
            earth_resolution=earth_res,
        )
        animate_maneuver_scenario_3d(
            save_path=os.path.join(plots_dir, 'maneuver_scenario_3d.gif') if plots_dir else None,
            earth_texture_path=earth_texture,
            earth_resolution=earth_res,
        )
        print('Maneuver scenario: 2D PNG + 2D GIF + 3D PNG + 3D GIF')
    else:
        print('Usage: python sim/plots.py evasion_2d         # static + dual-panel GIF + dynamic-only GIF')
        print('       python sim/plots.py evasion_2d_anim   # dual-panel GIF only')
        print('       python sim/plots.py evasion_2d_dynamic # dynamic-view GIF only')
        print('       python sim/plots.py adversary_transfer  # static + ECI GIF + RTN GIF')
        print('       python sim/plots.py maneuver_scenario    # 5-orbit maneuver 2D/3D + GIFs')
