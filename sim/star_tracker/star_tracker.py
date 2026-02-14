"""
Star tracker image simulation.

Renders 256x256 pixel images of visible satellites from the perspective of the
observer satellite (index 0), with the camera boresight pointing in the +x_eci direction.
"""

import numpy as np


def render_star_tracker_image(
    observer_pos,
    satellite_positions,
    fov_deg=10.0,
    image_size=256,
    pointing_direction=np.array([1.0, 0.0, 0.0]),
    up_direction=np.array([0.0, 0.0, 1.0])
):
    """
    Render a star tracker image showing visible satellites as dots.

    Parameters:
    -----------
    observer_pos : ndarray
        Position of observer satellite in ECI coordinates [x, y, z] in km
    satellite_positions : ndarray
        Positions of all other satellites, shape (n_satellites, 3) in ECI coordinates (km)
    fov_deg : float
        Field of view in degrees (default: 10.0)
    image_size : int
        Image dimensions in pixels (default: 256 for 256x256)
    pointing_direction : ndarray
        Camera boresight direction in ECI frame (default: +x_eci)
    up_direction : ndarray
        Camera "up" direction in ECI frame (default: +z_eci)

    Returns:
    --------
    image : ndarray
        256x256 pixel image with satellites rendered as dots (1.0 for satellite, 0.0 for background)
    visible_sats : ndarray
        Indices of satellites that are visible in the image
    pixel_coords : ndarray
        Pixel coordinates [row, col] for each visible satellite
    """

    # Initialize image
    image = np.zeros((image_size, image_size), dtype=np.float32)

    # Normalize camera axes
    camera_z = pointing_direction / np.linalg.norm(pointing_direction)  # Boresight (forward)
    camera_x = np.cross(up_direction, camera_z)  # Right
    camera_x = camera_x / np.linalg.norm(camera_x)
    camera_y = np.cross(camera_z, camera_x)  # Up

    # Compute relative positions of all satellites w.r.t. observer
    relative_positions = satellite_positions - observer_pos  # Shape: (n_satellites, 3)

    # Transform to camera frame
    # Camera frame: x=right, y=up, z=forward (boresight)
    camera_coords = np.stack([
        np.dot(relative_positions, camera_x),  # x-coordinate in camera frame
        np.dot(relative_positions, camera_y),  # y-coordinate in camera frame
        np.dot(relative_positions, camera_z)   # z-coordinate in camera frame (distance along boresight)
    ], axis=1)

    # Filter satellites in front of camera (positive z)
    in_front = camera_coords[:, 2] > 0
    camera_coords_filtered = camera_coords[in_front]
    satellite_indices = np.where(in_front)[0]

    if len(camera_coords_filtered) == 0:
        return image, np.array([]), np.array([])

    # Calculate angular offsets from boresight
    distances = np.linalg.norm(camera_coords_filtered, axis=1)

    # Normalize to get direction vectors
    directions = camera_coords_filtered / distances[:, np.newaxis]

    # Angle from boresight (z-axis in camera frame)
    cos_angles = directions[:, 2]  # z-component gives cos(angle from boresight)
    angles_deg = np.degrees(np.arccos(np.clip(cos_angles, -1.0, 1.0)))

    # Filter by field of view
    half_fov = fov_deg / 2.0
    in_fov = angles_deg < half_fov

    if not np.any(in_fov):
        return image, np.array([]), np.array([])

    camera_coords_visible = camera_coords_filtered[in_fov]
    visible_sat_indices = satellite_indices[in_fov]

    # Project onto image plane using pinhole camera model
    # Focal length in pixels (relates FOV to image size)
    focal_length_px = (image_size / 2.0) / np.tan(np.radians(half_fov))

    # Project: pixel_x = focal_length * (camera_x / camera_z)
    pixel_x = focal_length_px * (camera_coords_visible[:, 0] / camera_coords_visible[:, 2])
    pixel_y = focal_length_px * (camera_coords_visible[:, 1] / camera_coords_visible[:, 2])

    # Convert to image coordinates (origin at center, y-axis flipped for image)
    center = image_size / 2.0
    col = (center + pixel_x).astype(int)
    row = (center - pixel_y).astype(int)  # Flip y-axis

    # Filter pixels within image bounds
    valid = (row >= 0) & (row < image_size) & (col >= 0) & (col < image_size)

    row_valid = row[valid]
    col_valid = col[valid]
    visible_sat_indices_final = visible_sat_indices[valid]

    # Render dots on image
    image[row_valid, col_valid] = 1.0

    # Prepare output
    pixel_coords = np.stack([row_valid, col_valid], axis=1)

    return image, visible_sat_indices_final, pixel_coords


def render_star_tracker_sequence(positions, observer_index=0, **kwargs):
    """
    Render a sequence of star tracker images across multiple timesteps.

    Parameters:
    -----------
    positions : ndarray
        Position array from simulation, shape (n_timesteps, n_satellites, 3)
    observer_index : int
        Index of the observer satellite (default: 0)
    **kwargs : dict
        Additional arguments passed to render_star_tracker_image

    Returns:
    --------
    images : ndarray
        Array of images, shape (n_timesteps, image_size, image_size)
    visible_sats_list : list
        List of visible satellite indices for each timestep
    pixel_coords_list : list
        List of pixel coordinates for each timestep
    """

    n_timesteps = positions.shape[0]
    n_satellites = positions.shape[1]
    image_size = kwargs.get('image_size', 256)

    images = np.zeros((n_timesteps, image_size, image_size), dtype=np.float32)
    visible_sats_list = []
    pixel_coords_list = []

    for t in range(n_timesteps):
        observer_pos = positions[t, observer_index, :]

        # Get all satellite positions except the observer
        other_indices = np.arange(n_satellites) != observer_index
        other_positions = positions[t, other_indices, :]

        image, visible_sats, pixel_coords = render_star_tracker_image(
            observer_pos, other_positions, **kwargs
        )

        images[t] = image
        visible_sats_list.append(visible_sats)
        pixel_coords_list.append(pixel_coords)

    return images, visible_sats_list, pixel_coords_list
