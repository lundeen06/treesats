"""
Angles-only navigation: bearing angles, triangulation, and HCW RTN state from images.

Process (summary):
  1. Image coords (u, v) in [-1, +1]. Origin (0,0) = center; ±1 = edge of FOV.
  2. FOV = 10° total → ±5° from center. So (u=-1, v=0) = -5° azimuth from center.
  3. Camera boresight = -T (minus along-track). So center (0,0) is the -T direction;
     (u=-1, v=0) is -5° in azimuth from -T. Use image_coords_to_bearing(u, v) to get
     the unit bearing from the camera to that point.
  4. From there: with 3 images (3 (u,v) at 3 times + observer orbit), triangulation
     gives depth and finite difference gives velocity → HCW state vector in RTN.

Input: dots as (u,v) in [-1,1] or normalized (u,v). Per image: Nx3 [satellite_ID, u, v].
Sliding window of 3 images → bearing, depth, velocity → HCW RTN (position, velocity).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from sat.control.rtn_to_eci_propagate import (
    eci_to_rtn_basis,
    vector_eci_to_rtn,
    vector_rtn_to_eci,
)


# -----------------------------------------------------------------------------
# Sensor FOV and image coordinate system
# -----------------------------------------------------------------------------
# Image coords (x_img, y_img): origin at center, extent ±1 in x and y.
# So x_img, y_img in [-1, 1]; ±1 = edge of FOV. 10° full FOV → half-FOV 5°.
SENSOR_FOV_DEG = 10.0
SENSOR_HALF_FOV_DEG = SENSOR_FOV_DEG / 2.0
MAX_NORMALIZED_UV = float(np.tan(np.radians(SENSOR_HALF_FOV_DEG)))  # ≈ 0.0875 for 10° FOV
IMAGE_COORD_LIMIT = 1.0  # image x,y in [-1, 1]; ±1 = edge of FOV


def image_coords_to_bearing(
    x_img: Union[float, np.ndarray],
    y_img: Union[float, np.ndarray],
    fov_deg: float = SENSOR_FOV_DEG,
    clip: bool = True,
) -> np.ndarray:
    """
    Image coords (x_img, y_img) → unit bearing from camera to that point (the dot).

    Convention: image origin at center; x_img, y_img in [-1, 1]. ±1 = edge of FOV.
    So a dot at (x_img, y_img) gives the bearing from the camera (one point) to the
    target (the other point). The limit is enforced by clipping to [-1, 1] when clip=True.

    Parameters
    ----------
    x_img, y_img : float or array
        Image coordinates, range [-1, 1]. +x = right, +y = up (or your image convention).
    fov_deg : float
        Full FOV in degrees (default 10). Half-FOV used: u = x_img * tan(half), v = y_img * tan(half).
    clip : bool
        If True (default), clip x_img, y_img to [-1, 1] before converting.

    Returns
    -------
    b : ndarray
        Unit bearing vector(s) in camera frame: direction from camera to the dot. Shape (3,) or (n, 3).
    """
    x_img = np.asarray(x_img, dtype=float)
    y_img = np.asarray(y_img, dtype=float)
    if clip:
        x_img = np.clip(x_img, -IMAGE_COORD_LIMIT, IMAGE_COORD_LIMIT)
        y_img = np.clip(y_img, -IMAGE_COORD_LIMIT, IMAGE_COORD_LIMIT)
    half_rad = np.radians(fov_deg / 2.0)
    max_uv = float(np.tan(half_rad))
    u = np.asarray(x_img, dtype=float) * max_uv
    v = np.asarray(y_img, dtype=float) * max_uv
    return normalized_plane_to_bearing(u, v)


def image_coords_to_uv(
    x_img: Union[float, np.ndarray],
    y_img: Union[float, np.ndarray],
    fov_deg: float = SENSOR_FOV_DEG,
    clip: bool = True,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Image coords [-1, 1] → (u, v) on normalized plane for use in pipelines expecting [sat_id, u, v].

    Same mapping as image_coords_to_bearing: u = x_img * tan(half_fov), v = y_img * tan(half_fov).
    Returns (u, v) so you can build detections as np.array([[sat_id, u, v], ...]).
    """
    x_img = np.asarray(x_img, dtype=float)
    y_img = np.asarray(y_img, dtype=float)
    if clip:
        x_img = np.clip(x_img, -IMAGE_COORD_LIMIT, IMAGE_COORD_LIMIT)
        y_img = np.clip(y_img, -IMAGE_COORD_LIMIT, IMAGE_COORD_LIMIT)
    half_rad = np.radians(fov_deg / 2.0)
    max_uv = float(np.tan(half_rad))
    u = x_img * max_uv
    v = y_img * max_uv
    return (u, v) if u.shape != () else (float(u), float(v))


def image_coords_in_fov(
    x_img: Union[float, np.ndarray],
    y_img: Union[float, np.ndarray],
    limit: float = IMAGE_COORD_LIMIT,
) -> Union[bool, np.ndarray]:
    """True if (x_img, y_img) is inside the image / FOV (|x|, |y| ≤ limit). Default limit=1."""
    x_img = np.asarray(x_img)
    y_img = np.asarray(y_img)
    ok = (np.abs(x_img) <= limit) & (np.abs(y_img) <= limit)
    return bool(ok) if np.isscalar(x_img) else ok


def uv_in_fov(
    u: Union[float, np.ndarray],
    v: Union[float, np.ndarray],
    max_uv: float = MAX_NORMALIZED_UV,
) -> Union[bool, np.ndarray]:
    """
    True if (u,v) lies inside the sensor FOV (square: |u|,|v| ≤ max_uv).

    max_uv = tan(half_FOV_rad). Default uses SENSOR_FOV_DEG (10°).
    """
    u, v = np.asarray(u), np.asarray(v)
    ok = (np.abs(u) <= max_uv) & (np.abs(v) <= max_uv)
    return bool(ok) if np.isscalar(u) else ok


def clip_uv_to_fov(
    u: Union[float, np.ndarray],
    v: Union[float, np.ndarray],
    max_uv: float = MAX_NORMALIZED_UV,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Clip (u,v) to the sensor FOV square. Returns (u_clipped, v_clipped)."""
    u = np.clip(np.asarray(u, dtype=float), -max_uv, max_uv)
    v = np.clip(np.asarray(v, dtype=float), -max_uv, max_uv)
    return (u, v) if u.shape != () else (float(u), float(v))


# -----------------------------------------------------------------------------
# 1) BEARING = UNIT VECTOR (camera frame, optical axis +z)
# -----------------------------------------------------------------------------
# (u, v) on normalized plane → b = (u, v, 1) / sqrt(u^2 + v^2 + 1). No intrinsics.
#
# (u,v) coordinate system:
#   - Origin (0, 0) = boresight (optical axis, center of "image").
#   - u = tangent of angle in x (right); v = tangent of angle in y (up).
#   - With SENSOR_FOV_DEG = 10: |u|, |v| ≤ MAX_NORMALIZED_UV ≈ 0.0875.
# -----------------------------------------------------------------------------

def normalized_plane_to_bearing(
    u: Union[float, np.ndarray],
    v: Union[float, np.ndarray],
) -> np.ndarray:
    """
    (u, v) on normalized plane → unit bearing in camera frame.

    Optical axis +z. Formula: b = (u, v, 1) / sqrt(u^2 + v^2 + 1).

    Parameters
    ----------
    u, v : float or array
        Coordinates on normalized plane. Shape (n,) for multiple points.

    Returns
    -------
    b : ndarray
        Unit vector(s) in camera frame, shape (3,) or (n, 3). Each row (b_x, b_y, b_z).
    """
    u = np.atleast_1d(np.asarray(u, dtype=float))
    v = np.atleast_1d(np.asarray(v, dtype=float))
    one = np.ones_like(u)
    ray = np.stack([u, v, one], axis=-1)
    nrm = np.linalg.norm(ray, axis=-1, keepdims=True)
    nrm = np.where(nrm > 0, nrm, 1.0)
    b = ray / nrm
    return b.squeeze() if b.shape[0] == 1 else b


def bearing_to_angles(
    b: np.ndarray,
    convention: str = "azimuth_elevation",
) -> Union[Tuple[float, float], np.ndarray]:
    """
    Unit bearing vector → angles (for display or legacy interfaces).

    Estimator should keep unit vectors; angles have singularities. Optical axis +z.

    Conventions:
      - "azimuth_elevation": theta = arctan2(b_x, b_z), phi = arctan2(b_y, b_z)
      - "spherical": alpha = arctan2(b_y, b_x), beta = arccos(b_z) (inclination from +z)

    Parameters
    ----------
    b : ndarray shape (3,) or (n, 3)
        Unit bearing vector(s) in camera frame.
    convention : str
        "azimuth_elevation" or "spherical".

    Returns
    -------
    angles : (theta, phi) or (alpha, beta) in radians, or (n, 2) array.
    """
    b = np.asarray(b, dtype=float)
    single = b.ndim == 1
    if single:
        b = b.reshape(1, 3)
    if convention == "azimuth_elevation":
        theta = np.arctan2(b[:, 0], b[:, 2])
        phi = np.arctan2(b[:, 1], b[:, 2])
        out = np.column_stack([theta, phi])
    else:
        alpha = np.arctan2(b[:, 1], b[:, 0])
        beta = np.arccos(np.clip(b[:, 2], -1.0, 1.0))
        out = np.column_stack([alpha, beta])
    return (out[0, 0], out[0, 1]) if single else out


def angles_to_bearing(
    theta: Union[float, np.ndarray],
    phi: Union[float, np.ndarray],
    convention: str = "azimuth_elevation",
) -> np.ndarray:
    """
    Angles → unit bearing in camera frame (optical axis +z).

    "azimuth_elevation": b_z = 1/sqrt(1 + tan^2(theta) + tan^2(phi)), b_x = tan(theta)*b_z, b_y = tan(phi)*b_z.
    "spherical": alpha, beta → b = (sin(beta)*cos(alpha), sin(beta)*sin(alpha), cos(beta)).
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    if convention == "spherical":
        ct, st = np.cos(theta), np.sin(theta)
        cp, sp = np.cos(phi), np.sin(phi)
        b = np.stack([st * cp, st * sp, ct], axis=-1)
    else:
        bz = 1.0 / np.sqrt(1.0 + np.tan(theta) ** 2 + np.tan(phi) ** 2)
        bz = np.where(np.isfinite(bz), bz, 0.0)
        bx = np.tan(theta) * bz
        by = np.tan(phi) * bz
        b = np.stack([bx, by, bz], axis=-1)
    nrm = np.linalg.norm(b, axis=-1, keepdims=True)
    nrm = np.where(nrm > 0, nrm, 1.0)
    b = b / nrm
    return b.squeeze() if b.shape[0] == 1 else b


# -----------------------------------------------------------------------------
# 2) ECI ↔ RTN (Hill frame) — uses eci_to_rtn_basis, vector_eci_to_rtn, vector_rtn_to_eci
#    from sat.control.rtn_to_eci_propagate (see imports at top).
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# 3) CAMERA FRAME → ECI (then we can go to RTN)
# -----------------------------------------------------------------------------
# Star tracker: pointing_direction = boresight (camera +z), up_direction in ECI.
# Camera: x = right, y = up, z = forward. So in ECI:
#   cam_z_eci = pointing_direction (forward)
#   cam_x_eci = cross(up_direction, cam_z_eci) (right)
#   cam_y_eci = cross(cam_z_eci, cam_x_eci) (up)
# Columns of R_cam_to_eci = [cam_x_eci, cam_y_eci, cam_z_eci]. So v_eci = R_cam_to_eci @ v_cam.
# -----------------------------------------------------------------------------

def camera_to_eci_rotation(
    pointing_direction: np.ndarray,
    up_direction: np.ndarray,
) -> np.ndarray:
    """
    Rotation matrix from camera frame to ECI.

    Camera: x=right, y=up, z=forward (boresight). Returns 3x3 such that v_eci = R @ v_cam.
    """
    z = np.asarray(pointing_direction, dtype=float).ravel()
    z = z / np.linalg.norm(z)
    up = np.asarray(up_direction, dtype=float).ravel()
    x = np.cross(up, z)
    n = np.linalg.norm(x)
    if n < 1e-12:
        if abs(z[2]) < 0.9:
            x = np.cross(z, np.array([0, 0, 1.0]))
        else:
            x = np.cross(z, np.array([1.0, 0, 0]))
        x = x / np.linalg.norm(x)
    else:
        x = x / n
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def los_cam_to_eci(los_cam: np.ndarray, R_cam_to_eci: np.ndarray) -> np.ndarray:
    """Convert unit LOS in camera frame to unit LOS in ECI."""
    los = np.asarray(los_cam, dtype=float)
    if los.ndim == 1:
        return R_cam_to_eci @ los
    return (R_cam_to_eci @ los.T).T


# -----------------------------------------------------------------------------
# 4) TRIANGULATION (depth from two views)
# -----------------------------------------------------------------------------
# Same target P: P = O1 + ρ ℓ1  and  P = O2 + ρ' ℓ2  (ℓ1, ℓ2 unit LOS in ECI).
# So (O1 - O2) + ρ ℓ1 = ρ' ℓ2  =>  d + ρ ℓ1 - ρ' ℓ2 = 0  with d = O1 - O2.
# Condition that P lies on ray from O2: (P - O2) × ℓ2 = 0  =>  (d + ρ ℓ1) × ℓ2 = 0
# =>  ρ (ℓ1 × ℓ2) = ℓ2 × d   =>  ρ = (ℓ2 × d) · (ℓ1 × ℓ2) / |ℓ1 × ℓ2|^2.
# We use the same formula in ECI; then relative position at t1 in ECI is ρ ℓ1,
# and we convert to RTN at t1.
# -----------------------------------------------------------------------------

def triangulate_range(
    los1_eci: np.ndarray,
    los2_eci: np.ndarray,
    observer1_eci: np.ndarray,
    observer2_eci: np.ndarray,
) -> Tuple[float, float]:
    """
    Get range from observer 1 along los1 such that the ray meets the ray from observer 2.

    P = O1 + ρ*los1 = O2 + ρ'*los2. Returns (ρ, ρ') for the first and second observer.
    If rays are nearly parallel, returns (np.nan, np.nan).
    """
    d = np.asarray(observer1_eci, dtype=float).ravel() - np.asarray(observer2_eci, dtype=float).ravel()
    l1 = np.asarray(los1_eci, dtype=float).ravel()
    l2 = np.asarray(los2_eci, dtype=float).ravel()
    l1 = l1 / np.linalg.norm(l1)
    l2 = l2 / np.linalg.norm(l2)
    cross = np.cross(l1, l2)
    denom = np.dot(cross, cross)
    if denom < 1e-20:
        return np.nan, np.nan
    rho = np.dot(np.cross(l2, d), cross) / denom
    rho2 = np.dot(np.cross(d, l1), cross) / denom
    return float(rho), float(rho2)


def relative_position_rtn_at_t(
    rho: float,
    los_eci: np.ndarray,
    observer_eci: np.ndarray,
    observer_vel_eci: np.ndarray,
) -> np.ndarray:
    """
    Relative position of target in RTN at time t: r_rtn = R_eci_to_rtn @ (ρ * los_eci).
    (Target ECI = observer_eci + ρ*los_eci; relative = ρ*los_eci; express in RTN.)
    """
    rel_eci = rho * np.asarray(los_eci, dtype=float).ravel()
    basis = eci_to_rtn_basis(observer_eci, observer_vel_eci)
    return vector_eci_to_rtn(rel_eci, basis)


def rtn_position_to_predicted_image_coords(
    r_rtn: np.ndarray,
    observer_pos_eci: np.ndarray,
    observer_vel_eci: np.ndarray,
    R_cam_to_eci: np.ndarray,
    fov_deg: float = SENSOR_FOV_DEG,
) -> Tuple[float, float]:
    """
    From relative position in RTN, compute predicted (u_img, v_img) in [-1, 1] as seen by the camera.

    rel_eci = vector_rtn_to_eci(r_rtn), los = rel_eci/|rel_eci|, los_cam = R_cam_to_eci.T @ los,
    then u_img = (los_cam_x / los_cam_z) / tan(half_fov), same for v_img.
    """
    basis = eci_to_rtn_basis(observer_pos_eci, observer_vel_eci)
    rel_eci = vector_rtn_to_eci(r_rtn, basis)
    nrm = np.linalg.norm(rel_eci)
    if nrm < 1e-12:
        return 0.0, 0.0
    los_eci = rel_eci / nrm
    los_cam = np.dot(np.asarray(R_cam_to_eci, dtype=float).T, los_eci)
    if abs(los_cam[2]) < 1e-12:
        return 0.0, 0.0
    u_norm = los_cam[0] / los_cam[2]
    v_norm = los_cam[1] / los_cam[2]
    half_fov_rad = np.radians(fov_deg / 2.0)
    max_uv = np.tan(half_fov_rad)
    u_img = u_norm / max_uv
    v_img = v_norm / max_uv
    return float(u_img), float(v_img)


# -----------------------------------------------------------------------------
# 5) THREE-FRAME STATE: position and velocity at middle or latest image
# -----------------------------------------------------------------------------
# output_at="latest" (default): state at t3 (most recent image). Use for sliding window.
# output_at="middle": state at t2.
# -----------------------------------------------------------------------------

def three_frame_rtn_state(
    los1_eci: np.ndarray,
    los2_eci: np.ndarray,
    los3_eci: np.ndarray,
    obs_pos_eci: np.ndarray,
    obs_vel_eci: np.ndarray,
    t1: float,
    t2: float,
    t3: float,
    output_at: str = "latest",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute relative position (RTN) and velocity (RTN) from 3 bearings and observer orbit.

    output_at="latest" (default): position and velocity at t3 (third / most recent image).
    output_at="middle": position and velocity at t2.

    Returns
    -------
    r_rtn : ndarray (3,) position in RTN (km)
    v_rtn : ndarray (3,) velocity in RTN (km/s)
    """
    O1 = obs_pos_eci[0]
    O2 = obs_pos_eci[1]
    O3 = obs_pos_eci[2]
    v2 = obs_vel_eci[1]
    v3 = obs_vel_eci[2]

    rho1, _ = triangulate_range(los1_eci, los2_eci, O1, O2)
    _, rho2_from_12 = triangulate_range(los1_eci, los2_eci, O1, O2)
    rho2_from_23, rho3 = triangulate_range(los2_eci, los3_eci, O2, O3)

    if np.isnan(rho1) or np.isnan(rho3):
        return np.full(3, np.nan), np.full(3, np.nan)

    rho2 = rho2_from_12 if not np.isnan(rho2_from_12) else rho2_from_23
    if not np.isnan(rho2_from_23) and not np.isnan(rho2_from_12):
        rho2 = 0.5 * (rho2_from_12 + rho2_from_23)

    rel1_eci = rho1 * np.asarray(los1_eci, dtype=float).ravel()
    rel2_eci = rho2 * np.asarray(los2_eci, dtype=float).ravel()
    rel3_eci = rho3 * np.asarray(los3_eci, dtype=float).ravel()

    if output_at == "latest":
        r_rtn = relative_position_rtn_at_t(rho3, los3_eci, O3, v3)
        dt_vel = t3 - t2
        if dt_vel <= 0:
            v_rtn = np.zeros(3)
        else:
            v_rel_eci = (rel3_eci - rel2_eci) / dt_vel
            r_mag_sq = np.dot(O3, O3)
            h_vec = np.cross(O3, v3)
            omega_eci = h_vec / r_mag_sq
            v_rel_eci_minus_omega_r = v_rel_eci - np.cross(omega_eci, rel3_eci)
            basis3 = eci_to_rtn_basis(O3, v3)
            v_rtn = vector_eci_to_rtn(v_rel_eci_minus_omega_r, basis3)
    else:
        r_rtn = relative_position_rtn_at_t(rho2, los2_eci, O2, v2)
        dt = t3 - t1
        if dt <= 0:
            v_rtn = np.zeros(3)
        else:
            v_rel_eci = (rel3_eci - rel1_eci) / dt
            r_mag_sq = np.dot(O2, O2)
            h_vec = np.cross(O2, v2)
            omega_eci = h_vec / r_mag_sq
            v_rel_eci_minus_omega_r = v_rel_eci - np.cross(omega_eci, rel2_eci)
            basis2 = eci_to_rtn_basis(O2, v2)
            v_rtn = vector_eci_to_rtn(v_rel_eci_minus_omega_r, basis2)

    return r_rtn, v_rtn


def three_frame_rtn_positions(
    los1_eci: np.ndarray,
    los2_eci: np.ndarray,
    los3_eci: np.ndarray,
    obs_pos_eci: np.ndarray,
    obs_vel_eci: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Triangulate and return relative position in RTN at t1, t2, t3.
    Returns (r1_rtn, r2_rtn, r3_rtn). Use with rtn_position_to_predicted_image_coords for plotting.
    """
    O1, O2, O3 = obs_pos_eci[0], obs_pos_eci[1], obs_pos_eci[2]
    rho1, _ = triangulate_range(los1_eci, los2_eci, O1, O2)
    _, rho2_from_12 = triangulate_range(los1_eci, los2_eci, O1, O2)
    rho2_from_23, rho3 = triangulate_range(los2_eci, los3_eci, O2, O3)
    if np.isnan(rho1) or np.isnan(rho3):
        return np.full(3, np.nan), np.full(3, np.nan), np.full(3, np.nan)
    rho2 = rho2_from_12 if not np.isnan(rho2_from_12) else rho2_from_23
    if not np.isnan(rho2_from_23) and not np.isnan(rho2_from_12):
        rho2 = 0.5 * (rho2_from_12 + rho2_from_23)
    r1_rtn = relative_position_rtn_at_t(rho1, los1_eci, O1, obs_vel_eci[0])
    r2_rtn = relative_position_rtn_at_t(rho2, los2_eci, O2, obs_vel_eci[1])
    r3_rtn = relative_position_rtn_at_t(rho3, los3_eci, O3, obs_vel_eci[2])
    return r1_rtn, r2_rtn, r3_rtn


# -----------------------------------------------------------------------------
# 6) DETECTION MATRICES AND FRAME PROCESSING
# -----------------------------------------------------------------------------
# Each frame: detections = Nx3 array, rows [sat_id, u, v] on normalized plane.
# -----------------------------------------------------------------------------

def detections_to_los_eci_per_sat(
    frame_detections: List[np.ndarray],
    R_cam_to_eci_per_frame: List[np.ndarray],
) -> Dict[int, List[np.ndarray]]:
    """
    For each satellite ID present in any frame, build list of (frame_index, los_eci).

    frame_detections[i] = Nx3 array [sat_id, u, v]. (u, v) are on the normalized plane.
    """
    sat_to_frames: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    for fi, det in enumerate(frame_detections):
        if det is None or len(det) == 0:
            continue
        R = R_cam_to_eci_per_frame[fi]
        for row in det:
            sid = int(row[0])
            u, v = float(row[1]), float(row[2])
            los_cam = normalized_plane_to_bearing(u, v)
            los_eci = los_cam_to_eci(los_cam, R)
            if sid not in sat_to_frames:
                sat_to_frames[sid] = []
            sat_to_frames[sid].append((fi, los_eci))
    return sat_to_frames


def compute_rtn_state_for_sat(
    sat_id: int,
    frame_indices: List[int],
    los_eci_list: List[np.ndarray],
    obs_pos_eci: np.ndarray,
    obs_vel_eci: np.ndarray,
    times: np.ndarray,
    output_at: str = "latest",
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Compute RTN position and velocity for one satellite from 3 consecutive frames.

    output_at="latest": state at third (most recent) frame; "middle": at second frame.
    frame_indices and los_eci_list must have length 3; obs_pos_eci and obs_vel_eci shape (3,3); times length 3.
    """
    if len(frame_indices) != 3 or len(los_eci_list) != 3:
        return None
    i1, i2, i3 = frame_indices[0], frame_indices[1], frame_indices[2]
    r_rtn, v_rtn = three_frame_rtn_state(
        los_eci_list[0], los_eci_list[1], los_eci_list[2],
        obs_pos_eci, obs_vel_eci,
        times[i1], times[i2], times[i3],
        output_at=output_at,
    )
    if np.any(np.isnan(r_rtn)):
        return None
    return (r_rtn, v_rtn)


# -----------------------------------------------------------------------------
# 7) SLIDING FILTER: maintain latest 3 frames, running list of RTN states
# -----------------------------------------------------------------------------

@dataclass
class FrameInput:
    """Single frame: detections Nx3 [sat_id, u, v] (normalized plane), timestamp, observer state in ECI."""
    detections: np.ndarray  # (N, 3)
    timestamp: float
    observer_pos_eci: np.ndarray  # (3,)
    observer_vel_eci: np.ndarray  # (3,)
    R_cam_to_eci: np.ndarray  # (3, 3)


@dataclass
class RTNState:
    """HCW state in RTN: position (km), velocity (km/s)."""
    position_rtn: np.ndarray  # (3,) R, T, N
    velocity_rtn: np.ndarray  # (3,)
    timestamp: float  # time of this estimate


class SlidingFilter:
    """
    Keeps the latest 3 frames. For each satellite ID that appears in all 3,
    computes HCW RTN (position, velocity) at the latest image time.
    Adds new IDs when they get 3 frames; drops IDs that no longer appear in the latest 3.

    After each push(), .states is dict[sat_id -> RTNState] and .sat_ids is the list of
    all sat IDs that currently have an estimate (visible in the latest 3 frames).
    """

    def __init__(self) -> None:
        self._frames: List[FrameInput] = []
        self._max_frames = 3
        self._states: Dict[int, RTNState] = {}

    def push(
        self,
        detections: np.ndarray,
        timestamp: float,
        observer_pos_eci: np.ndarray,
        observer_vel_eci: np.ndarray,
        R_cam_to_eci: np.ndarray,
    ) -> Dict[int, RTNState]:
        """
        Append a new frame. If we have 3 frames, compute RTN for all sats visible in all 3
        at the latest image time; update running list (add new, remove missing).
        Returns current state dict (keys = all current sat IDs). Use .sat_ids for the list.
        """
        self._frames.append(FrameInput(
            detections=np.asarray(detections) if detections is not None and len(detections) else np.empty((0, 3)),
            timestamp=float(timestamp),
            observer_pos_eci=np.asarray(observer_pos_eci, dtype=float).ravel(),
            observer_vel_eci=np.asarray(observer_vel_eci, dtype=float).ravel(),
            R_cam_to_eci=np.asarray(R_cam_to_eci, dtype=float),
        ))
        if len(self._frames) > self._max_frames:
            self._frames.pop(0)

        self._update_states()
        return dict(self._states)

    def _update_states(self) -> None:
        if len(self._frames) < self._max_frames:
            return

        frame_detections = [f.detections for f in self._frames]
        R_per_frame = [f.R_cam_to_eci for f in self._frames]
        los_per_sat = detections_to_los_eci_per_sat(frame_detections, R_per_frame)

        obs_pos = np.stack([f.observer_pos_eci for f in self._frames], axis=0)
        obs_vel = np.stack([f.observer_vel_eci for f in self._frames], axis=0)
        times = np.array([f.timestamp for f in self._frames])

        current_ids = set()
        for sat_id, frame_los in los_per_sat.items():
            if len(frame_los) < self._max_frames:
                continue
            # One LOS per frame (take first detection per frame index)
            by_frame = {}
            for fi, los in frame_los:
                if fi not in by_frame:
                    by_frame[fi] = los
            if set(by_frame.keys()) != {0, 1, 2}:
                continue
            los_list = [by_frame[0], by_frame[1], by_frame[2]]
            out = compute_rtn_state_for_sat(
                sat_id, [0, 1, 2], los_list, obs_pos, obs_vel, times,
            )
            if out is not None:
                r_rtn, v_rtn = out
                t_latest = times[2]  # state is at latest (third) image
                self._states[sat_id] = RTNState(
                    position_rtn=r_rtn,
                    velocity_rtn=v_rtn,
                    timestamp=t_latest,
                )
                current_ids.add(sat_id)

        # Remove sats that are not in the current 3-frame set
        to_remove = [sid for sid in self._states if sid not in current_ids]
        for sid in to_remove:
            del self._states[sid]

    @property
    def states(self) -> Dict[int, RTNState]:
        """Current RTN state per satellite ID."""
        return dict(self._states)

    @property
    def sat_ids(self) -> List[int]:
        """All satellite IDs that currently have an HCW state (visible in the latest 3 frames)."""
        return sorted(self._states.keys())

    def get_state(self, sat_id: int) -> Optional[RTNState]:
        return self._states.get(sat_id)


def compute_hcw_vectors_from_three_frames(
    frame1: np.ndarray,
    frame2: np.ndarray,
    frame3: np.ndarray,
    t1: float,
    t2: float,
    t3: float,
    observer_pos_eci: np.ndarray,
    observer_vel_eci: np.ndarray,
    R_cam_to_eci_per_frame: Optional[List[np.ndarray]] = None,
    use_image_coords: bool = False,
    fov_deg: float = SENSOR_FOV_DEG,
    return_array: bool = False,
) -> Union[Dict[int, np.ndarray], Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray]]:
    """
    Run the 3-frame pipeline and return HCW 6-vectors per sat ID.
    State (position and velocity) is at the latest image time (t3).

    Parameters
    ----------
    frame1, frame2, frame3 : ndarray shape (N, 3)
        Each row is [sat_id, u, v]. (u, v) are on the normalized plane unless use_image_coords=True.
    t1, t2, t3 : float
        Timestamps (e.g. seconds).
    observer_pos_eci : ndarray shape (3, 3)
        Observer position in ECI at t1, t2, t3 (rows).
    observer_vel_eci : ndarray shape (3, 3)
        Observer velocity in ECI at t1, t2, t3 (rows).
    R_cam_to_eci_per_frame : list of 3 ndarrays (3, 3), optional
        Camera-to-ECI rotation at each time. If None, uses camera_rotation_from_observer_eci.
    use_image_coords : bool
        If True, (u, v) in frames are image coords in [-1, 1]; converted via image_coords_to_uv.
    fov_deg : float
        Used only when use_image_coords=True.
    return_array : bool
        If True, also return (sat_ids, hcw_array) where hcw_array is (n_sats, 6) in same order.

    Returns
    -------
    hcw_by_id : dict
        sat_id -> ndarray (6,) [R, T, N, R_dot, T_dot, N_dot] in km and km/s.
    If return_array=True: (hcw_by_id, sat_ids, hcw_array) with hcw_array shape (n_sats, 6).

    Example
    -------
    >>> frame1 = np.array([[1, 0.1, 0], [2, -0.2, 0.1]])   # sat 1 and 2, (u,v)
    >>> frame2 = np.array([[1, 0.12, 0], [2, -0.18, 0.12]])
    >>> frame3 = np.array([[1, 0.14, 0], [2, -0.16, 0.14]])
    >>> obs_pos = np.stack([pos_t1, pos_t2, pos_t3])   # (3, 3) ECI km
    >>> obs_vel = np.stack([vel_t1, vel_t2, vel_t3])   # (3, 3) ECI km/s
    >>> hcw_by_id = compute_hcw_vectors_from_three_frames(
    ...     frame1, frame2, frame3, t1, t2, t3, obs_pos, obs_vel
    ... )
    >>> hcw_by_id[1]   # 6-vector [R, T, N, R_dot, T_dot, N_dot] for sat 1
    >>> # Optional: get array (n_sats, 6) with sorted sat IDs:
    >>> hcw_by_id, sat_ids, hcw_array = compute_hcw_vectors_from_three_frames(
    ...     frame1, frame2, frame3, t1, t2, t3, obs_pos, obs_vel, return_array=True
    ... )
    """
    obs_pos = np.asarray(observer_pos_eci, dtype=float).reshape(3, 3)
    obs_vel = np.asarray(observer_vel_eci, dtype=float).reshape(3, 3)

    if R_cam_to_eci_per_frame is None:
        R_cam_to_eci_per_frame = [
            camera_rotation_from_observer_eci(obs_pos[i], obs_vel[i])
            for i in range(3)
        ]

    def _prepare_frame(frame: np.ndarray) -> np.ndarray:
        out = np.asarray(frame, dtype=float)
        if out.ndim == 1:
            out = out.reshape(1, -1)
        if use_image_coords and out.size > 0:
            rows = []
            for row in out:
                sid, u, v = row[0], row[1], row[2]
                u_plane, v_plane = image_coords_to_uv(u, v, fov_deg=fov_deg, clip=True)
                rows.append([sid, u_plane, v_plane])
            out = np.array(rows, dtype=float)
        return out

    f1 = _prepare_frame(frame1)
    f2 = _prepare_frame(frame2)
    f3 = _prepare_frame(frame3)

    flt = SlidingFilter()
    flt.push(f1, t1, obs_pos[0], obs_vel[0], R_cam_to_eci_per_frame[0])
    flt.push(f2, t2, obs_pos[1], obs_vel[1], R_cam_to_eci_per_frame[1])
    states = flt.push(f3, t3, obs_pos[2], obs_vel[2], R_cam_to_eci_per_frame[2])

    hcw_by_id: Dict[int, np.ndarray] = {}
    for sid, s in states.items():
        hcw_by_id[sid] = np.concatenate([s.position_rtn, s.velocity_rtn])

    if not return_array:
        return hcw_by_id

    sat_ids = sorted(hcw_by_id.keys())
    hcw_array = np.array([hcw_by_id[sid] for sid in sat_ids], dtype=float)
    return hcw_by_id, np.array(sat_ids), hcw_array


# -----------------------------------------------------------------------------
# 8) HELPERS: build R_cam_to_eci from observer orbit (e.g. camera along velocity)
# -----------------------------------------------------------------------------

def camera_rotation_from_observer_eci(
    observer_pos_eci: np.ndarray,
    observer_vel_eci: np.ndarray,
    boresight_along: str = "minus_T",
) -> np.ndarray:
    """
    Camera frame to ECI. Columns = (cam_x, cam_y, cam_z) in ECI.

    Default: boresight = -T (camera points opposite to velocity / minus along-track).
    So (u=0, v=0) is along -T; (u=-1, v=0) is -5° azimuth from -T (10° FOV, ±5°).

    Options:
      "minus_T" or "-T" : cam_z = -T (boresight backward along-track), cam_y = R, cam_x = N.
      "T" or "velocity" : cam_z = +T (boresight forward), cam_y = R, cam_x = -N.
      "R" : cam_z = R (boresight radial), cam_y = N, cam_x = T.
    """
    basis = eci_to_rtn_basis(observer_pos_eci, observer_vel_eci)
    R, T, N = basis[0], basis[1], basis[2]
    if boresight_along in ("minus_T", "-T"):
        cam_z_eci = -T
        cam_y_eci = R
        cam_x_eci = N
    elif boresight_along in ("velocity", "T"):
        cam_z_eci = T
        cam_y_eci = R
        cam_x_eci = -N
    else:
        cam_z_eci = R
        cam_y_eci = N
        cam_x_eci = T
    return np.column_stack([cam_x_eci, cam_y_eci, cam_z_eci])


def cv_detections_to_nx3(
    detections: List[dict],
    track_ids: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Convert CV pipeline output to Nx3 matrix [satellite_ID, x_pixel, y_pixel].

    detections: list of dicts from detect_single(), each with 'bbox_xywh' [x_center, y_center, w, h]
                or 'bbox_xyxy' [x1, y1, x2, y2] (center is then computed).
    track_ids: optional list of track IDs (same length as detections); if None, uses 0,1,2,...
    """
    rows = []
    for i, d in enumerate(detections):
        if "bbox_xywh" in d:
            x_center, y_center = float(d["bbox_xywh"][0]), float(d["bbox_xywh"][1])
        elif "bbox_xyxy" in d:
            xy = d["bbox_xyxy"]
            x_center = (xy[0] + xy[2]) / 2.0
            y_center = (xy[1] + xy[3]) / 2.0
        else:
            continue
        sid = track_ids[i] if track_ids is not None and i < len(track_ids) else i
        rows.append([sid, x_center, y_center])
    return np.array(rows, dtype=float) if rows else np.empty((0, 3))
