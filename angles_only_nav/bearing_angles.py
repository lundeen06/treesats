"""
Angles-only navigation: bearing angles, triangulation, and HCW RTN state from images.

CV pipeline output per image: Nx3 array with rows [satellite_ID, x_pixel, y_pixel].
Uses a sliding window of 3 images: image 1 → bearing, image 2 → depth (triangulation),
image 3 → velocity (finite difference). Produces HCW state (position, velocity) in RTN.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field


# -----------------------------------------------------------------------------
# 1) PIXEL → BEARING (unit line-of-sight in camera frame)
# -----------------------------------------------------------------------------
# Pinhole model: ray through (x_pixel, y_pixel) has direction proportional to
#   (x_pixel - cx, cy - y_pixel, f)  [image y is down, camera y is up]
# Unit LOS: ℓ_cam = (dx, dy, f) / sqrt(dx^2 + dy^2 + f^2)
# where dx = x_pixel - center_x, dy = center_y - y_pixel, f = focal_length_px.
# Focal length: f = (image_size/2) / tan(half_fov_rad)  (same as star_tracker).
# -----------------------------------------------------------------------------

def focal_length_px(image_size: int, fov_deg: float) -> float:
    """Focal length in pixels from image size and full FOV (degrees)."""
    half_fov_rad = np.radians(fov_deg / 2.0)
    return (image_size / 2.0) / np.tan(half_fov_rad)


def pixel_to_bearing(
    x_pixel: Union[float, np.ndarray],
    y_pixel: Union[float, np.ndarray],
    image_size: int,
    fov_deg: float = 10.0,
    center_x: Optional[float] = None,
    center_y: Optional[float] = None,
) -> np.ndarray:
    """
    Convert pixel coordinates to unit line-of-sight vector in camera frame.

    Camera frame: x = right, y = up, z = forward (boresight).
    Image: (x_pixel, y_pixel) = (column, row); origin top-left, y down.

    Parameters
    ----------
    x_pixel, y_pixel : float or array
        Pixel coordinates (column, row). Can be arrays of shape (n,) for multiple points.
    image_size : int
        Image width/height in pixels (e.g. 256).
    fov_deg : float
        Full field of view in degrees.
    center_x, center_y : float, optional
        Principal point. Default: image_size/2.

    Returns
    -------
    los_cam : ndarray
        Unit vector(s) in camera frame, shape (3,) or (n, 3). Each row is (x_cam, y_cam, z_cam).
    """
    cx = image_size / 2.0 if center_x is None else center_x
    cy = image_size / 2.0 if center_y is None else center_y
    f = focal_length_px(image_size, fov_deg)

    x = np.atleast_1d(np.asarray(x_pixel, dtype=float))
    y = np.atleast_1d(np.asarray(y_pixel, dtype=float))
    dx = x - cx
    dy = cy - y  # image y down → camera y up

    # Unnormalized ray: (dx, dy, f)
    z = np.full_like(dx, f)
    ray = np.stack([dx, dy, z], axis=-1)
    nrm = np.linalg.norm(ray, axis=-1, keepdims=True)
    nrm = np.where(nrm > 0, nrm, 1.0)
    los_cam = ray / nrm

    return los_cam.squeeze() if los_cam.shape[0] == 1 else los_cam


# -----------------------------------------------------------------------------
# 2) ECI ↔ RTN (Hill frame)
# -----------------------------------------------------------------------------
# RTN at observer: R = radial (from Earth outward), T = along-track, N = cross-track.
#   R = r / |r|
#   N = (r × v) / |r × v|
#   T = N × R
# So R_eci_to_rtn has rows (R, T, N) in ECI components; v_rtn = R_eci_to_rtn @ v_eci.
# -----------------------------------------------------------------------------

def eci_to_rtn_basis(pos_eci: np.ndarray, vel_eci: np.ndarray) -> np.ndarray:
    """
    Compute RTN basis vectors (rows) from observer position and velocity in ECI.

    Parameters
    ----------
    pos_eci, vel_eci : ndarray shape (3,)
        Observer position and velocity in ECI (e.g. km, km/s).

    Returns
    -------
    basis_rtn : ndarray shape (3, 3)
        Rows are R, T, N unit vectors in ECI components. So v_rtn = basis_rtn @ v_eci.
    """
    r = np.asarray(pos_eci, dtype=float).ravel()
    v = np.asarray(vel_eci, dtype=float).ravel()
    R = r / np.linalg.norm(r)
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    if hn < 1e-12:
        # Degenerate: use arbitrary N perpendicular to R
        if abs(R[2]) < 0.9:
            N = np.cross(R, np.array([0, 0, 1.0]))
        else:
            N = np.cross(R, np.array([1.0, 0, 0]))
        N = N / np.linalg.norm(N)
    else:
        N = h / hn
    T = np.cross(N, R)
    return np.stack([R, T, N], axis=0)


def vector_eci_to_rtn(vec_eci: np.ndarray, basis_rtn: np.ndarray) -> np.ndarray:
    """Transform vector(s) from ECI to RTN using precomputed basis (rows R,T,N)."""
    v = np.asarray(vec_eci, dtype=float)
    if v.ndim == 1:
        return np.dot(basis_rtn, v)
    return np.dot(vec_eci, basis_rtn.T)


def vector_rtn_to_eci(vec_rtn: np.ndarray, basis_rtn: np.ndarray) -> np.ndarray:
    """Transform vector(s) from RTN to ECI (basis rows are R,T,N in ECI)."""
    v = np.asarray(vec_rtn, dtype=float)
    if v.ndim == 1:
        return np.dot(basis_rtn.T, v)
    return np.dot(vec_rtn, basis_rtn)


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


# -----------------------------------------------------------------------------
# 5) THREE-FRAME STATE: positions at t1,t2,t3 and velocity at t2
# -----------------------------------------------------------------------------
# For each satellite we have 3 unit LOS in ECI and 3 observer states.
# Triangulate (1,2) to get ρ1, ρ2; (2,3) to get ρ2b, ρ3; (1,3) to get ρ1c, ρ3c.
# We use pairs (1,2) and (2,3) to get ranges at t1, t2, t3:
#   ρ1 from (los1, los2, O1, O2), then r1_rtn = rel_pos_rtn(ρ1, los1, O1, v1).
#   ρ2 from (los1, los2, O1, O2) gives ρ' at O2; or use (los2, los3, O2, O3) for ρ2.
#   ρ3 from (los2, los3, O2, O3).
# Then velocity at t2: v_rtn ≈ (r3_rtn - r1_rtn) / (t3 - t1).
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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute relative position (RTN) at t2 and velocity (RTN) at t2 from 3 bearings and observer orbit.

    Parameters
    ----------
    los1_eci, los2_eci, los3_eci : ndarray (3,)
        Unit line-of-sight in ECI at t1, t2, t3.
    obs_pos_eci : ndarray (3, 3)
        Observer position in ECI at t1, t2, t3 (rows).
    obs_vel_eci : ndarray (3, 3)
        Observer velocity in ECI at t1, t2, t3 (rows).
    t1, t2, t3 : float
        Timestamps (e.g. seconds).

    Returns
    -------
    r_rtn : ndarray (3,) position in RTN at t2 (km)
    v_rtn : ndarray (3,) velocity in RTN at t2 (km/s)
    """
    O1 = obs_pos_eci[0]
    O2 = obs_pos_eci[1]
    O3 = obs_pos_eci[2]
    v2 = obs_vel_eci[1]

    rho1, _ = triangulate_range(los1_eci, los2_eci, O1, O2)
    _, rho2_from_12 = triangulate_range(los1_eci, los2_eci, O1, O2)
    rho2_from_23, rho3 = triangulate_range(los2_eci, los3_eci, O2, O3)

    if np.isnan(rho1) or np.isnan(rho3):
        return np.full(3, np.nan), np.full(3, np.nan)

    # Prefer range at t2 from average of both triangulations if both valid
    rho2 = rho2_from_12 if not np.isnan(rho2_from_12) else rho2_from_23
    if not np.isnan(rho2_from_23) and not np.isnan(rho2_from_12):
        rho2 = 0.5 * (rho2_from_12 + rho2_from_23)

    r1_rtn = relative_position_rtn_at_t(rho1, los1_eci, O1, obs_vel_eci[0])
    r2_rtn = relative_position_rtn_at_t(rho2, los2_eci, O2, v2)
    r3_rtn = relative_position_rtn_at_t(rho3, los3_eci, O3, obs_vel_eci[2])

    dt = t3 - t1
    if dt <= 0:
        v_rtn = np.zeros(3)
    else:
        v_rtn = (r3_rtn - r1_rtn) / dt

    return r2_rtn, v_rtn


# -----------------------------------------------------------------------------
# 6) DETECTION MATRICES AND FRAME PROCESSING
# -----------------------------------------------------------------------------
# Each frame: detections = Nx3 array, rows [sat_id, x_pixel, y_pixel].
# We need to build LOS in ECI for each (sat_id, frame_idx) and observer orbit.
# -----------------------------------------------------------------------------

def detections_to_los_eci_per_sat(
    frame_detections: List[np.ndarray],
    image_size: int,
    fov_deg: float,
    R_cam_to_eci_per_frame: List[np.ndarray],
) -> Dict[int, List[np.ndarray]]:
    """
    For each satellite ID present in any frame, build list of (frame_index, los_eci) for frames where it appears.

    frame_detections[i] = Nx3 array [sat_id, x_pixel, y_pixel].
    R_cam_to_eci_per_frame[i] = 3x3 rotation at frame i.
    Returns dict: sat_id -> list of (frame_idx, los_eci) in order of frame index.
    """
    sat_to_frames: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    for fi, det in enumerate(frame_detections):
        if det is None or len(det) == 0:
            continue
        R = R_cam_to_eci_per_frame[fi]
        for row in det:
            sid = int(row[0])
            xp, yp = float(row[1]), float(row[2])
            los_cam = pixel_to_bearing(xp, yp, image_size, fov_deg)
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
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Compute RTN position and velocity for one satellite from 3 consecutive frames.

    frame_indices and los_eci_list must have length 3; obs_pos_eci and obs_vel_eci shape (3,3); times length 3.
    """
    if len(frame_indices) != 3 or len(los_eci_list) != 3:
        return None
    i1, i2, i3 = frame_indices[0], frame_indices[1], frame_indices[2]
    r_rtn, v_rtn = three_frame_rtn_state(
        los_eci_list[0], los_eci_list[1], los_eci_list[2],
        obs_pos_eci, obs_vel_eci,
        times[i1], times[i2], times[i3],
    )
    if np.any(np.isnan(r_rtn)):
        return None
    return (r_rtn, v_rtn)


# -----------------------------------------------------------------------------
# 7) SLIDING FILTER: maintain latest 3 frames, running list of RTN states
# -----------------------------------------------------------------------------

@dataclass
class FrameInput:
    """Single frame: detections Nx3 [sat_id, x_pixel, y_pixel], timestamp, observer state in ECI."""
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
    computes HCW RTN (position, velocity). Adds new IDs when they get 3 frames;
    drops IDs that no longer appear in the latest 3.
    """

    def __init__(
        self,
        image_size: int = 256,
        fov_deg: float = 10.0,
    ):
        self.image_size = image_size
        self.fov_deg = fov_deg
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
        Append a new frame. If we have 3 frames, compute RTN for all sats visible in all 3;
        update running list (add new, remove missing). Returns current state dict.
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
        los_per_sat = detections_to_los_eci_per_sat(
            frame_detections, self.image_size, self.fov_deg, R_per_frame,
        )

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
                t_mid = times[1]
                self._states[sat_id] = RTNState(
                    position_rtn=r_rtn,
                    velocity_rtn=v_rtn,
                    timestamp=t_mid,
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

    def get_state(self, sat_id: int) -> Optional[RTNState]:
        return self._states.get(sat_id)


# -----------------------------------------------------------------------------
# 8) HELPERS: build R_cam_to_eci from observer orbit (e.g. camera along velocity)
# -----------------------------------------------------------------------------

def camera_rotation_from_observer_eci(
    observer_pos_eci: np.ndarray,
    observer_vel_eci: np.ndarray,
    boresight_along: str = "velocity",
) -> np.ndarray:
    """
    Default camera: boresight along velocity (along-track), up along radial (R).
    So in RTN: camera z = T, camera y = R, camera x = -N. Then convert RTN to ECI.
    """
    basis = eci_to_rtn_basis(observer_pos_eci, observer_vel_eci)
    R, T, N = basis[0], basis[1], basis[2]
    if boresight_along == "velocity" or boresight_along == "T":
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
