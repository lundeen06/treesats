"""
Test: one sat ID + image coords (u,v) in [-1, 1] → bearing angles (±5° at ±1), then HCW state.

Image coords: u, v in [-1, 1]; origin at center; ±1 = edge of 10° FOV (±5° from boresight).
So --u 1 --v 0 gives +5° in azimuth from center (camera boresight = -T).

Three images: HCW needs 3 snapshots. We convert image coords to normalized-plane (u,v)
via image_coords_to_uv so the pipeline gets the correct bearing scale.

Run from repo root:  python test/test_bearing_hcw.py
With args:           python test/test_bearing_hcw.py --u 1 --v 0  (→ +5° azimuth)
"""

import argparse
import os
import sys

# Ensure repo root is on path for angles_only_nav and sat
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np

from angles_only_nav import (
    image_coords_to_bearing,
    image_coords_to_uv,
    bearing_to_angles,
    SlidingFilter,
    camera_rotation_from_observer_eci,
    normalized_plane_to_bearing,
    los_cam_to_eci,
    three_frame_rtn_positions,
    rtn_position_to_predicted_image_coords,
    SENSOR_FOV_DEG,
)

MU_EARTH = 398600.4418  # km^3/s^2


def circular_orbit_state(r_km: float, t_s: float, n_rad: float) -> tuple:
    """Position and velocity in ECI for circular equatorial orbit (km, km/s)."""
    x = r_km * np.cos(n_rad * t_s)
    y = r_km * np.sin(n_rad * t_s)
    z = 0.0
    vx = -r_km * n_rad * np.sin(n_rad * t_s)
    vy = r_km * n_rad * np.cos(n_rad * t_s)
    vz = 0.0
    return np.array([x, y, z]), np.array([vx, vy, vz])


def main():
    parser = argparse.ArgumentParser(description="Bearing from image coords [-1,1] (FOV ±5°), then HCW from 3 frames")
    parser.add_argument("--sat-id", type=int, default=1, help="Satellite ID")
    parser.add_argument("--u", type=float, default=0.2, help="First frame image u in [-1, 1]; ±1 = ±5°")
    parser.add_argument("--v", type=float, default=0.0, help="First frame image v in [-1, 1]")
    parser.add_argument("--u2", type=float, default=0.25, help="Second frame image u")
    parser.add_argument("--v2", type=float, default=0.0, help="Second frame image v")
    parser.add_argument("--u3", type=float, default=0.3, help="Third frame image u")
    parser.add_argument("--v3", type=float, default=0.0, help="Third frame image v")
    parser.add_argument("--radius-km", type=float, default=6921.0, help="Observer orbit radius (km)")
    parser.add_argument("--dt-s", type=float, default=60.0, help="Time step between frames (s)")
    parser.add_argument("--plot", action="store_true", help="Plot positions and predicted angles at t1, t2, t3")
    args = parser.parse_args()

    sat_id = args.sat_id
    # Image coords in [-1, 1]; ±1 = edge of FOV = ±5°
    u1_img, v1_img = args.u, args.v
    u2_img, v2_img = args.u2, args.v2
    u3_img, v3_img = args.u3, args.v3

    # ----- 1) Bearing from first image coord (u=1 → +5° azimuth) -----
    b = image_coords_to_bearing(u1_img, v1_img, fov_deg=SENSOR_FOV_DEG, clip=True)
    theta_rad, phi_rad = bearing_to_angles(b, convention="azimuth_elevation")
    theta_deg = np.degrees(theta_rad)
    phi_deg = np.degrees(phi_rad)

    print("Input (image coords in [-1, 1], FOV {} deg so ±1 = ±5°): sat_id = {}, u = {}, v = {}".format(
        int(SENSOR_FOV_DEG), sat_id, u1_img, v1_img))
    print()
    print("Bearing (unit vector in camera frame; boresight = -T):")
    print("  b = [ {:.6f}, {:.6f}, {:.6f} ]".format(float(b[0]), float(b[1]), float(b[2])))
    print("  |b| = {:.6f}".format(float(np.linalg.norm(b))))
    print("Angles from center (azimuth_elevation):")
    print("  theta = {:.6f} rad = {:.4f} deg  (azimuth)".format(theta_rad, theta_deg))
    print("  phi   = {:.6f} rad = {:.4f} deg  (elevation)".format(phi_rad, phi_deg))
    print()

    # ----- 2) HCW state from 3 frames: convert image coords to normalized-plane (u,v) for pipeline -----
    u1, v1 = image_coords_to_uv(u1_img, v1_img, fov_deg=SENSOR_FOV_DEG, clip=True)
    u2, v2 = image_coords_to_uv(u2_img, v2_img, fov_deg=SENSOR_FOV_DEG, clip=True)
    u3, v3 = image_coords_to_uv(u3_img, v3_img, fov_deg=SENSOR_FOV_DEG, clip=True)

    r_km = args.radius_km
    n_rad = np.sqrt(MU_EARTH / (r_km ** 3))
    dt = args.dt_s
    t1, t2, t3 = 0.0, dt, 2.0 * dt

    pos1, vel1 = circular_orbit_state(r_km, t1, n_rad)
    pos2, vel2 = circular_orbit_state(r_km, t2, n_rad)
    pos3, vel3 = circular_orbit_state(r_km, t3, n_rad)

    R_cam1 = camera_rotation_from_observer_eci(pos1, vel1)
    R_cam2 = camera_rotation_from_observer_eci(pos2, vel2)
    R_cam3 = camera_rotation_from_observer_eci(pos3, vel3)

    frame1 = np.array([[sat_id, u1, v1]], dtype=float)
    frame2 = np.array([[sat_id, u2, v2]], dtype=float)
    frame3 = np.array([[sat_id, u3, v3]], dtype=float)

    print("Three images (image coords → normalized u,v for pipeline):")
    print("  Image 1 (t = {:.0f} s):  image ({}, {}) → u = {:.6f}, v = {:.6f}".format(t1, u1_img, v1_img, u1, v1))
    print("  Image 2 (t = {:.0f} s):  image ({}, {}) → u = {:.6f}, v = {:.6f}".format(t2, u2_img, v2_img, u2, v2))
    print("  Image 3 (t = {:.0f} s):  image ({}, {}) → u = {:.6f}, v = {:.6f}".format(t3, u3_img, v3_img, u3, v3))
    print()

    flt = SlidingFilter()
    flt.push(frame1, t1, pos1, vel1, R_cam1)
    flt.push(frame2, t2, pos2, vel2, R_cam2)
    states = flt.push(frame3, t3, pos3, vel3, R_cam3)

    print("HCW state (3 frames: t = 0, {:.0f}, {:.0f} s, observer radius = {:.0f} km)".format(dt, 2 * dt, r_km))
    print("  → State is at the latest image (Image 3, t = {:.0f} s). Position and velocity both referenced to that time.".format(2 * dt))
    print()
    if sat_id in states:
        s = states[sat_id]
        r_rtn = s.position_rtn
        v_rtn = s.velocity_rtn
        print("Satellite ID {}:".format(sat_id))
        print("  Position RTN (km):  [ R = {:.4f},  T = {:.4f},  N = {:.4f} ]".format(r_rtn[0], r_rtn[1], r_rtn[2]))
        print("  Velocity RTN (km/s): [ R_dot = {:.6f},  T_dot = {:.6f},  N_dot = {:.6f} ]".format(v_rtn[0], v_rtn[1], v_rtn[2]))
        print("  HCW state vector (6): position_rtn + velocity_rtn")
        hcw = np.concatenate([r_rtn, v_rtn])
        print("  [ {:.6f}, {:.6f}, {:.6f}, {:.8f}, {:.8f}, {:.8f} ]".format(*hcw))
        print("  (at latest image, timestamp = {:.2f} s)".format(s.timestamp))
    else:
        print("No HCW state for sat_id {} (triangulation may have failed for these (u,v) / geometry).".format(sat_id))

    # N drift: the three LOS directions (even if only u changes) are not coplanar with the orbit
    # in a way that forces the triangulated point P to lie in the orbital plane. P can be off-plane,
    # so (P - O) has an N component that changes with time → N_dot appears.
    if sat_id in states:
        print()
        print("Note: N velocity can appear even with no out-of-plane image motion because the")
        print("triangulated point (where the 3 rays meet) may lie off the observer's orbital plane.")

    # ----- 3) Plot positions and predicted angles at all time steps -----
    if getattr(args, "plot", False) and sat_id in states:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # LOS in ECI at each time
        los1_cam = normalized_plane_to_bearing(u1, v1)
        los2_cam = normalized_plane_to_bearing(u2, v2)
        los3_cam = normalized_plane_to_bearing(u3, v3)
        los1_eci = los_cam_to_eci(los1_cam, R_cam1)
        los2_eci = los_cam_to_eci(los2_cam, R_cam2)
        los3_eci = los_cam_to_eci(los3_cam, R_cam3)
        obs_pos = np.stack([pos1, pos2, pos3])
        obs_vel = np.stack([vel1, vel2, vel3])

        r1_rtn, r2_rtn, r3_rtn = three_frame_rtn_positions(los1_eci, los2_eci, los3_eci, obs_pos, obs_vel)
        if not np.any(np.isnan(r1_rtn)):
            times = np.array([t1, t2, t3])
            positions = np.array([r1_rtn, r2_rtn, r3_rtn])  # (3, 3) R,T,N
            R_cams = [R_cam1, R_cam2, R_cam3]
            u_pred, v_pred = [], []
            for i in range(3):
                u_p, v_p = rtn_position_to_predicted_image_coords(
                    positions[i], obs_pos[i], obs_vel[i], R_cams[i], fov_deg=SENSOR_FOV_DEG
                )
                u_pred.append(u_p)
                v_pred.append(v_p)
            u_pred = np.array(u_pred)
            v_pred = np.array(v_pred)
            u_input = np.array([u1_img, u2_img, u3_img])
            v_input = np.array([v1_img, v2_img, v3_img])

            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            # Position RTN vs time
            ax = axes[0, 0]
            ax.plot(times, positions[:, 0], "r-o", label="R (km)")
            ax.plot(times, positions[:, 1], "g-s", label="T (km)")
            ax.plot(times, positions[:, 2], "b-^", label="N (km)")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Position RTN (km)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title("Relative position at t1, t2, t3")

            # Predicted vs input image u
            ax = axes[0, 1]
            ax.plot(times, u_input, "k-o", label="Input u (image)")
            ax.plot(times, u_pred, "c--x", label="Predicted u from RTN")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("u (image coords)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title("Image u: input vs predicted from triangulated position")

            # Predicted vs input image v
            ax = axes[1, 0]
            ax.plot(times, v_input, "k-o", label="Input v (image)")
            ax.plot(times, v_pred, "c--x", label="Predicted v from RTN")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("v (image coords)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title("Image v: input vs predicted")

            # Position in R-T plane (in-plane) and R-N (out-of-plane)
            ax = axes[1, 1]
            ax.plot(positions[:, 1], positions[:, 0], "g-o", label="R vs T (in-plane)")
            ax.set_xlabel("T (km)")
            ax.set_ylabel("R (km)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title("Position in R-T plane (N = {:.2f}, {:.2f}, {:.2f} km)".format(*positions[:, 2]))
            plt.tight_layout()
            out_path = os.path.join(_root, "test", "bearing_hcw_plot.png")
            plt.savefig(out_path, dpi=120)
            print()
            print("Plot saved to {}".format(out_path))
            plt.close()


if __name__ == "__main__":
    main()
