"""
Test collision avoidance: use only functions from sat.control.collision_avoidance
and sat.control.rtn_to_eci_propagate. No helper functions; hardcoded scenario.

Scenario: our spacecraft (chief) in 550 km circular orbit, i=53° (match simulate.py).
One adversary 0.5 km ahead in track → nominal near miss. Test that we detect it,
get a non-zero delta-v, apply it, propagate, and verify min separation >= 1 km.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sat.control.rtn_to_eci_propagate import propagate_chief, propagate_constellation
from sat.control.collision_avoidance import (
    apply_impulse_eci,
    detect_near_miss,
    collision_avoidance_delta_v,
    DEFAULT_RADIUS_KM,
)


def test_collision_avoidance_detection_and_evasion():
    # Chief: 550 km circular, i=53°, at ascending node (M=0). ECI [km], [km/s].
    # r = a = 6371+550 = 6921, v = sqrt(mu/a) ≈ 7.612
    r_chief = np.array([6921.0, 0.0, 0.0])
    v_chief = np.array([0.0, 4.58, 6.07])  # v*cos(53°), v*sin(53°), v≈7.61

    # One deputy: 0.5 km ahead in track → nominal separation < 1 km
    constellation_rtn = np.array([[0.0, 0.5, 0.0, 0.0, 0.0, 0.0]])

    # 1) Collision avoidance should return a non-zero delta-v
    delta_v = collision_avoidance_delta_v(
        r_chief, v_chief, constellation_rtn,
        backend="cpu",
        radius_km=DEFAULT_RADIUS_KM,
        n_times_orbit=80,
    )
    assert np.linalg.norm(delta_v) > 1e-6, (
        "Expected non-zero delta-v when deputy is within 1 km"
    )

    # 2) Apply impulse and propagate chief and constellation one orbit
    r_chief_evaded, v_chief_evaded = apply_impulse_eci(r_chief, v_chief, delta_v)
    T_orbit = 2 * np.pi * np.sqrt((6921.0 ** 3) / 398600.4418)
    time_array = np.linspace(0, T_orbit, 100)

    pos_chief = propagate_chief(r_chief_evaded, v_chief_evaded, time_array, backend="cpu")
    pos_constellation = propagate_constellation(
        r_chief, v_chief, constellation_rtn, time_array, backend="cpu", return_velocity=False
    )
    if pos_constellation.ndim == 2:
        pos_constellation = pos_constellation[:, np.newaxis, :]

    # 3) No near miss after evasion
    has_near, min_dist, _, _ = detect_near_miss(
        pos_chief, pos_constellation, radius_km=DEFAULT_RADIUS_KM
    )
    assert not has_near, (
        f"After evasion, min distance was {min_dist:.4f} km (required >= {DEFAULT_RADIUS_KM} km)"
    )
    assert min_dist >= DEFAULT_RADIUS_KM


if __name__ == "__main__":
    test_collision_avoidance_detection_and_evasion()
    print("PASS: collision avoidance detection and evasion")
