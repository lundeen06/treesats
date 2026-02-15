"""
Test collision avoidance: chief from params (SATELLITE, e=0), adversarial
satellite from ADVERSARY_TRANSFER. Uses only public API from sat.control
and parameters from sim.params.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))
from params import SATELLITE, CONSTANTS, ADVERSARY_TRANSFER

from sat.control.rtn_to_eci_propagate import (
    propagate_chief,
    propagate_constellation,
    eci_to_relative_rtn,
    kepler_to_rv,
)
from sat.control.collision_avoidance import (
    apply_impulse_eci,
    detect_near_miss,
    collision_avoidance_delta_v,
    DEFAULT_RADIUS_KM,
)


def _chief_initial_state_from_params():
    """Chief ECI state from params (SATELLITE with e=0, altitude -> semi-major axis)."""
    earth_km = CONSTANTS["earth_radius_km"]
    mu_km = CONSTANTS["mu"] / 1e9
    a = earth_km + float(SATELLITE["altitude"])
    e = float(SATELLITE["eccentricity"])
    i = np.radians(float(SATELLITE["inclination"]))
    omega = np.radians(float(SATELLITE["omega"]))
    Omega = np.radians(float(SATELLITE["Omega"]))
    M = np.radians(float(SATELLITE["M"]))
    r_chief, v_chief = kepler_to_rv(a, e, i, omega, Omega, M, mu=mu_km)
    return r_chief, v_chief


def _adversary_initial_state_from_params():
    """Adversary ECI state from ADVERSARY_TRANSFER (semi_major_axis in km, angles in degrees)."""
    mu_km = CONSTANTS["mu"] / 1e9
    a = float(ADVERSARY_TRANSFER["semi_major_axis"])
    e = float(ADVERSARY_TRANSFER["eccentricity"])
    i = np.radians(float(ADVERSARY_TRANSFER["inclination"]))
    omega = np.radians(float(ADVERSARY_TRANSFER["omega"]))
    Omega = np.radians(float(ADVERSARY_TRANSFER["Omega"]))
    M = np.radians(float(ADVERSARY_TRANSFER["M"]))
    r_adv, v_adv = kepler_to_rv(a, e, i, omega, Omega, M, mu=mu_km)
    return r_adv, v_adv


def test_collision_avoidance_adversary_transfer():
    """Chief from params (e=0); adversarial satellite from ADVERSARY_TRANSFER."""
    import sys as _sys
    _sys.stdout.flush()
    print("\n" + "=" * 60)
    print("  COLLISION AVOIDANCE TEST RESULTS (Adversary Transfer)")
    print("=" * 60)

    r_chief, v_chief = _chief_initial_state_from_params()
    r_adv, v_adv = _adversary_initial_state_from_params()
    constellation_rtn = eci_to_relative_rtn(r_chief, v_chief, r_adv, v_adv)
    constellation_rtn = np.asarray(constellation_rtn, dtype=float).reshape(1, 6)

    earth_km = CONSTANTS["earth_radius_km"]
    mu_km = CONSTANTS["mu"] / 1e9
    alt_chief = float(SATELLITE["altitude"])
    T_orbit = 2 * np.pi * np.sqrt(((earth_km + alt_chief) ** 3) / mu_km)
    n_times = max(120, int(T_orbit / 10.0))
    time_array = np.linspace(0, T_orbit, n_times)

    pos_chief = propagate_chief(r_chief, v_chief, time_array, backend="cpu")
    pos_adv = propagate_constellation(
        r_chief, v_chief, constellation_rtn, time_array, backend="cpu", return_velocity=False
    )
    if pos_adv.ndim == 2:
        pos_adv = pos_adv[:, np.newaxis, :]
    has_near, min_dist, t_idx, k_idx = detect_near_miss(
        pos_chief, pos_adv, radius_km=DEFAULT_RADIUS_KM
    )
    t_closest_s = time_array[t_idx] if (t_idx is not None and t_idx >= 0) else None

    print(f"  Separation at t=0:   {float(np.linalg.norm(pos_chief[0] - pos_adv[0, 0, :])):.4f} km")
    print(f"  Near miss:           {has_near}")
    print(f"  Min separation:      {min_dist:.4f} km  (radius = {DEFAULT_RADIUS_KM} km)")
    if t_closest_s is not None and t_idx >= 0:
        print(f"  Closest approach:    t = {t_closest_s:.0f} s")
    _sys.stdout.flush()

    # Use enough time steps so the solver sees the same near miss as the test (coarse grid can miss it)
    n_times_orbit = max(200, n_times)
    delta_v = collision_avoidance_delta_v(
        r_chief, v_chief, constellation_rtn,
        backend="cpu",
        radius_km=DEFAULT_RADIUS_KM,
        n_times_orbit=n_times_orbit,
        verbose=has_near,  # debug when we expect non-zero delta_v
    )
    dv_norm = np.linalg.norm(delta_v)
    print(f"  Delta-v:              {dv_norm * 1000:.4f} m/s")
    _sys.stdout.flush()

    if has_near:
        assert dv_norm > 1e-6, "Expected non-zero delta-v when adversary is within radius_km"
        r_chief_evaded, v_chief_evaded = apply_impulse_eci(r_chief, v_chief, delta_v)
        pos_chief_evaded = propagate_chief(r_chief_evaded, v_chief_evaded, time_array, backend="cpu")
        pos_adv_evaded = propagate_constellation(
            r_chief, v_chief, constellation_rtn, time_array, backend="cpu", return_velocity=False
        )
        if pos_adv_evaded.ndim == 2:
            pos_adv_evaded = pos_adv_evaded[:, np.newaxis, :]
        has_near_after, min_dist_after, _, _ = detect_near_miss(
            pos_chief_evaded, pos_adv_evaded, radius_km=DEFAULT_RADIUS_KM
        )
        print(f"  After evasion:        min sep = {min_dist_after:.4f} km")
        _sys.stdout.flush()
        assert not has_near_after, f"After evasion, min distance was {min_dist_after:.4f} km"
    else:
        print(f"  No maneuver needed.")
        _sys.stdout.flush()
        assert dv_norm < 1e-6, "Expected zero delta-v when no near miss (min_dist=%s km)" % (min_dist,)

    print("=" * 60 + "\n")
    _sys.stdout.flush()


def _run_visualization():
    """Generate and show adversary scenario plot (and optional GIF)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))
    from params import PATHS
    from plots import plot_adversary_transfer_scenario
    import matplotlib.pyplot as plt
    import os

    plots_dir = PATHS.get("plots_dir")
    anim_path = os.path.join(plots_dir, "adversary_transfer_scenario.gif") if plots_dir else None
    anim_rtn_path = os.path.join(plots_dir, "adversary_transfer_scenario_rtn.gif") if plots_dir else None

    fig, ax = plot_adversary_transfer_scenario(
        show_evasion=True,
        save_animation_path=anim_path,
        save_animation_rtn_path=anim_rtn_path,
    )
    plt.show()
    if anim_path and os.path.isfile(anim_path):
        try:
            os.startfile(anim_path)  # Windows: open in default viewer
        except (AttributeError, OSError):
            import webbrowser
            webbrowser.open("file:///" + os.path.abspath(anim_path).replace("\\", "/"))
        print("Animation opened (see GIF for evolution in time):", anim_path)


if __name__ == "__main__":
    do_visualize = "--visualize" in sys.argv or "-v" in sys.argv
    failed = False
    try:
        test_collision_avoidance_adversary_transfer()
        print("PASS: collision avoidance (chief from params e=0, adversary from ADVERSARY_TRANSFER)")
    except Exception as e:
        failed = True
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
    if do_visualize:
        print("\nGenerating plot (--visualize)...")
        _run_visualization()
    if failed:
        sys.exit(1)
