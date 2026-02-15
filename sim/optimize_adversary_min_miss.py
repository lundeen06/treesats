"""
Optimize adversary orbital parameters so that:
- At t=0 the adversary is at least 20 km from the chief (minimum start separation).
- Over the first half orbit the miss distance (minimum separation) is less than 1 km.

Chief from params (SATELLITE). Adversary altitude is constrained to 20–40 km above
or below the chief. Uses SLSQP with constraint sep(t=0) >= 20 km and objective
minimize min_miss over [0, T_half]. Multiple starts are used to find a solution.

Run from project root:
  python sim/optimize_adversary_min_miss.py
  python sim/optimize_adversary_min_miss.py --backend cuda
"""

import numpy as np
import sys
from pathlib import Path

# Ensure sim and project root are on path
SIM_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SIM_DIR.parent
sys.path.insert(0, str(SIM_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from params import SATELLITE, CONSTANTS

import tensorgator as tg


def chief_kepler_from_params():
    """Chief Kepler elements from params (SATELLITE). Returns (6,) in Tensorgator format: [a(m), e, inc, Omega, omega, M] in rad."""
    Re_m = CONSTANTS["earth_radius_m"]
    alt_km = float(SATELLITE["altitude"])
    a_m = Re_m + alt_km * 1000.0
    e = float(SATELLITE["eccentricity"])
    i = np.radians(float(SATELLITE["inclination"]))
    omega = np.radians(float(SATELLITE["omega"]))
    Omega = np.radians(float(SATELLITE["Omega"]))
    M = np.radians(float(SATELLITE["M"]))
    return np.array([a_m, e, i, Omega, omega, M], dtype=float)


def half_orbit_period_seconds(a_m):
    """Half orbital period in seconds. a_m in meters."""
    mu = CONSTANTS["mu"]
    T_full = 2 * np.pi * np.sqrt((a_m ** 3) / mu)
    return T_full / 2.0


def adversary_kepler(alt_km, e, i_deg, omega_deg, Omega_deg, M_deg):
    """Adversary Kepler elements. Returns (6,) in Tensorgator format: [a(m), e, inc, Omega, omega, M] in rad."""
    Re_m = CONSTANTS["earth_radius_m"]
    a_m = Re_m + alt_km * 1000.0
    i = np.radians(float(i_deg))
    Omega = np.radians(float(Omega_deg))
    omega = np.radians(float(omega_deg))
    M = np.radians(float(M_deg))
    return np.array([a_m, float(e), i, Omega, omega, M], dtype=float)


def min_miss_and_sep_t0(adversary_params, chief_kepler, time_array, backend="cpu"):
    """
    Minimum separation over the time array and separation at t=0.

    adversary_params : array_like
        [alt_km, e, omega_deg, M_deg]. Inclination and Omega are taken from chief.
    chief_kepler : (6,) in Tensorgator format
    time_array : 1D, seconds (must include 0; first point is t=0)
    backend : 'cpu' or 'cuda'

    Returns
    -------
    min_dist_km : float
        Minimum distance between the two satellites over the trajectory (km).
    sep_t0_km : float
        Distance between chief and adversary at t=0 (km).
    """
    alt_km, e, omega_deg, M_deg = adversary_params
    i_chief_deg = np.degrees(chief_kepler[2])    # inc
    Omega_chief_deg = np.degrees(chief_kepler[3])  # RAAN
    adv_kepler = adversary_kepler(alt_km, e, i_chief_deg, omega_deg, Omega_chief_deg, M_deg)

    constellation = np.stack([chief_kepler, adv_kepler], axis=0)

    # Tensorgator: (n_sats, n_times, 3) in meters
    pos = tg.satellite_positions(
        time_array,
        constellation,
        backend=backend,
        return_frame="eci",
        input_type="kepler",
    )
    # pos shape: (2, n_times, 3) in m
    diff = pos[0] - pos[1]
    distances_m = np.sqrt(np.sum(diff ** 2, axis=1))
    min_dist_km = float(np.min(distances_m)) / 1000.0
    sep_t0_km = float(distances_m[0]) / 1000.0
    return min_dist_km, sep_t0_km


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Optimize adversary orbit for minimum miss distance over half orbit")
    parser.add_argument("--backend", default="cpu", choices=["cpu", "cuda"], help="Tensorgator backend")
    parser.add_argument("--nsteps", type=int, default=200, help="Number of time steps in half orbit")
    parser.add_argument("--alt-tol", type=float, default=40.0, help="Altitude constraint: adversary within ± this (km) of chief")
    parser.add_argument("--min-alt-diff", type=float, default=20.0, help="Minimum altitude difference (km) between adversary and chief")
    args = parser.parse_args()

    chief_kepler = chief_kepler_from_params()
    chief_alt_km = (chief_kepler[0] - CONSTANTS["earth_radius_m"]) / 1000.0
    T_half = half_orbit_period_seconds(chief_kepler[0])
    time_array = np.linspace(0, T_half, args.nsteps)

    min_diff = args.min_alt_diff
    # Feasible altitude bands: at least min_diff km from chief
    alt_below = (chief_alt_km - args.alt_tol, chief_alt_km - min_diff)  # adversary below chief
    alt_above = (chief_alt_km + min_diff, chief_alt_km + args.alt_tol)  # adversary above chief

    print("Chief (from params):")
    print(f"  altitude = {chief_alt_km:.2f} km")
    print(f"  half-orbit period = {T_half:.1f} s ({T_half/60:.1f} min)")
    print(f"  Adversary altitude: within {args.alt_tol} km of chief, and at least {min_diff} km difference")
    print(f"  Feasible bands: [{alt_below[0]:.2f}, {alt_below[1]:.2f}] (below) and [{alt_above[0]:.2f}, {alt_above[1]:.2f}] (above) km")
    print()

    # Objective: minimize min_miss over half orbit, subject to separation at t=0 >= min_start_sep_km
    from scipy.optimize import minimize

    min_start_sep_km = args.min_alt_diff  # 20 km minimum separation at t=0

    def objective(x):
        # x = [alt_km, e, omega_deg, M_deg]
        min_miss, _ = min_miss_and_sep_t0(x, chief_kepler, time_array, backend=args.backend)
        return min_miss

    def constraint_sep_t0(x):
        """Must be >= 0: separation at t=0 minus min_start_sep_km."""
        _, sep_t0 = min_miss_and_sep_t0(x, chief_kepler, time_array, backend=args.backend)
        return sep_t0 - min_start_sep_km

    constraints = [{"type": "ineq", "fun": constraint_sep_t0}]

    # Run optimization in each altitude band and keep the better result.
    # Allow enough eccentricity so the orbit can cross chief altitude (enables close approach).
    bounds_template = [
        (0.0, 0.05),      # e (e.g. e=0.03 gives r from a(1-e) to a(1+e); need to cross chief)
        (0.0, 360.0),     # omega_deg
        (0.0, 360.0),     # M_deg
    ]

    result_best = None
    rng = np.random.default_rng(42)
    for label, (alt_lo, alt_hi) in [("below chief", alt_below), ("above chief", alt_above)]:
        bounds = [(alt_lo, alt_hi)] + bounds_template
        # Multiple starts to find a trajectory that gets close (< 1 km) while keeping sep(t=0) >= 20 km
        for attempt, (om_start, M_start) in enumerate([
            (180.0, 180.0),
            (0.0, 90.0),
            (90.0, 270.0),
            (270.0, 45.0),
            (rng.uniform(0, 360), rng.uniform(0, 360)),
        ]):
            x0 = [
                (alt_lo + alt_hi) / 2.0,
                0.005,
                om_start,
                M_start,
            ]
            print(f"Running optimization (adversary {label}, sep(t=0) >= {min_start_sep_km} km, start {attempt + 1}/5)...")
            result = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 400, "ftol": 1e-9},
            )
            _, sep0 = min_miss_and_sep_t0(result.x, chief_kepler, time_array, backend=args.backend)
            feasible = sep0 >= min_start_sep_km - 0.01  # allow small numerical tolerance
            print(f"  sep(t=0) = {sep0:.4f} km, min miss = {result.fun:.4f} km, feasible = {feasible}")
            if feasible and (result_best is None or result.fun < result_best.fun):
                result_best = result
                print(f"  -> new best min miss = {result.fun:.4f} km")
        print()

    result = result_best
    if result is None:
        print("No feasible solution found (sep(t=0) >= 20 km). Try relaxing --min-alt-diff or --alt-tol.")
        return None
    if not result.success:
        print(f"Optimization message: {result.message}")

    alt_opt, e_opt, omega_opt, M_opt = result.x
    min_miss_opt = result.fun
    _, sep_t0_opt = min_miss_and_sep_t0(result.x, chief_kepler, time_array, backend=args.backend)
    alt_diff = abs(alt_opt - chief_alt_km)

    # Semi-major axis for reporting (km)
    Re_km = CONSTANTS["earth_radius_km"]
    a_opt_km = Re_km + alt_opt

    print()
    print("Optimized adversary orbital parameters (minimize miss distance):")
    print(f"  separation at t=0  = {sep_t0_opt:.4f} km  (required >= {min_start_sep_km} km)")
    print(f"  min miss (half orbit) = {min_miss_opt:.4f} km  (target < 1 km)")
    print(f"  altitude     = {alt_opt:.4f} km  (|diff from chief| = {alt_diff:.4f} km)")
    print(f"  semi_major_axis = {a_opt_km:.4f} km")
    print(f"  eccentricity = {e_opt:.6f}")
    print(f"  omega (deg)  = {omega_opt:.4f}")
    print(f"  M (deg)      = {M_opt:.4f}")
    print()
    print("Suggested ADVERSARY_TRANSFER for params.py:")
    print(f"  ADVERSARY_TRANSFER = {{")
    print(f"      'altitude': {alt_opt:.4f},")
    print(f"      'semi_major_axis': {a_opt_km:.4f},")
    print(f"      'eccentricity': {e_opt:.6f},")
    print(f"      'inclination': {float(SATELLITE['inclination'])},")
    print(f"      'omega': {omega_opt:.4f},")
    print(f"      'Omega': {float(SATELLITE['Omega'])},")
    print(f"      'M': {M_opt:.4f},")
    print(f"  }}")
    if sep_t0_opt >= min_start_sep_km - 0.01 and min_miss_opt < 1.0:
        print(f"Goal met: sep(t=0) >= {min_start_sep_km} km and min miss < 1 km over half orbit.")
    return result


if __name__ == "__main__":
    main()
