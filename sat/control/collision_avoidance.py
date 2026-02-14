"""
Collision avoidance for the chief satellite relative to a constellation.

Forward-propagates chief and constellation for one orbit, detects near misses
(< radius_km), and solves a convex QP (minimize delta-v, enforce minimum separation)
to compute an impulsive maneuver. Applies delta-v in ECI.
"""

import numpy as np
import cvxpy as cp

from .rtn_to_eci_propagate import (
    rv_to_kepler,
    propagate_constellation,
    MU_EARTH,
)

# Default near-miss radius [km]
DEFAULT_RADIUS_KM = 1.0
# Constraint margin: add constraints at times where distance < this [km]
CONSTRAINT_MARGIN_KM = 2.0
# Numerical step for Phi_rv (velocity perturbation) [km/s]
FD_STEP_KM_S = 1e-6


def apply_impulse_eci(r_chief, v_chief, delta_v):
    """
    Apply an impulsive delta-v in ECI. Position unchanged; velocity updates immediately.

    Parameters
    ----------
    r_chief : array_like (3,)
        Chief position in ECI [km].
    v_chief : array_like (3,)
        Chief velocity in ECI [km/s].
    delta_v : array_like (3,)
        Impulsive velocity change in ECI [km/s].

    Returns
    -------
    r_chief : ndarray (3,)
        Unchanged (copy of input).
    v_chief_new : ndarray (3,)
        v_chief + delta_v [km/s].
    """
    r_chief = np.asarray(r_chief, dtype=float).reshape(3)
    v_chief = np.asarray(v_chief, dtype=float).reshape(3)
    delta_v = np.asarray(delta_v, dtype=float).reshape(3)
    return r_chief.copy(), (v_chief + delta_v).copy()


def _orbit_period_seconds(r_eci, v_eci, mu=MU_EARTH):
    """Orbit period in seconds from ECI state (km, km/s)."""
    r = np.linalg.norm(r_eci)
    v = np.linalg.norm(v_eci)
    a = 1.0 / (2.0 / r - (v * v) / mu)
    return 2 * np.pi * np.sqrt((a ** 3) / mu)


def _propagate_chief_only(r_chief, v_chief, time_array, backend="cpu"):
    """Propagate chief only; return positions (n_times, 3) in km, ECI."""
    import tensorgator as tg
    kepler = rv_to_kepler(r_chief, v_chief)
    const_si = np.zeros((1, 6))
    const_si[0, 0] = kepler[0] * 1000.0
    const_si[0, 1:6] = kepler[1:6]
    pos_si = tg.satellite_positions(
        np.asarray(time_array, dtype=float),
        const_si,
        backend=backend,
        return_frame="eci",
        input_type="kepler",
    )
    return (pos_si[0] / 1000.0).copy()  # (n_times, 3) km


def detect_near_miss(positions_chief, positions_constellation, radius_km=DEFAULT_RADIUS_KM):
    """
    Check if chief comes within radius_km of any constellation satellite.

    Parameters
    ----------
    positions_chief : ndarray (n_times, 3)
        Chief ECI positions [km].
    positions_constellation : ndarray (n_times, n_sats, 3)
        Constellation ECI positions [km].
    radius_km : float
        Near-miss radius [km].

    Returns
    -------
    has_near_miss : bool
    min_distance_km : float
    t_idx : int
        Time index of minimum distance (or -1 if no near miss).
    k_idx : int
        Satellite index of minimum distance (or -1).
    """
    pos_c = np.asarray(positions_chief, dtype=float)
    pos_d = np.asarray(positions_constellation, dtype=float)
    n_times, n_sats = pos_d.shape[0], pos_d.shape[1]
    min_d = np.inf
    t_best, k_best = -1, -1
    for t in range(n_times):
        for k in range(n_sats):
            d = np.linalg.norm(pos_c[t] - pos_d[t, k])
            if d < min_d:
                min_d = d
                t_best, k_best = t, k
    return min_d < radius_km, min_d, t_best, k_best


def _compute_phi_rv(r_chief, v_chief, time_array, backend="cpu", h=FD_STEP_KM_S):
    """
    Sensitivity of chief position at each time to chief velocity at t=0.
    Phi_rv(t) = d r_chief(t) / d v_chief(0), shape (n_times, 3, 3).
    """
    pos_nominal = _propagate_chief_only(r_chief, v_chief, time_array, backend)
    n_times = len(time_array)
    Phi = np.zeros((n_times, 3, 3))
    for j in range(3):
        ej = np.zeros(3)
        ej[j] = h
        pos_j = _propagate_chief_only(r_chief, v_chief + ej, time_array, backend)
        Phi[:, :, j] = (pos_j - pos_nominal) / h
    return Phi


def collision_avoidance_delta_v(
    r_chief,
    v_chief,
    constellation_rtn,
    backend="cpu",
    radius_km=DEFAULT_RADIUS_KM,
    constraint_margin_km=CONSTRAINT_MARGIN_KM,
    n_times_orbit=100,
):
    """
    Compute impulsive delta-v (ECI) for the chief to avoid near misses with the constellation.

    Forward-propagates chief and constellation for one orbit. If no point is within
    radius_km, returns zeros. Otherwise solves a convex QP: minimize ||delta_v||_2^2
    subject to linearized constraints that keep chief-deputy distance >= radius_km
    at all constrained time steps (those where nominal distance < constraint_margin_km).

    Parameters
    ----------
    r_chief, v_chief : array_like (3,)
        Chief state in ECI (km, km/s).
    constellation_rtn : array_like (n_sats, 6)
        Relative RTN state for each constellation satellite.
    backend : str
        'cpu' or 'cuda' for propagation.
    radius_km : float
        Minimum allowed separation [km].
    constraint_margin_km : float
        Add constraints at times where nominal distance < this [km].
    n_times_orbit : int
        Number of time steps over one orbit for propagation.

    Returns
    -------
    delta_v : ndarray (3,)
        Impulsive velocity change in ECI [km/s]. Zero if no near miss.
    """
    r_chief = np.asarray(r_chief, dtype=float).reshape(3)
    v_chief = np.asarray(v_chief, dtype=float).reshape(3)
    constellation_rtn = np.asarray(constellation_rtn, dtype=float)
    if constellation_rtn.ndim == 1:
        constellation_rtn = constellation_rtn.reshape(1, 6)
    n_sats = constellation_rtn.shape[0]

    T = _orbit_period_seconds(r_chief, v_chief)
    time_array = np.linspace(0, T, n_times_orbit)

    pos_chief = _propagate_chief_only(r_chief, v_chief, time_array, backend)
    pos_constellation = propagate_constellation(
        r_chief, v_chief, constellation_rtn, time_array, backend=backend, return_velocity=False
    )
    if pos_constellation.ndim == 2:
        pos_constellation = pos_constellation[:, np.newaxis, :]

    has_near, min_d, _, _ = detect_near_miss(pos_chief, pos_constellation, radius_km)
    if not has_near:
        return np.zeros(3)

    Phi_rv = _compute_phi_rv(r_chief, v_chief, time_array, backend)

    A_list = []
    b_list = []
    for t in range(len(time_array)):
        for k in range(n_sats):
            r_rel = pos_chief[t] - pos_constellation[t, k]
            d = np.linalg.norm(r_rel)
            if d < 1e-9:
                d = 1e-9
            if d >= constraint_margin_km:
                continue
            u = r_rel / d
            # Linearized: u^T (r_rel + Phi_rv(t) @ dv) >= radius_km  =>  u^T Phi @ dv >= radius_km - d
            A_list.append(u @ Phi_rv[t])
            b_list.append(radius_km - d)

    if not A_list:
        return np.zeros(3)

    A = np.array(A_list)
    b = np.array(b_list)
    # minimize ||dv||_2^2  s.t.  A @ dv >= b  (convex QP)
    dv = cp.Variable(3)
    objective = cp.Minimize(cp.sum_squares(dv))
    constraints = [A @ dv >= b]
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve()
        if "optimal" not in str(problem.status).lower():
            return np.zeros(3)
        return dv.value
    except Exception:
        return np.zeros(3)


def propagate_constellation_with_avoidance(
    r_chief,
    v_chief,
    constellation_rtn,
    time_array,
    backend="cpu",
    return_velocity=True,
    radius_km=DEFAULT_RADIUS_KM,
):
    """
    Propagate constellation; if a near miss is detected, compute and apply an
    impulsive collision-avoidance delta-v to the chief, then propagate using
    the updated chief velocity.

    Returns the same as propagate_constellation (positions, optionally velocities),
    and in addition the chief state used is (r_chief, v_chief + delta_v) when
    a maneuver was applied.
    """
    delta_v = collision_avoidance_delta_v(
        r_chief, v_chief, constellation_rtn, backend=backend, radius_km=radius_km
    )
    r_chief_use, v_chief_use = apply_impulse_eci(r_chief, v_chief, delta_v)
    return propagate_constellation(
        r_chief_use,
        v_chief_use,
        constellation_rtn,
        time_array,
        backend=backend,
        return_velocity=return_velocity,
    )
