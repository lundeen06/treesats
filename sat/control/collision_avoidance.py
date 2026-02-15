"""
Collision avoidance for the chief satellite relative to a constellation.

Forward-propagates chief and constellation for one orbit, detects near misses
(< radius_km), and solves a convex QP (minimize delta-v, enforce minimum separation)
to compute an impulsive maneuver. Applies delta-v in ECI.

How the convex problem works
----------------------------
1. Nominal trajectories
   - Propagate chief (r_chief, v_chief) and constellation over one orbit.
   - pos_chief(t), pos_constellation(t,k) in ECI [km].

2. Linearization of the new trajectory
   - Apply impulsive dv at t=0: new velocity = v_chief + dv, position unchanged.
   - Sensitivity: Phi_rv(t) = d r_chief(t) / d v_chief(0) [3x3], computed by
     finite differences (propagate with v_chief + e_j, subtract nominal, divide by h).
   - To first order, chief position after maneuver at time t is:
       r_chief_new(t) = r_chief_nominal(t) + Phi_rv(t) @ dv.

3. Distance constraint (linearized)
   - For deputy k at time t: r_rel = pos_chief(t) - pos_constellation(t,k), d = |r_rel|, u = r_rel/d.
   - New relative position: r_rel_new = r_rel + Phi_rv(t) @ dv (deputy unchanged).
   - Change in distance along u:  delta_d ≈ u · (Phi_rv(t) @ dv) = (u^T Phi_rv(t)) dv.
   - We want new distance >= radius_km:  d + delta_d >= radius_km  =>  (u^T Phi_rv(t)) dv >= radius_km - d.
   - So one linear inequality per (t,k):  a_row = u @ Phi_rv(t),  b = radius_km - d;  a_row @ dv >= b.

4. Convex QP
   - minimize    ||dv||^2
   - subject to  A @ dv >= b   (one row per constrained (t,k)).
   - This is convex (quadratic objective, linear constraints). Solution dv pushes the chief
     away at the constrained times so that the linearized distance is >= radius_km there.

5. What is actually guaranteed
   - The constraints are only added at a small set of times: for each deputy k, at the
     time of *nominal* closest approach t_min and its neighbors (t_min-1, t_min+1), and
     only when nominal d < constraint_margin_km (default 2 km). So we do NOT constrain
     every (t,k) along the orbit.
   - The guarantee is linearized: at each (t,k) where a constraint was added, the
     *first-order* change in distance yields d_new >= radius_km. So:
     * Under the linear approximation, the new trajectory has distance >= radius_km at
       those specific (t,k).
     * The real trajectory can differ due to linearization error (especially if ||dv||
       is large). Also, the minimum distance on the new trajectory could occur at a
       different time than the nominal t_min.
   - The code now enforces that the *simulated* trajectory has no violation: after
     each QP solve we propagate (r_chief, v_chief + dv) and check every (t,k). If
     any distance < radius_km, we add constraints at those (t,k) and re-solve (up to
     max_verify_iter). We only return a non-zero dv when a final verification pass
     shows no (t,k) with distance < radius_km over the full orbit.
"""

import numpy as np
import cvxpy as cp

from .rtn_to_eci_propagate import (
    rv_to_kepler,
    propagate_constellation,
    propagate_chief,
    MU_EARTH,
)

# Default near-miss radius [km]
DEFAULT_RADIUS_KM = 1.0
# Constraint margin: add constraints at times where distance < this [km]
CONSTRAINT_MARGIN_KM = 2.0
# Numerical step for Phi_rv (velocity perturbation) [km/s]
FD_STEP_KM_S = 1e-6
# Minimum delta-v when a maneuver is applied [m/s] (thruster minimum impulse)
MIN_DELTA_V_M_S = 0.002


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
    return propagate_chief(r_chief, v_chief, time_array, backend=backend)  # (n_times, 3) km


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


def _verify_dv(r_chief, v_chief, dv, constellation_rtn, time_array, radius_km, backend="cpu"):
    """
    Simulate chief with (r_chief, v_chief + dv) over time_array; check distance
    to every constellation satellite at every timestep. Return (violated_pairs, min_dist_km)
    where violated_pairs is a list of (t, k) for which distance < radius_km.
    """
    v_new = v_chief + np.asarray(dv, dtype=float).reshape(3)
    pos_chief_new = _propagate_chief_only(r_chief, v_new, time_array, backend)
    pos_constellation = propagate_constellation(
        r_chief, v_chief, constellation_rtn, time_array, backend=backend, return_velocity=False
    )
    if pos_constellation.ndim == 2:
        pos_constellation = pos_constellation[:, np.newaxis, :]
    n_times, n_sats = pos_constellation.shape[0], pos_constellation.shape[1]
    violated = []
    min_d = np.inf
    for t in range(n_times):
        for k in range(n_sats):
            d = np.linalg.norm(pos_chief_new[t] - pos_constellation[t, k])
            if d < min_d:
                min_d = d
            if d < radius_km:
                violated.append((t, k))
    return violated, float(min_d)


def _constraints_for_pairs(pos_chief, pos_constellation, Phi_rv, pairs, radius_km):
    """Build (A_rows, b_rows) for linearized distance >= radius_km at each (t,k) in pairs."""
    A_list = []
    b_list = []
    for (t, k) in pairs:
        if t < 0 or t >= pos_chief.shape[0]:
            continue
        r_rel = pos_chief[t] - pos_constellation[t, k]
        d = max(np.linalg.norm(r_rel), 1e-9)
        u = r_rel / d
        A_list.append(u @ Phi_rv[t])
        b_list.append(radius_km - d)
    return A_list, b_list


def collision_avoidance_delta_v(
    r_chief,
    v_chief,
    constellation_rtn,
    backend="cpu",
    radius_km=DEFAULT_RADIUS_KM,
    constraint_margin_km=CONSTRAINT_MARGIN_KM,
    n_times_orbit=100,
    min_delta_v_m_s=MIN_DELTA_V_M_S,
    max_verify_iter=10,
    verbose=False,
):
    """
    Compute impulsive delta-v (ECI) for the chief to avoid near misses with the constellation.

    Forward-propagates chief and constellation for one orbit. If no point is within
    radius_km, returns zeros. Otherwise solves a convex QP: minimize ||delta_v||_2^2
    subject to linearized constraints that keep chief-deputy distance >= radius_km
    at the constrained time steps (near each deputy's nominal closest approach; see
    module docstring for the math).

    Verification: The QP only guarantees the linearized distance >= radius_km at the
    constrained (t,k). To ensure the *actual* new trajectory has no miss distance
    < radius_km with any satellite, propagate (r_chief, v_chief + delta_v) and the
    constellation and run detect_near_miss(...). The test does this and asserts
    no near miss after evasion.

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
    min_delta_v_m_s : float
        Minimum delta-v magnitude when a maneuver is applied [m/s] (thruster minimum
        impulse). If the QP solution has 0 < ||dv|| < this, dv is scaled up to this.
    max_verify_iter : int
        After each QP solve, simulate (r_chief, v_chief + dv) and check all (t,k).
        If any distance < radius_km, add constraints at those (t,k) and re-solve.
        Stop when no violations or after this many iterations.
    verbose : bool
        If True, print debug info (min distance, constraint count, QP status).

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

    has_near, min_d, t_near, k_near = detect_near_miss(pos_chief, pos_constellation, radius_km)
    if verbose:
        print(f"  [collision_avoidance] n_times_orbit={n_times_orbit}, has_near={has_near}, min_distance_km={min_d:.4f}, t_idx={t_near}, k_idx={k_near}")
    if not has_near:
        if verbose:
            print("  [collision_avoidance] No near miss -> delta_v = 0")
        return np.zeros(3)

    Phi_rv = _compute_phi_rv(r_chief, v_chief, time_array, backend)

    # Constraint set: (t, k) pairs where we enforce linearized distance >= radius_km.
    # Start with nominal close-approach times (and neighbors) per deputy.
    constrained_pairs = set()
    for k in range(n_sats):
        d_vals = [
            np.linalg.norm(pos_chief[t] - pos_constellation[t, k])
            for t in range(1, len(time_array))
        ]
        t_min = 1 + int(np.argmin(d_vals))
        d_min = np.linalg.norm(pos_chief[t_min] - pos_constellation[t_min, k])
        if d_min >= constraint_margin_km:
            continue
        for t in (t_min - 1, t_min, t_min + 1):
            if t < 1 or t >= len(time_array):
                continue
            d = np.linalg.norm(pos_chief[t] - pos_constellation[t, k])
            if d < constraint_margin_km:
                constrained_pairs.add((t, k))

    if not constrained_pairs:
        if verbose:
            print("  [collision_avoidance] No initial constraints -> delta_v = 0")
        return np.zeros(3)

    min_dv_km_s = float(min_delta_v_m_s) / 1000.0
    dv_out = None

    for verify_iter in range(max_verify_iter):
        A_list, b_list = _constraints_for_pairs(
            pos_chief, pos_constellation, Phi_rv, list(constrained_pairs), radius_km
        )
        if not A_list:
            break
        A = np.array(A_list)
        b = np.array(b_list)
        if verbose:
            print(f"  [collision_avoidance] iter {verify_iter}: n_constraints={len(A_list)}")

        dv_var = cp.Variable(3)
        objective = cp.Minimize(cp.sum_squares(dv_var))
        constraints = [A @ dv_var >= b]
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve()
            status = str(problem.status).lower()
            if "optimal" not in status:
                if verbose:
                    print(f"  [collision_avoidance] QP not optimal -> delta_v = 0")
                return np.zeros(3)
            val = dv_var.value
            if val is None:
                return np.zeros(3)
            dv_out = np.asarray(val).reshape(3)
        except Exception as e:
            if verbose:
                print(f"  [collision_avoidance] QP exception: {e} -> delta_v = 0")
            return np.zeros(3)

        # Minimum impulse: scale up if 0 < ||dv|| < min
        n_dv = np.linalg.norm(dv_out)
        if n_dv > 1e-12 and n_dv < min_dv_km_s:
            dv_out = dv_out * (min_dv_km_s / n_dv)
            if verbose:
                print(f"  [collision_avoidance] scaled delta_v to min {min_delta_v_m_s} m/s")

        # Verify: simulate (r_chief, v_chief + dv_out) and check all (t, k)
        violated, min_dist_km = _verify_dv(
            r_chief, v_chief, dv_out, constellation_rtn, time_array, radius_km, backend
        )
        if verbose:
            print(f"  [collision_avoidance] verify: min_dist_km={min_dist_km:.4f}, n_violations={len(violated)}")
        if not violated:
            break
        for (t, k) in violated:
            constrained_pairs.add((t, k))

    if dv_out is None:
        return np.zeros(3)
    # Final verification: only return non-zero dv if simulated trajectory has no violation
    violated_final, _ = _verify_dv(
        r_chief, v_chief, dv_out, constellation_rtn, time_array, radius_km, backend
    )
    if violated_final:
        if verbose:
            print(f"  [collision_avoidance] after {max_verify_iter} iters still {len(violated_final)} violations -> delta_v = 0 (unsafe)")
        return np.zeros(3)
    return dv_out


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
