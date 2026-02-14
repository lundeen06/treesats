"""
Convert relative RTN state to absolute ECI and propagate the deputy.

All inputs and outputs are in ECI (km, km/s). Propagation is done via Tensorgator
(return_frame='eci'). We convert deputy state to Keplerian elements only because
Tensorgator's API accepts Keplerian elements for propagation (it does not accept
(r,v) state vectors on the CPU backend); the propagation itself stays in ECI.
"""

import numpy as np

MU_EARTH = 398600.4418  # km^3/s^2

def relative_rtn_to_absolute_eci(r_chief, v_chief, relative_state_rtn):
    """
    Converts relative RTN state to absolute ECI state.
    Velocity in ECI comes from rotating the RTN velocity (plus Coriolis term).

    Parameters
    ----------
    r_chief, v_chief : array_like
        Chief position and velocity in ECI (km, km/s).
    relative_state_rtn : array_like
        [dR, dT, dN, dR_dot, dT_dot, dN_dot] in km and km/s.

    Returns
    -------
    r_deputy_eci, v_deputy_eci : ndarray (3,)
    """
    r_chief = np.asarray(r_chief, dtype=float).reshape(3)
    v_chief = np.asarray(v_chief, dtype=float).reshape(3)
    relative_state_rtn = np.asarray(relative_state_rtn, dtype=float).reshape(6)
    dr_rtn = relative_state_rtn[:3]
    dv_rtn = relative_state_rtn[3:]

    u_r = r_chief / np.linalg.norm(r_chief)
    h_vec = np.cross(r_chief, v_chief)
    u_n = h_vec / np.linalg.norm(h_vec)
    u_t = np.cross(u_n, u_r)
    Q = np.column_stack((u_r, u_t, u_n))  # RTN -> ECI

    r_mag_sq = np.dot(r_chief, r_chief)
    omega_vec = h_vec / r_mag_sq

    dr_eci = Q @ dr_rtn
    r_deputy_eci = r_chief + dr_eci
    v_deputy_eci = v_chief + (Q @ dv_rtn) + np.cross(omega_vec, dr_eci)
    return r_deputy_eci, v_deputy_eci


def rv_to_kepler(r_eci, v_eci, mu=MU_EARTH):
    """Cartesian ECI (km, km/s) to Keplerian [a, e, i, omega, Omega, M] (km, rad)."""
    r_eci = np.asarray(r_eci, dtype=float).reshape(3)
    v_eci = np.asarray(v_eci, dtype=float).reshape(3)
    r = np.linalg.norm(r_eci)
    v = np.linalg.norm(v_eci)
    h_vec = np.cross(r_eci, v_eci)
    h = np.linalg.norm(h_vec)
    a = 1.0 / (2.0 / r - (v * v) / mu)
    e_vec = (np.cross(v_eci, h_vec) / mu) - (r_eci / r)
    e = np.linalg.norm(e_vec)
    i = np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))
    n_vec = np.array([-h_vec[1], h_vec[0], 0.0])
    n = np.linalg.norm(n_vec)
    Omega = 0.0 if n < 1e-12 else np.arccos(np.clip(n_vec[0] / n, -1.0, 1.0))
    if n >= 1e-12 and n_vec[1] < 0:
        Omega = 2 * np.pi - Omega
    omega = 0.0
    if n >= 1e-12 and e >= 1e-12:
        omega = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1.0, 1.0))
        if e_vec[2] < 0:
            omega = 2 * np.pi - omega
    if e < 1e-12:
        nu = np.arccos(np.clip(np.dot(r_eci, v_eci) / (r * v), -1.0, 1.0))
        if np.dot(r_eci, v_eci) < 0:
            nu = 2 * np.pi - nu
    else:
        nu = np.arccos(np.clip(np.dot(e_vec, r_eci) / (e * r), -1.0, 1.0))
        if np.dot(r_eci, v_eci) < 0:
            nu = 2 * np.pi - nu
    E = 2.0 * np.arctan2(np.sqrt(1 - e) * np.tan(nu / 2), np.sqrt(1 + e))
    M = E - e * np.sin(E)
    M = np.mod(M, 2 * np.pi)
    return np.array([a, e, i, omega, Omega, M])


def propagate_deputy(
    r_chief,
    v_chief,
    relative_state_rtn,
    time_array,
    backend="cpu",
    return_velocity=True,
):
    """
    Convert relative RTN to ECI, then propagate with Tensorgator in ECI.

    Position at each time comes from Tensorgator. Velocity is computed by finite
    differences (central difference; forward/backward at endpoints) of those
    positions so it matches the same trajectory.

    Parameters
    ----------
    r_chief, v_chief : array_like
        Chief state in ECI at reference time (km, km/s).
    relative_state_rtn : array_like
        [dR, dT, dN, dR_dot, dT_dot, dN_dot].
    time_array : array_like
        Times in seconds (relative to epoch).
    backend : str
        'cpu' or 'cuda' for Tensorgator.
    return_velocity : bool
        If True (default), return (positions, velocities); else positions only.
        Velocities are from finite difference of Tensorgator positions.

    Returns
    -------
    positions : (n_times, 3) km, ECI
    velocities : (n_times, 3) km/s, ECI, if return_velocity=True
    """
    import tensorgator as tg

    time_array = np.asarray(time_array, dtype=float)
    r_dep, v_dep = relative_rtn_to_absolute_eci(r_chief, v_chief, relative_state_rtn)
    kepler_km = rv_to_kepler(r_dep, v_dep)
    constellation_si = np.zeros((1, 6))
    constellation_si[0, 0] = kepler_km[0] * 1000.0  # a km -> m
    constellation_si[0, 1:6] = kepler_km[1:6]

    pos_si = tg.satellite_positions(
        time_array,
        constellation_si,
        backend=backend,
        return_frame="eci",
        input_type="kepler",
    )
    # (1, n_times, 3) in m -> (n_times, 3) in km
    positions = (np.transpose(pos_si, (1, 0, 2))[:, 0, :] / 1000.0).copy()

    if not return_velocity:
        return positions

    n = len(time_array)
    velocities = np.zeros((n, 3))
    for i in range(n):
        if i == 0:
            dt = time_array[1] - time_array[0]
            velocities[i] = (positions[1] - positions[0]) / dt
        elif i == n - 1:
            dt = time_array[-1] - time_array[-2]
            velocities[i] = (positions[-1] - positions[-2]) / dt
        else:
            dt = time_array[i + 1] - time_array[i - 1]
            velocities[i] = (positions[i + 1] - positions[i - 1]) / dt
    return positions, velocities
