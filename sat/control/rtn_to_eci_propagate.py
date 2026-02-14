"""
Convert relative RTN state to absolute ECI and propagate a constellation.

All inputs and outputs are in ECI (km, km/s). Chief is the reference; constellation
is an array of relative RTN states (one per satellite). Propagation is done via
Tensorgator (return_frame='eci').
"""

import numpy as np

MU_EARTH = 398600.4418  # km^3/s^2


def eci_to_rtn_basis(pos_eci, vel_eci):
    """
    Compute RTN basis vectors (rows) from observer position and velocity in ECI.

    Same convention as relative_rtn_to_absolute_eci: R = radial, T = along-track, N = cross-track.
    Returns 3x3 with rows (R, T, N) in ECI components so that v_rtn = basis_rtn @ v_eci.

    Parameters
    ----------
    pos_eci, vel_eci : array_like shape (3,)
        Observer position and velocity in ECI (e.g. km, km/s).

    Returns
    -------
    basis_rtn : ndarray shape (3, 3)
        Rows are R, T, N unit vectors in ECI. v_rtn = basis_rtn @ v_eci.
    """
    r = np.asarray(pos_eci, dtype=float).reshape(3)
    v = np.asarray(vel_eci, dtype=float).reshape(3)
    u_r = r / np.linalg.norm(r)
    h_vec = np.cross(r, v)
    h = np.linalg.norm(h_vec)
    if h < 1e-12:
        if abs(u_r[2]) < 0.9:
            u_n = np.cross(u_r, np.array([0.0, 0.0, 1.0]))
        else:
            u_n = np.cross(u_r, np.array([1.0, 0.0, 0.0]))
        u_n = u_n / np.linalg.norm(u_n)
    else:
        u_n = h_vec / h
    u_t = np.cross(u_n, u_r)
    Q = np.column_stack((u_r, u_t, u_n))  # ECI from RTN: v_eci = Q @ v_rtn
    return Q.T  # rows R,T,N so v_rtn = Q.T @ v_eci


def vector_eci_to_rtn(vec_eci, basis_rtn):
    """Transform vector(s) from ECI to RTN. basis_rtn has rows (R, T, N) from eci_to_rtn_basis."""
    vec_eci = np.asarray(vec_eci, dtype=float)
    if vec_eci.ndim == 1:
        return np.dot(basis_rtn, vec_eci)
    return np.dot(vec_eci, basis_rtn.T)


def vector_rtn_to_eci(vec_rtn, basis_rtn):
    """Transform vector(s) from RTN to ECI. basis_rtn has rows (R, T, N) from eci_to_rtn_basis."""
    vec_rtn = np.asarray(vec_rtn, dtype=float)
    if vec_rtn.ndim == 1:
        return np.dot(basis_rtn.T, vec_rtn)
    return np.dot(vec_rtn, basis_rtn)


def relative_rtn_to_absolute_eci(r_chief, v_chief, constellation_rtn):
    """
    Convert relative RTN state(s) to absolute ECI for a constellation.

    Velocity in ECI comes from rotating the RTN velocity (plus Coriolis term).
    Same RTN frame (chief) for all satellites.

    Parameters
    ----------
    r_chief, v_chief : array_like
        Chief position and velocity in ECI (km, km/s), shape (3,).
    constellation_rtn : array_like
        Relative state(s) in RTN. Either shape (6,) for one satellite or
        (n_sats, 6) for a constellation. Each row is [dR, dT, dN, dR_dot, dT_dot, dN_dot]
        in km and km/s.

    Returns
    -------
    r_eci : ndarray
        Position(s) in ECI (km). Shape (3,) if input was (6,); (n_sats, 3) if (n_sats, 6).
    v_eci : ndarray
        Velocity(ies) in ECI (km/s). Same shape as r_eci.
    """
    r_chief = np.asarray(r_chief, dtype=float).reshape(3)
    v_chief = np.asarray(v_chief, dtype=float).reshape(3)
    constellation_rtn = np.asarray(constellation_rtn, dtype=float)
    single = constellation_rtn.ndim == 1
    if single:
        constellation_rtn = constellation_rtn.reshape(1, 6)

    u_r = r_chief / np.linalg.norm(r_chief)
    h_vec = np.cross(r_chief, v_chief)
    u_n = h_vec / np.linalg.norm(h_vec)
    u_t = np.cross(u_n, u_r)
    Q = np.column_stack((u_r, u_t, u_n))
    r_mag_sq = np.dot(r_chief, r_chief)
    omega_vec = h_vec / r_mag_sq

    dr_rtn = constellation_rtn[:, :3]
    dv_rtn = constellation_rtn[:, 3:]
    dr_eci = (Q @ dr_rtn.T).T
    r_eci = r_chief + dr_eci
    v_eci = v_chief + (Q @ dv_rtn.T).T + np.cross(omega_vec, dr_eci)

    if single:
        return r_eci[0], v_eci[0]
    return r_eci, v_eci


def rv_to_kepler(r_eci, v_eci, mu=MU_EARTH):
    """
    Cartesian ECI (km, km/s) to Keplerian [a, e, i, omega, Omega, M] (km, rad).

    Accepts single state (3,) and (3,) or constellation (n_sats, 3) and (n_sats, 3);
    returns (6,) or (n_sats, 6).
    """
    r_eci = np.asarray(r_eci, dtype=float)
    v_eci = np.asarray(v_eci, dtype=float)
    single = r_eci.ndim == 1
    if single:
        r_eci = r_eci.reshape(1, 3)
        v_eci = v_eci.reshape(1, 3)
    n_sats = r_eci.shape[0]
    kepler = np.zeros((n_sats, 6))
    for k in range(n_sats):
        r, v = np.linalg.norm(r_eci[k]), np.linalg.norm(v_eci[k])
        h_vec = np.cross(r_eci[k], v_eci[k])
        h = np.linalg.norm(h_vec)
        a = 1.0 / (2.0 / r - (v * v) / mu)
        e_vec = (np.cross(v_eci[k], h_vec) / mu) - (r_eci[k] / r)
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
            nu = np.arccos(np.clip(np.dot(r_eci[k], v_eci[k]) / (r * v), -1.0, 1.0))
            if np.dot(r_eci[k], v_eci[k]) < 0:
                nu = 2 * np.pi - nu
        else:
            nu = np.arccos(np.clip(np.dot(e_vec, r_eci[k]) / (e * r), -1.0, 1.0))
            if np.dot(r_eci[k], v_eci[k]) < 0:
                nu = 2 * np.pi - nu
        E = 2.0 * np.arctan2(np.sqrt(1 - e) * np.tan(nu / 2), np.sqrt(1 + e))
        M = E - e * np.sin(E)
        kepler[k] = [a, e, i, omega, Omega, np.mod(M, 2 * np.pi)]
    if single:
        return kepler[0]
    return kepler


def propagate_constellation(
    r_chief,
    v_chief,
    constellation_rtn,
    time_array,
    backend="cpu",
    return_velocity=True,
):
    """
    Convert relative RTN states to ECI, then propagate constellation with Tensorgator.

    Position at each time comes from Tensorgator. Velocity is computed by finite
    differences (central difference; forward/backward at endpoints) per satellite.

    Parameters
    ----------
    r_chief, v_chief : array_like
        Chief state in ECI at reference time (km, km/s), shape (3,).
    constellation_rtn : array_like
        Relative RTN state(s). Shape (6,) for one satellite or (n_sats, 6) for
        a constellation. Each row is [dR, dT, dN, dR_dot, dT_dot, dN_dot] in km and km/s.
    time_array : array_like
        Times in seconds (relative to epoch).
    backend : str
        'cpu' or 'cuda' for Tensorgator.
    return_velocity : bool
        If True (default), return (positions, velocities); else positions only.

    Returns
    -------
    positions : ndarray
        ECI positions (km). Shape (n_times, 3) if one sat; (n_times, n_sats, 3) if constellation.
    velocities : ndarray, optional
        ECI velocities (km/s). Same shape as positions. Only if return_velocity=True.
    """
    import tensorgator as tg

    time_array = np.asarray(time_array, dtype=float)
    constellation_rtn = np.asarray(constellation_rtn, dtype=float)
    single = constellation_rtn.ndim == 1
    if single:
        constellation_rtn = constellation_rtn.reshape(1, 6)

    r_eci, v_eci = relative_rtn_to_absolute_eci(r_chief, v_chief, constellation_rtn)
    kepler_km = rv_to_kepler(r_eci, v_eci)
    n_sats = kepler_km.shape[0]
    constellation_si = np.zeros((n_sats, 6))
    constellation_si[:, 0] = kepler_km[:, 0] * 1000.0  # a km -> m
    constellation_si[:, 1:6] = kepler_km[:, 1:6]

    pos_si = tg.satellite_positions(
        time_array,
        constellation_si,
        backend=backend,
        return_frame="eci",
        input_type="kepler",
    )
    # (n_sats, n_times, 3) in m -> (n_times, n_sats, 3) in km
    positions = np.transpose(pos_si, (1, 0, 2)) / 1000.0

    if not return_velocity:
        if single:
            return positions[:, 0, :]
        return positions

    n_times = len(time_array)
    velocities = np.zeros((n_times, n_sats, 3))
    for k in range(n_sats):
        for i in range(n_times):
            if i == 0:
                dt = time_array[1] - time_array[0]
                velocities[i, k] = (positions[1, k] - positions[0, k]) / dt
            elif i == n_times - 1:
                dt = time_array[-1] - time_array[-2]
                velocities[i, k] = (positions[-1, k] - positions[-2, k]) / dt
            else:
                dt = time_array[i + 1] - time_array[i - 1]
                velocities[i, k] = (positions[i + 1, k] - positions[i - 1, k]) / dt

    if single:
        return positions[:, 0, :], velocities[:, 0, :]
    return positions, velocities


def propagate_deputy(r_chief, v_chief, relative_state_rtn, time_array, backend="cpu", return_velocity=True):
    """
    Propagate a single deputy. Alias for propagate_constellation with one satellite.
    relative_state_rtn can be shape (6,) or (1, 6). Returns (n_times, 3) arrays.
    """
    return propagate_constellation(
        r_chief, v_chief, relative_state_rtn, time_array, backend=backend, return_velocity=return_velocity
    )
