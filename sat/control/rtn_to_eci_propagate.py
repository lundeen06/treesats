"""
Convert relative RTN state to absolute ECI and propagate a constellation.

All inputs and outputs are in ECI (km, km/s). Chief is the reference; constellation
is an array of relative RTN states (one per satellite). Propagation is done via
Tensorgator (return_frame='eci').

Physical constants (earth radius, mu) are taken from sim.params.CONSTANTS when
available; otherwise fallbacks are used. params uses: earth_radius_km [km],
earth_radius_m [m], mu [m^3/s^2]. Here we use km and km^3/s^2 (mu_km = mu / 1e9).
"""

import numpy as np


def _constants_from_params():
    """Load earth_radius_km and mu in km^3/s^2 from sim.params.CONSTANTS. Fallback if unavailable."""
    try:
        import sys
        from pathlib import Path
        sim_dir = Path(__file__).resolve().parent.parent.parent / "sim"
        if str(sim_dir) not in sys.path:
            sys.path.insert(0, str(sim_dir))
        from params import CONSTANTS
        # params: mu in m^3/s^2 -> km^3/s^2 = mu / 1e9
        earth_radius_km = float(CONSTANTS["earth_radius_km"])
        mu_km = float(CONSTANTS["mu"]) / 1e9
        return earth_radius_km, mu_km
    except Exception:
        return 6378.137, 398600.4418


_EARTH_RADIUS_KM, _MU_KM = _constants_from_params()
MU_EARTH = _MU_KM  # km^3/s^2 (alias for backward compatibility)


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


def eci_to_relative_rtn(r_chief, v_chief, r_eci, v_eci):
    """
    Convert absolute ECI state(s) to relative RTN state(s) w.r.t. chief.

    Inverse of relative_rtn_to_absolute_eci: if you pass the result as
    constellation_rtn with this chief, you get back (r_eci, v_eci).

    Parameters
    ----------
    r_chief, v_chief : array_like (3,)
        Chief position and velocity in ECI (km, km/s).
    r_eci, v_eci : array_like
        Deputy position(s) and velocity(ies) in ECI. Either (3,) and (3,)
        for one deputy, or (n_sats, 3) and (n_sats, 3).

    Returns
    -------
    state_rtn : ndarray
        Relative state(s) [dR, dT, dN, dR_dot, dT_dot, dN_dot]. Shape (6,) or (n_sats, 6).
    """
    r_chief = np.asarray(r_chief, dtype=float).reshape(3)
    v_chief = np.asarray(v_chief, dtype=float).reshape(3)
    r_eci = np.asarray(r_eci, dtype=float)
    v_eci = np.asarray(v_eci, dtype=float)
    single = r_eci.ndim == 1
    if single:
        r_eci = r_eci.reshape(1, 3)
        v_eci = v_eci.reshape(1, 3)
    basis_rtn = eci_to_rtn_basis(r_chief, v_chief)
    h_vec = np.cross(r_chief, v_chief)
    r_mag_sq = np.dot(r_chief, r_chief)
    omega_vec = h_vec / r_mag_sq
    dr_eci = r_eci - r_chief
    dv_eci = v_eci - v_chief - np.cross(omega_vec, dr_eci)
    dr_rtn = (basis_rtn @ dr_eci.T).T
    dv_rtn = (basis_rtn @ dv_eci.T).T
    state_rtn = np.hstack((dr_rtn, dv_rtn))
    if single:
        return state_rtn[0]
    return state_rtn


def _orbit_period_seconds(r_eci, v_eci, mu=None):
    """Orbit period [s] from ECI state (km, km/s). mu in km^3/s^2; if None, from params."""
    if mu is None:
        mu = _MU_KM
    r = np.linalg.norm(np.asarray(r_eci).reshape(3))
    v = np.linalg.norm(np.asarray(v_eci).reshape(3))
    a = 1.0 / (2.0 / r - (v * v) / mu)
    return 2 * np.pi * np.sqrt((a ** 3) / mu)


def deputy_state_for_rendezvous(
    r_chief,
    v_chief,
    T_rendezvous=None,
    alt_inner_km=None,
    earth_radius_km=None,
    backend="cpu",
):
    """
    Deputy initial state (at t=0) on a Hohmann transfer that arrives at the chief's
    position at T_rendezvous. Chief is propagated half an orbit to get the meeting
    point; deputy is on a transfer orbit (inner alt -> chief alt) and is backwards
    propagated from that meeting point.

    Parameters
    ----------
    r_chief, v_chief : array_like (3,)
        Chief position and velocity in ECI (km, km/s) at t=0.
    T_rendezvous : float, optional
        Time of rendezvous [s]. If None, use half of the Hohmann transfer period.
    alt_inner_km : float, optional
        Inner orbit altitude [km] for the Hohmann (periapsis of transfer). If None,
        use chief altitude minus 50 km.
    earth_radius_km : float, optional
        Earth radius [km]. If None, from params.CONSTANTS['earth_radius_km'].
    backend : str
        Unused; kept for API compatibility.

    Returns
    -------
    r_deputy_0, v_deputy_0 : ndarray (3,)
        Deputy position and velocity in ECI at t=0 (periapsis of transfer).
    r_meet, v_meet : ndarray (3,)
        Meeting point (chief position at T_rendezvous) and chief velocity there.
    T_rendezvous : float
        Rendezvous time [s] used.
    """
    if earth_radius_km is None:
        earth_radius_km = _EARTH_RADIUS_KM
    r_chief = np.asarray(r_chief, dtype=float).reshape(3)
    v_chief = np.asarray(v_chief, dtype=float).reshape(3)
    r_outer = np.linalg.norm(r_chief)
    alt_chief_km = r_outer - earth_radius_km
    if alt_inner_km is None:
        alt_inner_km = max(200.0, alt_chief_km - 50.0)
    r_inner = earth_radius_km + alt_inner_km
    a_transfer = (r_inner + r_outer) / 2.0
    e_transfer = (r_outer - r_inner) / (r_outer + r_inner)
    T_transfer_half = np.pi * np.sqrt((a_transfer ** 3) / _MU_KM)
    if T_rendezvous is None:
        T_rendezvous = T_transfer_half

    # Chief at T_rendezvous
    time_forward = np.array([0.0, T_rendezvous], dtype=float)
    pos_chief, vel_chief = propagate_constellation(
        r_chief, v_chief,
        np.zeros((1, 6)),
        time_forward,
        backend=backend,
        return_velocity=True,
    )
    r_meet = np.asarray(pos_chief[1, 0], dtype=float)
    v_chief_meet = np.asarray(vel_chief[1, 0], dtype=float)

    # Deputy at meeting point: on transfer orbit at apoapsis (same position r_meet, velocity tangent)
    v_apo = np.sqrt(_MU_KM * (2.0 / r_outer - 1.0 / a_transfer))
    u_along = v_chief_meet / np.linalg.norm(v_chief_meet)
    v_deputy_meet = v_apo * u_along

    # Transfer orbit through (r_meet, v_deputy_meet); backwards by T_rendezvous to get deputy at t=0
    kepler_transfer = rv_to_kepler(r_meet, v_deputy_meet)
    a, e, i, omega, Omega, M_meet = kepler_transfer
    n_mean = np.sqrt(_MU_KM / (a ** 3))
    M0 = np.mod(M_meet - n_mean * T_rendezvous, 2 * np.pi)
    r_deputy_0, v_deputy_0 = kepler_to_rv(a, e, i, omega, Omega, M0)

    return r_deputy_0, v_deputy_0, r_meet, v_chief_meet, T_rendezvous


def deputy_above_chief_rendezvous_delta_v(
    r_chief,
    v_chief,
    alt_offset_km=3.0,
    rendezvous_tol_km=0.5,
    earth_radius_km=None,
    mu_km=None,
    backend="cpu",
    max_search_orbits=5,
):
    """
    Deputy is in a circular orbit `alt_offset_km` above the chief (same plane, same
    phase: radially above the chief at t=0). Compute the delta-v (single burn at t=0)
    so the deputy rendezvouses with the chief: deputy position matches chief position
    to within `rendezvous_tol_km` at some time in the propagated orbit. Uses Hohmann
    transfer (chief_alt+offset down to chief_alt). Searches over time for a rendezvous
    within the tolerance (phasing may require waiting multiple half-transfer periods).

    Parameters
    ----------
    r_chief, v_chief : array_like (3,)
        Chief position and velocity in ECI (km, km/s) at t=0.
    alt_offset_km : float
        Deputy altitude above chief [km] (deputy at chief_alt + alt_offset_km).
    rendezvous_tol_km : float
        Rendezvous success: deputy within this distance [km] of chief at rendezvous time.
    earth_radius_km, mu_km : float, optional
        Earth radius [km] and gravitational parameter [km^3/s^2]. If None, from params.CONSTANTS.
    backend : str
        For propagation.
    max_search_orbits : int
        Search over time up to this many chief orbits for a rendezvous.

    Returns
    -------
    delta_v : ndarray (3,)
        Delta-v in ECI [km/s] to apply to deputy at t=0 (retrograde for transfer down).
    T_rendezvous : float
        Time of rendezvous [s] when distance <= rendezvous_tol_km.
    r_chief_0 : ndarray (3,)
        Chief position at t=0 (same as input r_chief).
    v_chief_0 : ndarray (3,)
        Chief velocity at t=0 (same as input v_chief).
    r_deputy_0 : ndarray (3,)
        Deputy position at t=0 (3 km above chief).
    v_deputy_0 : ndarray (3,)
        Deputy velocity at t=0 (circular) before applying delta_v.
    r_meet : ndarray (3,)
        Chief position at T_rendezvous (rendezvous point).
    """
    if earth_radius_km is None:
        earth_radius_km = _EARTH_RADIUS_KM
    if mu_km is None:
        mu_km = _MU_KM
    r_chief = np.asarray(r_chief, dtype=float).reshape(3)
    v_chief = np.asarray(v_chief, dtype=float).reshape(3)
    r_chief_mag = np.linalg.norm(r_chief)
    alt_chief = r_chief_mag - earth_radius_km
    r_inner = earth_radius_km + alt_chief
    r_dep_mag = earth_radius_km + alt_chief + alt_offset_km

    a_transfer = (r_inner + r_dep_mag) / 2.0
    T_transfer_half = np.pi * np.sqrt((a_transfer ** 3) / mu_km)
    T_chief = 2.0 * np.pi * np.sqrt((r_inner ** 3) / mu_km)
    t_max = max_search_orbits * T_chief

    # Deputy at t=0: 3 km above chief (same radial line, same velocity direction)
    u_radial = r_chief / r_chief_mag
    r_deputy_0 = r_dep_mag * u_radial
    u_along = v_chief / np.linalg.norm(v_chief)
    v_circ_dep = np.sqrt(mu_km / r_dep_mag)
    v_deputy_0 = v_circ_dep * u_along

    # Hohmann first burn: at apoapsis, reduce speed (retrograde)
    v_apo_transfer = np.sqrt(mu_km * (2.0 / r_dep_mag - 1.0 / a_transfer))
    delta_v_mag = v_apo_transfer - v_circ_dep
    delta_v = delta_v_mag * u_along

    # Search for a time when deputy and chief are within rendezvous_tol_km
    n_steps = max(200, int(t_max / 20.0))
    time_grid = np.linspace(0.0, t_max, n_steps)
    pos_chief = propagate_chief(r_chief, v_chief, time_grid, backend=backend)
    pos_deputy = propagate_chief(
        r_deputy_0, v_deputy_0 + delta_v, time_grid, backend=backend
    )
    distances = np.linalg.norm(pos_deputy - pos_chief, axis=1)
    min_idx = np.argmin(distances)
    min_dist_km = distances[min_idx]
    t_rendezvous = time_grid[min_idx]

    if min_dist_km > rendezvous_tol_km:
        raise ValueError(
            f"No rendezvous within {rendezvous_tol_km} km in {max_search_orbits} orbits: "
            f"minimum distance = {min_dist_km:.4f} km at t = {t_rendezvous:.0f} s."
        )

    r_meet = np.asarray(pos_chief[min_idx], dtype=float)
    return (
        delta_v,
        t_rendezvous,
        r_chief.copy(),
        v_chief.copy(),
        r_deputy_0,
        v_deputy_0,
        r_meet,
    )


def kepler_to_rv(a, e, i, omega, Omega, M, mu=None):
    """
    Keplerian [a, e, i, omega, Omega, M] (km, rad) to Cartesian ECI (km, km/s).
    Single state only. Elliptic orbits (e < 1). mu in km^3/s^2; if None, from params.
    """
    if mu is None:
        mu = _MU_KM
    a, e = float(a), float(e)
    i, omega, Omega, M = float(i), float(omega), float(Omega), float(M)
    # Solve M = E - e*sin(E) for E (Newton)
    E = M
    for _ in range(20):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(E / 2.0),
        np.sqrt(1.0 - e) * np.cos(E / 2.0),
    )
    p = a * (1.0 - e * e)
    r_mag = p / (1.0 + e * np.cos(nu))
    r_pqw = r_mag * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pqw = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])
    # Perifocal to ECI: Rz(-Omega) Rx(-i) Rz(-omega)
    co, so = np.cos(Omega), np.sin(Omega)
    ci, si = np.cos(i), np.sin(i)
    cw, sw = np.cos(omega), np.sin(omega)
    R = np.array([
        [co * cw - so * sw * ci, -co * sw - so * cw * ci, so * si],
        [so * cw + co * sw * ci, -so * sw + co * cw * ci, -co * si],
        [sw * si, cw * si, ci],
    ])
    r_eci = R @ r_pqw
    v_eci = R @ v_pqw
    return r_eci, v_eci


def rv_to_kepler(r_eci, v_eci, mu=None):
    """
    Cartesian ECI (km, km/s) to Keplerian [a, e, i, omega, Omega, M] (km, rad).

    Accepts single state (3,) and (3,) or constellation (n_sats, 3) and (n_sats, 3);
    returns (6,) or (n_sats, 6). mu in km^3/s^2; if None, from params.
    """
    if mu is None:
        mu = _MU_KM
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
            # Circular: get true anomaly from position (r·v is zero so acos would give pi/2)
            cos_i = np.cos(i)
            if abs(cos_i) < 1e-9:
                nu = np.arctan2(r_eci[k, 2], r_eci[k, 0])
            else:
                nu = np.arctan2(r_eci[k, 1] / cos_i, r_eci[k, 0] / r)
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


def propagate_chief(r_chief, v_chief, time_array, backend="cpu"):
    """
    Propagate chief (single satellite) only; return ECI positions (n_times, 3) in km.
    Tensorgator expects [a, e, inc, Omega, omega, M0]; we reorder from our [a, e, i, omega, Omega, M].
    """
    import tensorgator as tg
    r_chief = np.asarray(r_chief, dtype=float).reshape(3)
    v_chief = np.asarray(v_chief, dtype=float).reshape(3)
    kepler = rv_to_kepler(r_chief, v_chief)
    const_si = np.zeros((1, 6))
    const_si[0, 0] = kepler[0] * 1000.0
    const_si[0, 1], const_si[0, 2] = kepler[1], kepler[2]
    const_si[0, 3], const_si[0, 4] = kepler[4], kepler[3]  # Omega, omega
    const_si[0, 5] = kepler[5]
    time_array = np.asarray(time_array, dtype=float)
    pos_si = tg.satellite_positions(
        time_array, const_si, backend=backend, return_frame="eci", input_type="kepler"
    )
    return (pos_si[0] / 1000.0).copy()


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

    Tensorgator expects Kepler elements as [a, e, inc, Omega, omega, M0] (RAAN before
    arg of periapsis). Our rv_to_kepler returns [a, e, i, omega, Omega, M]; we reorder
    when building the array for Tensorgator.

    Parameters
    ----------
    r_chief, v_chief : array_like
        Chief state in ECI at reference time (km, km/s), shape (3,).
    constellation_rtn : array_like
        Relative RTN state(s). Shape (6,) for one satellite or (n_sats, 6) for
        a constellation. Each row is [dR, dT, dN, dR_dot, dT_dot, dN_dot] in km and km/s.
    time_array : array_like
        Times in seconds (relative to epoch; times[0] is the epoch).
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
    # Tensorgator expects [a, e, inc, Omega, omega, M0]; we have [a, e, i, omega, Omega, M]
    constellation_si = np.zeros((n_sats, 6))
    constellation_si[:, 0] = kepler_km[:, 0] * 1000.0  # a km -> m
    constellation_si[:, 1] = kepler_km[:, 1]           # e
    constellation_si[:, 2] = kepler_km[:, 2]           # inc (i)
    constellation_si[:, 3] = kepler_km[:, 4]           # Omega (RAAN)
    constellation_si[:, 4] = kepler_km[:, 3]           # omega (arg of periapsis)
    constellation_si[:, 5] = kepler_km[:, 5]           # M0

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
