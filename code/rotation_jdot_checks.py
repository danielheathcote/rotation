"""
Checks for the prototype solver in rotation_jdot.py.

The reduced model of rotation_jdot Section 3: by linearity, the inertia
perturbation is j = j^zeta + (I3 - I1) K omega, where j^zeta is the response to
the direct load with the ocean responding and omega = 0, and K is the 2 x 2
rotational feedback matrix (K = kappa 1 without an ocean). The same trapezoid
step on l then needs only a 2 x 2 solve per step. Also the analytic results
of Sections 4 and 8 used as benchmarks.

Frequencies are non-dimensional unless named *_per_day; times are in days.
"""

import numpy as np

from rotation_jdot import E, ID, solve_step, step_matrices, days_to_model_time


#=======================================================================
# Feedback matrix and analytic results
#=======================================================================

def feedback_matrix(state, constants, /, *, ocean=True, rtol=1e-12):
    """
    The feedback matrix K, from two solves with a prescribed unit rotation and
    no load: column alpha is j / (I3 - I1) for omega = e_alpha.

    Args:
        state: The background EarthState.
        constants: Dict from rotation_constants.
        ocean: If False, K = kappa 1.
        rtol: Solver tolerance.

    Returns:
        K (2 x 2).
    """
    zero_load = state.model.zero_grid()
    columns = []
    for unit_rotation in ID:
        step = solve_step(
            state,
            constants,
            zero_load,
            mode="fixed",
            omega_fixed=unit_rotation,
            omega_n=unit_rotation,
            ocean=ocean,
            rtol=rtol,
        )
        columns.append(step["j"] / constants["di"])
    return np.array(columns).T


def per_day(frequency, constants):
    """Converts a non-dimensional angular frequency to rad/day."""
    return frequency / constants["time_scale"] * 86400.0


def chandler_frequency(constants, k_mat):
    """
    Free-wobble frequency with j-dot kept (rotation_jdot Section 4.3),
    sigma'_CW = Omega (I3 - I1) sqrt(det(1 - K) / det(I1 1 + (I3 - I1) K)).
    """
    i1 = constants["i1"]
    di = constants["di"]
    ratio = np.linalg.det(ID - k_mat) / np.linalg.det(i1 * ID + di * k_mat)
    return constants["omega"] * di * np.sqrt(ratio)


def discrete_chandler_frequency(constants, k_mat, dt_days):
    """
    Free-wobble frequency of the discrete trapezoid step (no load), from the
    eigenvalues exp(+-i theta) of the one-step map: sigma = theta / dt.
    """
    dt = days_to_model_time(dt_days, constants)
    step_map = _free_step_map(constants, k_mat, dt)
    theta = np.max(np.abs(np.angle(np.linalg.eigvals(step_map))))
    return theta / dt


def periodic_ratio(constants, nu):
    """
    Forced response relative to static for circular forcing j^zeta ~ e^{i nu t}
    with K = kappa 1 (rotation_jdot Section 8, test 3):
    (1 + nu / Omega) / (1 - nu / sigma'_CW). nu > 0 is prograde.
    """
    sigma_cw = chandler_frequency(constants, constants["kappa"] * ID)
    return (1 + nu / constants["omega"]) / (1 - nu / sigma_cw)


def step_jump_ratio(constants):
    """
    Instantaneous jump in omega from a step in j^zeta, relative to the eventual
    static response, for K = kappa 1 (Section 8, test 2):
    -(I3 - I1)(1 - kappa) / (I1 + B).
    """
    return -constants["di"] * (1 - constants["kappa"]) / (constants["i1"] + constants["b_inertia"])


def forced_response(constants, k_mat, nu, j_hat):
    """
    Steady response omega = Re(omega_hat e^{i nu t}) to j^zeta = Re(j_hat e^{i nu t}),
    from [i nu (I1 1 + (I3 - I1) K) - Omega (I3 - I1) E (1 - K)] omega_hat
    = -(i nu 1 + Omega E) j_hat.
    """
    i1 = constants["i1"]
    di = constants["di"]
    omega = constants["omega"]
    lhs = 1j * nu * (i1 * ID + di * k_mat) - omega * di * E @ (ID - k_mat)
    rhs = -(1j * nu * ID + omega * E) @ j_hat
    return np.linalg.solve(lhs, rhs)


#=======================================================================
# Reduced time march
#=======================================================================

def march_reduced(
    constants,
    k_mat,
    j_zeta,
    dt_days,
    /,
    *,
    h=None,
    gamma=None,
    omega_0=None,
):
    """
    The trapezoid step on l with j = j^zeta + (I3 - I1) K omega, solved directly:
    (P + (I3 - I1) C K) omega_{n+1} = b_n - C j^zeta_{n+1}.

    Args:
        constants: Dict from rotation_constants.
        k_mat: Feedback matrix K.
        j_zeta: (nt, 2) inertia perturbation from the load with omega = 0.
        dt_days: Time step in days.
        h, gamma: (nt, 2) relative angular momentum and torque, or None.
        omega_0: Initial rotation (zero by default).

    Returns:
        Dict with t_days, omega, j, ell, torque (each (nt, 2)).
    """
    nt = len(j_zeta)
    h = np.zeros((nt, 2)) if h is None else h
    gamma = np.zeros((nt, 2)) if gamma is None else gamma
    i1 = constants["i1"]
    di = constants["di"]

    dt = days_to_model_time(dt_days, constants)
    p_mat, c_mat, s = step_matrices(constants, dt)
    lhs = p_mat + di * c_mat @ k_mat

    omega = np.zeros((nt, 2))
    j = np.zeros((nt, 2))
    ell = np.zeros((nt, 2))
    torque = np.zeros((nt, 2))

    if omega_0 is not None:
        omega[0] = omega_0
    for n in range(nt):
        if n > 0:
            b_n = ell[n - 1] + s * torque[n - 1] + s * (gamma[n - 1] + gamma[n]) - c_mat @ h[n]
            omega[n] = np.linalg.solve(lhs, b_n - c_mat @ j_zeta[n])
        j[n] = j_zeta[n] + di * k_mat @ omega[n]
        ell[n] = i1 * omega[n] + j[n] + h[n]
        torque[n] = constants["omega"] * E @ (di * omega[n] - j[n] - h[n])

    return {
        "t_days": np.arange(nt) * dt_days,
        "omega": omega,
        "j": j,
        "ell": ell,
        "torque": torque,
    }


#=======================================================================
# Fitting and diagnostics
#=======================================================================

def fit_forced_response(t_days, omega, nu_per_day, sigma_per_day):
    """
    Least-squares fit of m = omega_1 + i omega_2 by
    c_1 e^{i nu t} + c_2 e^{-i nu t} + c_3 e^{i sigma t} + c_4 e^{-i sigma t},
    separating the forced (nu) and free (sigma) parts. The negative-frequency
    terms allow for elliptical motion when K is anisotropic.

    Returns:
        The four complex coefficients.
    """
    m = omega[:, 0] + 1j * omega[:, 1]
    basis = np.stack(
        [
            np.exp(1j * nu_per_day * t_days),
            np.exp(-1j * nu_per_day * t_days),
            np.exp(1j * sigma_per_day * t_days),
            np.exp(-1j * sigma_per_day * t_days),
        ],
        axis=1,
    )
    coefficients, *_ = np.linalg.lstsq(basis, m, rcond=None)
    return coefficients


def wobble_period_days(t_days, omega):
    """
    Mean period of a free wobble from the unwrapped polar angle of omega,
    using whole turns only.
    """
    angle = np.unwrap(np.arctan2(omega[:, 1], omega[:, 0]))
    turns = int(np.floor((angle[-1] - angle[0]) / (2 * np.pi)))
    assert turns >= 1, "Run is shorter than one wobble period"
    t_end = np.interp(angle[0] + 2 * np.pi * turns, angle, t_days)
    return (t_end - t_days[0]) / turns


def budget_residual(result, gamma, dt_days, constants):
    """
    Relative residual of the discrete angular-momentum budget
    l_{n+1} - l_n = s (T_n + T_{n+1} + Gamma_n + Gamma_{n+1}).
    """
    s = days_to_model_time(dt_days, constants) / 2
    ell = result["ell"]
    torque = result["torque"]
    residual = ell[1:] - ell[:-1] - s * (torque[1:] + torque[:-1] + gamma[1:] + gamma[:-1])
    return np.max(np.abs(residual)) / np.max(np.abs(ell))


def _free_step_map(constants, k_mat, dt):
    """The one-step map omega_n -> omega_{n+1} of the reduced model with no forcing."""
    i1 = constants["i1"]
    di = constants["di"]
    p_mat, c_mat, s = step_matrices(constants, dt)
    lhs = p_mat + di * c_mat @ k_mat
    rhs = i1 * ID + di * k_mat + s * constants["omega"] * di * E @ (ID - k_mat)
    return np.linalg.solve(lhs, rhs)
