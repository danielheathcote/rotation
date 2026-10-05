"""
Prototype time-marching solver for dynamic rotation with j-dot kept.

Implements the algorithm of notes/rotation_jdot.pdf (Sections 5-7): an elastic
Earth with a hydrostatic ocean, prescribed relative angular momentum h(t) and
external torque Gamma(t), no fluid core and no damping. Only the equatorial
rotation is solved (omega_3 = 0).

The state variable is the equatorial angular momentum l = I1 omega + j + h,
which is stepped with the trapezoidal rule. At each step, the sea level
equation is iterated as in pyslfp's solve_generalised_equation, with the
rotation update replaced by the solve P omega = b_n - C j.

All quantities are in pyslfp's non-dimensional units, except times, which are
given in days. Sign conventions follow the notes: E = [[0, -1], [1, 0]].
"""

import numpy as np
from dataclasses import replace
from pyshtools import SHCoeffs, SHGrid

from pyslfp import EarthModel, EarthModelParameters, EarthState
from pyslfp.ice import IceNG


# Sidereal rotation rate (rad/s). pyslfp's default is the solar-day value.
SIDEREAL_ROTATION_FREQUENCY = 7.2921150e-5

# Rotation by +90 degrees of equatorial 2-vectors
E = np.array([[0.0, -1.0], [1.0, 0.0]])
ID = np.eye(2)


#=======================================================================
# Set-up: Earth model, background state and constants
#=======================================================================

def make_state(lmax, /, *, rotation_frequency=SIDEREAL_ROTATION_FREQUENCY):
    """
    Builds the present-day background state (PREM, ICE-7G) with a chosen
    rotation rate, as EarthState.from_defaults does.

    Args:
        lmax: Spherical harmonic truncation degree.
        rotation_frequency: Rotation rate Omega in rad/s.

    Returns:
        The EarthState.
    """
    default_parameters = EarthModelParameters.from_defaults()
    parameters = replace(
        default_parameters,
        raw_rotation_frequency=rotation_frequency,
    )
    model = EarthModel(lmax, parameters=parameters)

    ice_model = IceNG(length_scale=model.parameters.length_scale)
    ice_thickness, sea_level = ice_model.get_ice_thickness_and_sea_level(
        0.0,
        lmax,
        grid=model.grid_name,
        sampling=model.sampling,
        extend=model.extend,
    )
    return EarthState(ice_thickness, sea_level, model)


def rotation_constants(model):
    """
    Collects the constants used by the rotational part of the solver.

    Args:
        model: The pyslfp EarthModel.

    Returns:
        Dict with the moments of inertia (i1, i3, di = I3 - I1), Omega (omega),
        inertia factor beta, rotation factor r, degree-2 load and tidal Love
        numbers (k2, h2, kt2, ht2), kappa = beta r kt2, the wobble-induced
        inertia B = di kappa (b_inertia), the Euler frequency sigma_e, and the
        unit scales (time_scale in s, g, water_density).
    """
    parameters = model.parameters
    love_numbers = model.love_numbers

    i1 = parameters.equatorial_moment_of_inertia
    i3 = parameters.polar_moment_of_inertia
    di = i3 - i1
    omega = parameters.rotation_frequency
    beta = parameters.inertia_factor
    r = parameters.rotation_factor
    kt2 = love_numbers.k_t[2]
    kappa = beta * r * kt2

    return {
        "i1": i1,
        "i3": i3,
        "di": di,
        "omega": omega,
        "beta": beta,
        "r": r,
        "k2": love_numbers.k[2],
        "h2": love_numbers.h[2],
        "kt2": kt2,
        "ht2": love_numbers.h_t[2],
        "kappa": kappa,
        "b_inertia": di * kappa,
        "sigma_e": omega * di / i1,
        "time_scale": parameters.time_scale,
        "g": parameters.gravitational_acceleration,
        "water_density": parameters.water_density,
    }


def days_to_model_time(days, constants):
    """Converts a time in days to pyslfp's non-dimensional time."""
    return days * 86400.0 / constants["time_scale"]


def step_matrices(constants, dt):
    """
    The matrices of the trapezoid step, P omega_{n+1} = b_n - C j_{n+1}.

    Args:
        constants: Dict from rotation_constants.
        dt: Time step (non-dimensional).

    Returns:
        P = I1 1 - s Omega (I3 - I1) E, C = 1 + s Omega E, and s = dt / 2.
    """
    s = dt / 2
    p_mat = constants["i1"] * ID - s * constants["omega"] * constants["di"] * E
    c_mat = ID + s * constants["omega"] * E
    return p_mat, c_mat, s


def harmonic_load(model, l, m, /, *, sine=False, amplitude=1.0):
    """
    A load equal to a single real spherical harmonic (orthonormalised, as in
    pyslfp). Its integral is zero for l > 0, so it conserves mass.

    Args:
        model: The pyslfp EarthModel.
        l, m: Degree and order.
        sine: If True use the sine harmonic, else the cosine one.
        amplitude: The coefficient.

    Returns:
        The load as an SHGrid.
    """
    coeffs = np.zeros((2, model.lmax + 1, model.lmax + 1))
    coeffs[int(sine), l, m] = amplitude
    clm = SHCoeffs.from_array(
        coeffs,
        normalization=model.normalization,
        csphase=model.csphase,
    )
    return model.expand_coefficient(clm)


def omega_to_mas(omega, constants):
    """
    Converts the equatorial rotation perturbation to pole position in
    milliarcseconds, (omega_1, omega_2) / Omega. (IERS polar motion is
    (x_p, y_p) = (omega_1, -omega_2) / Omega.)
    """
    return np.asarray(omega) / constants["omega"] * (180.0 / np.pi) * 3600.0e3


#=======================================================================
# The main solver: one time step (rotation_jdot, Section 6)
#=======================================================================

def solve_step(
    state,
    constants,
    zeta_next,
    /,
    *,
    mode="dynamic",
    dt=None,
    omega_n=None,
    load_lm_n=None,
    ell_n=None,
    torque_n=None,
    h_next=None,
    gamma_n=None,
    gamma_next=None,
    omega_fixed=None,
    ocean=True,
    closed_form=False,
    rtol=1e-10,
    max_iterations=500,
):
    """
    Advances the solution by one time step, following the algorithm box of
    rotation_jdot Section 6. Two other modes solve the instantaneous problem
    with a different rotation equation, written in the same form
    P omega = b - C j, and are used for initial conditions and checks:
      - "static":  (I3 - I1) omega = j + h, the static limit (AA24);
      - "fixed":   omega = omega_fixed, a prescribed rotation.

    Args:
        state: The background EarthState.
        constants: Dict from rotation_constants.
        zeta_next: Direct load zeta_{n+1} (SHGrid).
        mode: "dynamic", "static" or "fixed".
        dt: Time step (non-dimensional); "dynamic" only.
        omega_n: Rotation at step n, used as the initial iterate (2-vector).
        load_lm_n: Converged load coefficients sigma_n, for a warm start.
        ell_n, torque_n: l_n and T_n; "dynamic" only.
        h_next: Relative angular momentum h_{n+1} (2-vector).
        gamma_n, gamma_next: Torques Gamma_n, Gamma_{n+1}; "dynamic" only.
        omega_fixed: The prescribed rotation; "fixed" only.
        ocean: If False, there is no ocean and sigma = zeta.
        closed_form: If True, use the alternative rotation update (2(d)).
        rtol: Relative tolerance on the change in ocean sea level and in omega.
        max_iterations: Iteration limit.

    Returns:
        Dict with omega, j, ell, torque (all at n+1), load_lm (sigma_{n+1}),
        sea_level (SHGrid, None without an ocean) and iterations.
    """
    model = state.model
    love_numbers = model.love_numbers
    di = constants["di"]
    beta = constants["beta"]
    r = constants["r"]
    g = constants["g"]
    water_density = constants["water_density"]

    zero = np.zeros(2)
    omega_n = zero if omega_n is None else np.asarray(omega_n, dtype=float)
    h_next = zero if h_next is None else np.asarray(h_next, dtype=float)
    gamma_n = zero if gamma_n is None else np.asarray(gamma_n, dtype=float)
    gamma_next = zero if gamma_next is None else np.asarray(gamma_next, dtype=float)

    # Love numbers shaped to act on coefficient arrays
    h_l = love_numbers.h[None, :, None]
    k_l = love_numbers.k[None, :, None]

    # --- 0. Mass conservation and the ocean ---
    # The eustatic sea level change sets the mean of xi over the ocean.
    # Without an ocean the load is just zeta, and must conserve mass.
    zeta_data = zeta_next.data
    if ocean:
        ocean_function = state.ocean_function.data
        ocean_area = state.ocean_area
        mean_slc = -model.integrate(zeta_next) / (water_density * ocean_area)
    else:
        zeta_scale = _integrate(model, np.abs(zeta_data)) + 1e-300
        assert abs(model.integrate(zeta_next)) <= 1e-10 * zeta_scale, (
            f"Without an ocean the load must conserve mass: "
            f"integral = {model.integrate(zeta_next):.3e}"
        )

    # --- 1. Form the rotation equation P omega_{n+1} = b_n - C j_{n+1} ---
    if mode == "dynamic":
        # b_n = l_n + s T_n + s (Gamma_n + Gamma_{n+1}) - C h_{n+1}
        p_mat, c_mat, s = step_matrices(constants, dt)
        b_n = ell_n + s * torque_n + s * (gamma_n + gamma_next) - c_mat @ h_next
    elif mode == "static":
        # Static limit (I3 - I1) omega = j + h, i.e. P = -(I3 - I1) 1, C = 1, b = -h
        p_mat = -di * ID
        c_mat = ID
        b_n = -h_next
    elif mode == "fixed":
        # Prescribed rotation, i.e. P = 1, C = 0, b = omega_fixed
        p_mat = ID
        c_mat = np.zeros((2, 2))
        b_n = np.asarray(omega_fixed, dtype=float)
    else:
        raise ValueError(f"Unknown mode {mode}")

    # Initialise omega^(0) = omega_n, and the load sigma^(0) = sigma_n (warm
    # start), or zeta plus the eustatic ocean load on the first step
    omega = omega_n.copy()
    if not ocean:
        load_lm = model.expand_field(zeta_next).coeffs
    elif load_lm_n is not None:
        load_lm = load_lm_n.copy()
    else:
        initial_load = zeta_data + water_density * ocean_function * mean_slc
        load_lm = _expand(model, initial_load)

    # --- 2. Iterate until the load (and omega) converge ---
    sea_level = None
    ocean_slc_prev = None
    converged = False
    for count in range(1, max_iterations + 1):

        # 2(a). Centrifugal potential psi_21 = r omega^(k)
        psi_21 = r * omega

        # 2(b). u_lm = h_l sigma_lm and phi_lm = k_l sigma_lm, adding
        #       h2^T psi_21 and k2^T psi_21 at (2, 1)
        disp_lm = h_l * load_lm
        pot_lm = k_l * load_lm
        disp_lm[:, 2, 1] += constants["ht2"] * psi_21
        pot_lm[:, 2, 1] += constants["kt2"] * psi_21

        # 2(c). j^(k) = (I3 - I1) beta phi_21, with phi excluding psi
        j = di * beta * pot_lm[:, 2, 1]

        # 2(d). Rotation update omega^(k+1) = P^-1 (b_n - C j^(k))
        if not closed_form:
            omega_new = np.linalg.solve(p_mat, b_n - c_mat @ j)
        else:
            # ALTERNATIVE (not the algorithm of the notes): write
            # j = A sigma + B omega, with A sigma = (I3 - I1) beta k2 sigma_21 and
            # B = (I3 - I1) kappa, and solve (P + C B) omega = b_n - C A sigma
            # exactly, as pyslfp's static solver does. The (2, 1) fields and j
            # are then recomputed with the new omega.
            a_sigma = di * beta * constants["k2"] * load_lm[:, 2, 1]
            omega_new = np.linalg.solve(
                p_mat + constants["b_inertia"] * c_mat,
                b_n - c_mat @ a_sigma,
            )
            psi_21 = r * omega_new
            disp_lm[:, 2, 1] = constants["h2"] * load_lm[:, 2, 1] + constants["ht2"] * psi_21
            pot_lm[:, 2, 1] = constants["k2"] * load_lm[:, 2, 1] + constants["kt2"] * psi_21
            j = di * beta * pot_lm[:, 2, 1]

        omega_scale = np.max(np.abs(omega_new))
        omega_err = np.max(np.abs(omega_new - omega)) / omega_scale if omega_scale > 0 else 0.0
        omega = omega_new

        # Without an ocean the load is fixed, so only omega is iterated
        if not ocean:
            if omega_err <= rtol:
                converged = True
                break
            continue

        # 2(e). Add r omega^(k+1) to phi_21, form xi with mass conserved, and
        #       set sigma^(k+1) = zeta_{n+1} + rho_w C xi
        pot_lm[:, 2, 1] += r * omega_new
        slc_lm = -(disp_lm + pot_lm / g)
        slc = _synthesise(model, slc_lm)
        ocean_mean = _integrate(model, ocean_function * slc) / ocean_area
        slc += mean_slc - ocean_mean
        ocean_slc = ocean_function * slc

        if ocean_slc_prev is not None:
            slc_scale = np.max(np.abs(ocean_slc))
            slc_err = np.max(np.abs(ocean_slc - ocean_slc_prev)) / slc_scale if slc_scale > 0 else 0.0
            if slc_err <= rtol and omega_err <= rtol:
                converged = True
                sea_level = slc
                break

        ocean_slc_prev = ocean_slc
        sea_level = slc
        load_lm = _expand(model, zeta_data + water_density * ocean_slc)

    assert converged, f"solve_step ({mode}) did not converge in {max_iterations} iterations"

    # --- 3. Store omega_{n+1}, j_{n+1}, l_{n+1} = I1 omega + j + h and
    #        T_{n+1} = Omega E [(I3 - I1) omega - j - h] ---
    ell = constants["i1"] * omega + j + h_next
    torque = constants["omega"] * E @ (di * omega - j - h_next)

    if sea_level is not None:
        sea_level = SHGrid.from_array(sea_level, grid=model.grid)

    return {
        "omega": omega,
        "j": j,
        "ell": ell,
        "torque": torque,
        "load_lm": load_lm,
        "sea_level": sea_level,
        "iterations": count,
    }


#=======================================================================
# Initial conditions (rotation_jdot, Section 7)
#=======================================================================

def initial_conditions(
    state,
    constants,
    zeta_0,
    /,
    *,
    start="rest",
    omega_0=None,
    h_0=None,
    ocean=True,
    closed_form=False,
    rtol=1e-10,
):
    """
    The state at t = 0.
      - "rest" (or a prescribed wobble): omega_0 is given (zero by default),
        and j_0 comes from a solve with omega held at omega_0.
      - "equilibrium": a static solve with the load zeta_0 and h_0, which is
        pyslfp's static solver with angular_momentum_change = -h_0 (but
        without the axial rotation). T_0 is then zero to solver tolerance.

    Args:
        state: The background EarthState.
        constants: Dict from rotation_constants.
        zeta_0: Direct load at t = 0 (SHGrid).
        start: "rest" or "equilibrium".
        omega_0: Prescribed initial rotation (2-vector) for "rest".
        h_0: Relative angular momentum at t = 0.
        ocean, closed_form, rtol: As for solve_step.

    Returns:
        The solve_step dict at t = 0.
    """
    if start == "equilibrium":
        return solve_step(
            state,
            constants,
            zeta_0,
            mode="static",
            h_next=h_0,
            ocean=ocean,
            closed_form=closed_form,
            rtol=rtol,
        )
    elif start == "rest":
        omega_0 = np.zeros(2) if omega_0 is None else omega_0
        return solve_step(
            state,
            constants,
            zeta_0,
            mode="fixed",
            omega_fixed=omega_0,
            omega_n=omega_0,
            h_next=h_0,
            ocean=ocean,
            rtol=rtol,
        )
    raise ValueError(f"Unknown start {start}")


#=======================================================================
# Time march
#=======================================================================

def march(
    state,
    constants,
    zeta,
    dt_days,
    nt,
    /,
    *,
    h=None,
    gamma=None,
    start="rest",
    omega_0=None,
    store_steps=(),
    ocean=True,
    closed_form=False,
    rtol=1e-10,
):
    """
    Marches the dynamic problem over nt time levels t_n = n dt.

    The direct load is a function zeta(t_days) -> SHGrid. (Passing it as a sum
    of fixed patterns times time series would cost about the same: the load
    sigma = zeta + rho_w C xi is transformed again at every iteration anyway,
    so evaluating zeta(t) only adds one cheap grid sum per step.)

    Args:
        state: The background EarthState.
        constants: Dict from rotation_constants.
        zeta: Function of time in days returning the direct load (SHGrid), or
            None for no load.
        dt_days: Time step in days.
        nt: Number of time levels, including t = 0.
        h, gamma: Relative angular momentum and torque, (nt, 2) arrays in
            non-dimensional units, or None.
        start, omega_0: Initial condition, see initial_conditions.
        store_steps: Steps at which the sea level is kept.
        ocean, closed_form, rtol: As for solve_step.

    Returns:
        Dict with t_days, omega, j, ell, torque (each (nt, 2)), iterations
        (nt,), and sea_level, a dict of SHGrids keyed by step.
    """
    if zeta is None:
        zero_load = state.model.zero_grid()
        zeta = lambda t: zero_load
    h = np.zeros((nt, 2)) if h is None else np.asarray(h, dtype=float)
    gamma = np.zeros((nt, 2)) if gamma is None else np.asarray(gamma, dtype=float)
    assert h.shape == (nt, 2), f"h has shape {h.shape}, expected {(nt, 2)}"
    assert gamma.shape == (nt, 2), f"gamma has shape {gamma.shape}, expected {(nt, 2)}"

    t_days = np.arange(nt) * dt_days
    dt = days_to_model_time(dt_days, constants)

    omega = np.zeros((nt, 2))
    j = np.zeros((nt, 2))
    ell = np.zeros((nt, 2))
    torque = np.zeros((nt, 2))
    iterations = np.zeros(nt, dtype=int)
    sea_level = {}

    step = initial_conditions(
        state,
        constants,
        zeta(t_days[0]),
        start=start,
        omega_0=omega_0,
        h_0=h[0],
        ocean=ocean,
        closed_form=closed_form,
        rtol=rtol,
    )

    for n in range(nt):
        if n > 0:
            step = solve_step(
                state,
                constants,
                zeta(t_days[n]),
                mode="dynamic",
                dt=dt,
                omega_n=omega[n - 1],
                load_lm_n=step["load_lm"],
                ell_n=ell[n - 1],
                torque_n=torque[n - 1],
                h_next=h[n],
                gamma_n=gamma[n - 1],
                gamma_next=gamma[n],
                ocean=ocean,
                closed_form=closed_form,
                rtol=rtol,
            )
        omega[n] = step["omega"]
        j[n] = step["j"]
        ell[n] = step["ell"]
        torque[n] = step["torque"]
        iterations[n] = step["iterations"]
        if n in store_steps:
            sea_level[n] = step["sea_level"]

    return {
        "t_days": t_days,
        "omega": omega,
        "j": j,
        "ell": ell,
        "torque": torque,
        "iterations": iterations,
        "sea_level": sea_level,
    }


#=======================================================================
# Transform and quadrature helpers (as in pyslfp's solver)
#=======================================================================

def _expand(model, data):
    """Forward transform of grid values to a coefficient array."""
    grid = SHGrid.from_array(data, grid=model.grid)
    return model.expand_field(grid).coeffs


def _synthesise(model, coeffs):
    """Inverse transform of a coefficient array to grid values."""
    clm = SHCoeffs.from_array(
        coeffs,
        normalization=model.normalization,
        csphase=model.csphase,
    )
    return model.expand_coefficient(clm).data


def _integrate(model, data):
    """Surface integral of grid values."""
    return model.integrate(SHGrid.from_array(data, grid=model.grid))
