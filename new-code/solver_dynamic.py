"""
Forward solver for dynamic rotational feedbacks in the elastic sea-level problem.

Implements:
  - The instantaneous map j = J(zeta, omega), a.k.a. the inertia perturbation
    of a static sea-level solve with direct load zeta and rotation held at omega;
  - The right-hand side omega_dot = f(omega, t) of the Liouville equation, with
    j_dot = J(zeta_dot, omega_dot) found by fixed-point iteration;
  - Initial conditions and time integration.

Assumptions: Elastic Earth, hydrostatic ocean on fixed shorelines, prescribed relative angular
momentum h(t) and torque Gamma(t), I1 = I2. Only the equatorial rotation is solved for.
The fluid core is included to leading order, through the mantle's moment A_m in the
wobble term, and damping is optional, set by the quality factor Q of the free wobble.

All quantities, including time, are in pyslfp's non-dimensional units.
E = [[0, -1], [1, 0]] rotates equatorial 2-vectors by +90 degrees.
"""

import numpy as np
from dataclasses import replace
from pyshtools import SHCoeffs, SHGrid

from pyslfp import EarthModel, EarthModelParameters, EarthState
from pyslfp.ice import IceNG


# Sidereal rotation rate (rad/s). pyslfp's default is the solar-day value.
SIDEREAL_ROTATION_FREQUENCY = 7.2921150e-5

# Equatorial moment of PREM's fluid outer core as a fraction of I1, A_f / I1
# (from integrating PREM's outer-core density; not provided by pyslfp)
OUTER_CORE_FRACTION = 0.1132

# Default quality factor of the free wobble (the observed Chandler wobble has Q of about 100)
DEFAULT_QUALITY_FACTOR = 100.0

E = np.array([[0.0, -1.0], [1.0, 0.0]])
ZERO_2 = np.zeros(2)


#=======================================================================
# Set-up: Earth model, background state and constants
#=======================================================================

def make_state(lmax, /, *, rotation_frequency=SIDEREAL_ROTATION_FREQUENCY):
    """
    Builds the present-day background state (PREM, ICE-7G).

    Args:
        lmax: Spherical harmonic truncation degree.
        rotation_frequency: Rotation rate Omega in rad/s.

    Returns:
        The pyslfp EarthState.
    """
    parameters = replace(
        EarthModelParameters.from_defaults(),
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


def rotation_constants(model, /, *, core_fraction=OUTER_CORE_FRACTION):
    """
    Collects the various required constants.

    Args:
        model: The pyslfp EarthModel.
        core_fraction: A_f / I1, the fluid core's share of the equatorial moment.
            Setting it to zero gives an Earth without a fluid core.

    Returns:
        Dict with the moments of inertia (i1, i3, di = I3 - I1), the wobble
        inertia a_w = I1 - A_f (the mantle moment A_m), Omega (omega), the
        constants r and beta, kappa = beta r k2T, the load Love numbers (h_l,
        k_l, shaped to act on coefficient arrays), the degree-2 tidal Love
        numbers (ht2, kt2), g, water_density, and the time and length scales.
    """
    parameters = model.parameters
    love_numbers = model.love_numbers
    i1 = parameters.equatorial_moment_of_inertia
    i3 = parameters.polar_moment_of_inertia

    return {
        "i1": i1,
        "i3": i3,
        "di": i3 - i1,
        "a_w": i1 * (1.0 - core_fraction),
        "omega": parameters.rotation_frequency,
        "r": parameters.rotation_factor,
        "beta": parameters.inertia_factor,
        "kappa": parameters.inertia_factor * parameters.rotation_factor * love_numbers.k_t[2],
        "h_l": love_numbers.h[None, :, None],
        "k_l": love_numbers.k[None, :, None],
        "ht2": love_numbers.h_t[2],
        "kt2": love_numbers.k_t[2],
        "g": parameters.gravitational_acceleration,
        "water_density": parameters.water_density,
        "time_scale": parameters.time_scale,
        "length_scale": parameters.length_scale,
    }


def days_to_model_time(days, constants):
    """Converts a time in days to pyslfp's non-dimensional time."""
    return np.asarray(days) * 86400.0 / constants["time_scale"]


def model_time_to_days(t, constants):
    """Converts pyslfp's non-dimensional time to days."""
    return np.asarray(t) * constants["time_scale"] / 86400.0


def omega_to_mas(omega, constants):
    """
    Converts the equatorial rotation perturbation to pole position in
    milliarcseconds, (omega_1, omega_2) / Omega. (IERS polar motion is
    (x_p, y_p) = (omega_1, -omega_2) / Omega.)
    """
    return np.asarray(omega) / constants["omega"] * (180.0 / np.pi) * 3600.0e3


def harmonic_load(model, l, m, /, *, sine=False, amplitude=1.0):
    """
    A load equal to one real spherical harmonic (orthonormal, as in pyslfp).
    It conserves mass for l > 0.

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
    return _synthesise(model, coeffs, as_grid=True)


#=======================================================================
# Prescribed forcings: each has a value(t) and a rate(t)
#=======================================================================

class Amplitude:
    """A scalar function of time a(t), with its rate a_dot(t)."""

    def value(self, t):
        raise NotImplementedError

    def rate(self, t):
        raise NotImplementedError

    def __add__(self, other):
        return Sum(self, _as_amplitude(other))

    __radd__ = __add__

    def __mul__(self, other):
        return Product(self, _as_amplitude(other))

    __rmul__ = __mul__

    def __neg__(self):
        return Product(self, Constant(-1.0))

    def __sub__(self, other):
        return Sum(self, -_as_amplitude(other))

    def __rsub__(self, other):
        return Sum(_as_amplitude(other), -self)


class Constant(Amplitude):
    """a(t) = c."""

    def __init__(self, c=1.0):
        self.c = c

    def value(self, t):
        return self.c

    def rate(self, t):
        return 0.0


class LinearRamp(Amplitude):
    """a(t) rises linearly from 0 at t_on to 1 at t_on + duration, then stays at 1."""

    def __init__(self, duration, t_on=0.0):
        self.duration = duration
        self.t_on = t_on

    def value(self, t):
        return float(np.clip((t - self.t_on) / self.duration, 0.0, 1.0))

    def rate(self, t):
        inside = self.t_on <= t < self.t_on + self.duration
        return 1.0 / self.duration if inside else 0.0


class CosineRamp(Amplitude):
    """
    a(t) rises smoothly from 0 at t_on to 1 at t_on + duration,
    a = (1 - cos(pi s)) / 2 with s = (t - t_on) / duration. A smoothed step.
    """

    def __init__(self, duration, t_on=0.0):
        self.duration = duration
        self.t_on = t_on

    def value(self, t):
        s = np.clip((t - self.t_on) / self.duration, 0.0, 1.0)
        return 0.5 * (1.0 - np.cos(np.pi * s))

    def rate(self, t):
        if not self.t_on <= t < self.t_on + self.duration:
            return 0.0
        s = (t - self.t_on) / self.duration
        return 0.5 * np.pi / self.duration * np.sin(np.pi * s)


class Sinusoid(Amplitude):
    """a(t) = sin(frequency t + phase)."""

    def __init__(self, frequency, phase=0.0):
        self.frequency = frequency
        self.phase = phase

    def value(self, t):
        return np.sin(self.frequency * t + self.phase)

    def rate(self, t):
        return self.frequency * np.cos(self.frequency * t + self.phase)


class Sum(Amplitude):
    """a(t) = a1(t) + a2(t)."""

    def __init__(self, first, second):
        self.first = first
        self.second = second

    def value(self, t):
        return self.first.value(t) + self.second.value(t)

    def rate(self, t):
        return self.first.rate(t) + self.second.rate(t)


class Product(Amplitude):
    """a(t) = a1(t) a2(t)."""

    def __init__(self, first, second):
        self.first = first
        self.second = second

    def value(self, t):
        return self.first.value(t) * self.second.value(t)

    def rate(self, t):
        return (
            self.first.rate(t) * self.second.value(t)
            + self.first.value(t) * self.second.rate(t)
        )


class Load:
    """A direct surface load zeta(x, t): value(t) and rate(t) return SHGrids."""

    def value(self, t):
        raise NotImplementedError

    def rate(self, t):
        raise NotImplementedError


class SeparableLoad(Load):
    """
    A load that is a sum of separable terms, zeta(x, t) = sum_i p_i(x) a_i(t),
    from fixed patterns p_i (SHGrids) and amplitudes a_i (Amplitude objects).
    """

    def __init__(self, terms):
        assert len(terms) > 0, "SeparableLoad needs at least one (pattern, amplitude) term"
        self.terms = [(pattern, _as_amplitude(amplitude)) for pattern, amplitude in terms]

    def value(self, t):
        data = sum(amplitude.value(t) * pattern.data for pattern, amplitude in self.terms)
        return SHGrid.from_array(data, grid=self.terms[0][0].grid)

    def rate(self, t):
        data = sum(amplitude.rate(t) * pattern.data for pattern, amplitude in self.terms)
        return SHGrid.from_array(data, grid=self.terms[0][0].grid)


class VectorForcing:
    """
    An equatorial 2-vector forcing v(t), for h(t) or Gamma(t): value(t) and
    rate(t) return 2-vectors.
    """

    def value(self, t):
        raise NotImplementedError

    def rate(self, t):
        raise NotImplementedError


class SeparableVector(VectorForcing):
    """
    A 2-vector forcing that is a sum of separable terms, v(t) = sum_i v_i a_i(t),
    from fixed 2-vectors v_i and amplitudes a_i (Amplitude objects).
    """

    def __init__(self, terms):
        self.terms = [(np.asarray(vector, dtype=float), _as_amplitude(amplitude))
                      for vector, amplitude in terms]

    def value(self, t):
        return sum((amplitude.value(t) * vector for vector, amplitude in self.terms), ZERO_2.copy())

    def rate(self, t):
        return sum((amplitude.rate(t) * vector for vector, amplitude in self.terms), ZERO_2.copy())


def _as_amplitude(a):
    """Wraps a number as a Constant amplitude."""
    return a if isinstance(a, Amplitude) else Constant(a)


#=======================================================================
# The steps of the instantaneous solve j = J(zeta, omega)
#=======================================================================

def centrifugal_potential(constants, omega):
    """Step 1: psi_21 = r omega. All other coefficients of psi are zero."""
    return constants["r"] * np.asarray(omega, dtype=float)


def eustatic_level(state, constants, zeta):
    """Step 2: xi_bar = -(1 / (rho_w A_o)) int zeta dS, the mass-conserving ocean mean."""
    return -state.model.integrate(zeta) / (constants["water_density"] * state.ocean_area)


def elastic_response(constants, load_lm, psi_21):
    """
    Step 4(b): u_lm = h_l sigma_lm and phi_lm = k_l sigma_lm, then add the
    response to psi at (2, 1): u_21 += h2T psi_21, phi_21 += k2T psi_21.
    phi excludes psi.

    Args:
        constants: Dict from rotation_constants.
        load_lm: Coefficients sigma_lm of the total load.
        psi_21: The centrifugal potential coefficients (cosine, sine) at (2, 1).

    Returns:
        Coefficient arrays of the radial displacement u_r and of phi.
    """
    displacement_lm = constants["h_l"] * load_lm
    potential_lm = constants["k_l"] * load_lm
    displacement_lm[:, 2, 1] += constants["ht2"] * psi_21
    potential_lm[:, 2, 1] += constants["kt2"] * psi_21
    return displacement_lm, potential_lm


def inertia_perturbation(constants, potential_lm):
    """Step 4(c): j = (I3 - I1) beta phi_21 (MacCullagh's formula), phi excluding psi."""
    return constants["di"] * constants["beta"] * potential_lm[:, 2, 1]


def sea_level_from_response(state, constants, displacement_lm, potential_lm, psi_21, mean_sea_level):
    """
    Step 4(d): xi_lm = -(u_lm + (phi_lm + psi_lm) / g), synthesised on the grid,
    then shifted by a constant so that (1 / A_o) int C xi dS = xi_bar.

    Returns:
        Grid values of xi.
    """
    total_potential_lm = potential_lm.copy()
    total_potential_lm[:, 2, 1] += psi_21
    sea_level_lm = -(displacement_lm + total_potential_lm / constants["g"])
    sea_level = _synthesise(state.model, sea_level_lm)

    ocean_function = state.ocean_function.data
    ocean_mean = _integrate(state.model, ocean_function * sea_level) / state.ocean_area
    return sea_level + (mean_sea_level - ocean_mean)


def total_load(state, constants, zeta, sea_level):
    """Steps 3 and 4(e): sigma = zeta + rho_w C xi, as grid values."""
    return zeta.data + constants["water_density"] * state.ocean_function.data * sea_level


def solve_instantaneous(
    state,
    constants,
    zeta,
    omega,
    /,
    *,
    ocean=True,
    sea_level_guess=None,
    rtol=1e-10,
    max_iterations=500,
):
    """
    The instantaneous solve j = J(zeta, omega): the static sea-level
    problem with direct load zeta and the rotation held at omega.

    Args:
        state: The background EarthState.
        constants: Dict from rotation_constants.
        zeta: Direct load (SHGrid).
        omega: Equatorial rotation, held fixed (2-vector).
        ocean: If False, there is no ocean and sigma = zeta.
        sea_level_guess: Grid values of xi from an earlier solve, used as the
            warm start sigma^(0) = zeta + rho_w C xi_guess.
        rtol: Relative tolerance on the change in C xi (step 4(f)).
        max_iterations: Iteration limit.

    Returns:
        Dict with j (2-vector), sea_level (SHGrid, or None without an ocean)
        and iterations.
    """
    model = state.model

    # 1. Centrifugal potential: psi_21 = r omega
    psi_21 = centrifugal_potential(constants, omega)

    # Without an ocean, sigma = zeta, and steps 4(b) and 4(c) give j directly
    if not ocean:
        zeta_scale = _integrate(model, np.abs(zeta.data)) + 1e-300
        assert abs(model.integrate(zeta)) <= 1e-10 * zeta_scale, (
            f"Without an ocean the load must conserve mass: integral = {model.integrate(zeta):.3e}"
        )
        load_lm = _expand(model, zeta.data)
        _, potential_lm = elastic_response(constants, load_lm, psi_21)
        j = inertia_perturbation(constants, potential_lm)
        return {"j": j, "sea_level": None, "iterations": 0}

    # 2. Eustatic level: xi_bar = -(1 / (rho_w A_o)) int zeta dS
    mean_sea_level = eustatic_level(state, constants, zeta)

    # 3. Initial load: sigma^(0) = zeta + rho_w C xi_bar (or a warm start from xi_guess)
    if sea_level_guess is None:
        sea_level_guess = np.full_like(zeta.data, mean_sea_level)
    load = total_load(state, constants, zeta, sea_level_guess)

    # 4. Iterate on the load
    ocean_function = state.ocean_function.data
    ocean_sea_level_previous = None
    converged = False
    for count in range(1, max_iterations + 1):

        # 4(a). Expand sigma^(k)
        load_lm = _expand(model, load)

        # 4(b). u_lm = h_l sigma_lm, phi_lm = k_l sigma_lm, plus h2T psi_21, k2T psi_21 at (2, 1)
        displacement_lm, potential_lm = elastic_response(constants, load_lm, psi_21)

        # 4(c). j^(k) = (I3 - I1) beta phi_21
        j = inertia_perturbation(constants, potential_lm)

        # 4(d). xi = -(u + (phi + psi) / g), shifted so that its ocean mean is xi_bar
        sea_level = sea_level_from_response(
            state,
            constants,
            displacement_lm,
            potential_lm,
            psi_21,
            mean_sea_level,
        )

        # 4(e). sigma^(k+1) = zeta + rho_w C xi
        load = total_load(state, constants, zeta, sea_level)

        # 4(f). Stop when the relative change in C xi is below tolerance
        ocean_sea_level = ocean_function * sea_level
        if ocean_sea_level_previous is not None:
            scale = np.max(np.abs(ocean_sea_level))
            change = np.max(np.abs(ocean_sea_level - ocean_sea_level_previous))
            if change <= rtol * scale:
                converged = True
                break
        ocean_sea_level_previous = ocean_sea_level

    assert converged, f"solve_instantaneous did not converge in {max_iterations} iterations"

    # 5. Return j, and xi
    return {
        "j": j,
        "sea_level": SHGrid.from_array(sea_level, grid=model.grid),
        "iterations": count,
    }


#=======================================================================
# The forward right-hand side omega_dot = f(omega, t)
#=======================================================================

def damping_constant(constants, quality_factor):
    """
    The damping constant gamma that gives the free wobble a quality factor Q.
    N.B. eps = epsilon.
    The free wobble has complex frequency sigma'(1 + i eps) / (1 + eps^2), so
    Q = (1 + eps^2) / (2 eps), i.e. eps = Q - sqrt(Q^2 - 1), and
    gamma = eps (A_w + (I3 - I1) kappa) / Omega. This is exact without an ocean;
    with the ocean the resulting Q differs from the target by about 1e-4.

    Args:
        constants: Dict from rotation_constants.
        quality_factor: The target Q, or None for no damping.

    Returns:
        gamma (zero if quality_factor is None).
    """
    if quality_factor is None:
        return 0.0
    assert quality_factor > 1, f"The quality factor must exceed 1, not {quality_factor}"
    eps = quality_factor - np.sqrt(quality_factor**2 - 1)
    return eps * (constants["a_w"] + constants["di"] * constants["kappa"]) / constants["omega"]


def wobble_matrix(constants, damping):
    """W = A_w 1 - Omega gamma E, the matrix multiplying omega_dot in step 3."""
    return constants["a_w"] * np.eye(2) - constants["omega"] * damping * E


def rotation_term(constants, omega, j, h):
    """Step 2: T = Omega E [(I3 - I1) omega - j - h]."""
    return constants["omega"] * E @ (constants["di"] * omega - j - h)


def rotation_rate(wobble, rotation_term, torque, h_dot, j_dot):
    """Step 3 (initial and part b): omega_dot = W^-1 (T + Gamma - h_dot - j_dot)."""
    return np.linalg.solve(wobble, rotation_term + torque - h_dot - j_dot)


def forward_rhs(
    state,
    constants,
    load,
    /,
    *,
    angular_momentum=None,
    torque=None,
    quality_factor=DEFAULT_QUALITY_FACTOR,
    ocean=True,
    rtol=1e-10,
    inner_rtol=1e-12,
    max_inner_iterations=50,
):
    """
    Builds the forward right-hand side f(t, omega), for use with
    any time integrator. The fluid core enters through the wobble inertia
    constants["a_w"], and damping through quality_factor.

    Args:
        state: The background EarthState.
        constants: Dict from rotation_constants.
        load: The direct load, a Load (value(t) and rate(t) give zeta and zeta_dot).
        angular_momentum: Relative angular momentum h, a VectorForcing (value and
            rate give h and h_dot), or None for zero.
        torque: External torque Gamma, a VectorForcing, or None for zero.
        quality_factor: Quality factor Q of the free wobble, or None for no damping.
        ocean: If False, there is no ocean.
        rtol: Tolerance of each instantaneous solve.
        inner_rtol: Relative tolerance on the change in omega_dot (step 3).
        max_inner_iterations: Limit on the iteration of step 3.

    Returns:
        The function f(t, omega) -> omega_dot. Its attribute inner_iterations
        records the number of step-3 iterations of each call, and damping holds gamma.
    """
    # W = A_w 1 - Omega gamma E, with the core in A_w and the damping in gamma
    damping = damping_constant(constants, quality_factor)
    wobble = wobble_matrix(constants, damping)

    # Warm starts, kept separately for the solve with zeta and the solve with zeta_dot
    warm_start = {"value": None, "rate": None}

    def rhs(t, omega):
        omega = np.asarray(omega, dtype=float)
        h = ZERO_2 if angular_momentum is None else angular_momentum.value(t)
        h_dot = ZERO_2 if angular_momentum is None else angular_momentum.rate(t)
        torque_value = ZERO_2 if torque is None else torque.value(t)

        # --- 1. j = J(zeta(t), omega) ---
        solve = solve_instantaneous(
            state,
            constants,
            load.value(t),
            omega,
            ocean=ocean,
            sea_level_guess=warm_start["value"],
            rtol=rtol,
        )
        if ocean:
            warm_start["value"] = solve["sea_level"].data
        j = solve["j"]

        # --- 2. T = Omega E [(I3 - I1) omega - j - h] ---
        torque_term = rotation_term(constants, omega, j, h)

        # --- 3. Iterate on omega_dot, starting from j_dot^(0) = 0 ---
        # omega_dot^(0) = W^-1 (T + Gamma - h_dot)
        zeta_dot = load.rate(t)
        omega_dot = rotation_rate(wobble, torque_term, torque_value, h_dot, ZERO_2)
        converged = False
        for count in range(1, max_inner_iterations + 1):

            # (a) j_dot^(k+1) = J(zeta_dot(t), omega_dot^(k))
            rate_solve = solve_instantaneous(
                state,
                constants,
                zeta_dot,
                omega_dot,
                ocean=ocean,
                sea_level_guess=warm_start["rate"],
                rtol=rtol,
            )
            if ocean:
                warm_start["rate"] = rate_solve["sea_level"].data

            # (b) omega_dot^(k+1) = W^-1 (T + Gamma - h_dot - j_dot^(k+1))
            omega_dot_new = rotation_rate(wobble, torque_term, torque_value, h_dot, rate_solve["j"])
            change = np.max(np.abs(omega_dot_new - omega_dot))
            omega_dot = omega_dot_new
            if change <= inner_rtol * np.max(np.abs(omega_dot)):
                converged = True
                break

        assert converged, f"omega_dot iteration did not converge in {max_inner_iterations} iterations"
        rhs.inner_iterations.append(count)

        # --- 4. Return omega_dot ---
        return omega_dot

    rhs.inner_iterations = []
    rhs.damping = damping
    return rhs


#=======================================================================
# Initial conditions, time integration and output
#=======================================================================

def equilibrium_rotation(
    state,
    constants,
    zeta_0,
    /,
    *,
    h_0=None,
    ocean=True,
    rtol=1e-10,
    max_iterations=200,
):
    """
    The rotation in equilibrium with a pre-existing load, (I3 - I1) omega_0 =
    J(zeta_0, omega_0) + h_0, found by the iteration
    omega_0 <- [J(zeta_0, omega_0) + h_0] / (I3 - I1).

    Args:
        state: The background EarthState.
        constants: Dict from rotation_constants.
        zeta_0: The pre-existing direct load (SHGrid).
        h_0: The pre-existing relative angular momentum, or None for zero.
        ocean: If False, there is no ocean.
        rtol: Relative tolerance on omega_0, and of each instantaneous solve.
        max_iterations: Iteration limit.

    Returns:
        Dict with omega (the equilibrium rotation), j, sea_level and iterations.
    """
    h_0 = ZERO_2 if h_0 is None else np.asarray(h_0, dtype=float)
    omega = ZERO_2.copy()
    sea_level_guess = None
    converged = False
    for count in range(1, max_iterations + 1):
        solve = solve_instantaneous(
            state,
            constants,
            zeta_0,
            omega,
            ocean=ocean,
            sea_level_guess=sea_level_guess,
            rtol=rtol,
        )
        if ocean:
            sea_level_guess = solve["sea_level"].data
        omega_new = (solve["j"] + h_0) / constants["di"]
        change = np.max(np.abs(omega_new - omega))
        omega = omega_new
        if change <= rtol * np.max(np.abs(omega)):
            converged = True
            break

    assert converged, f"equilibrium_rotation did not converge in {max_iterations} iterations"
    return {
        "omega": omega,
        "j": solve["j"],
        "sea_level": solve["sea_level"],
        "iterations": count,
    }


def rk4_step(f, t, omega, dt):
    """One classical Runge-Kutta step of omega_dot = f(t, omega)."""
    k1 = f(t, omega)
    k2 = f(t + dt / 2, omega + dt / 2 * k1)
    k3 = f(t + dt / 2, omega + dt / 2 * k2)
    k4 = f(t + dt, omega + dt * k3)
    return omega + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def trapezoid_step(f, t, omega, dt, *, rtol=1e-12, max_iterations=100):
    """
    One step of the trapezoidal rule, omega_{n+1} = omega_n + (dt / 2)(f_n + f_{n+1}),
    solved by fixed-point iteration. This contracts by about sigma'_CW dt / 2, so
    it needs steps well below the Chandler period.
    """
    f_n = f(t, omega)
    omega_next = omega + dt * f_n
    converged = False
    for count in range(max_iterations):
        omega_new = omega + dt / 2 * (f_n + f(t + dt, omega_next))
        change = np.max(np.abs(omega_new - omega_next))
        omega_next = omega_new
        if change <= rtol * np.max(np.abs(omega_next)):
            converged = True
            break
    assert converged, f"trapezoid_step did not converge in {max_iterations} iterations"
    return omega_next


def integrate(f, omega_0, t, /, *, step=rk4_step):
    """
    Integrates omega_dot = f(t, omega) over the time levels t (any spacing),
    with a one-step method step(f, t, omega, dt).

    Args:
        f: The right-hand side, e.g. from forward_rhs.
        omega_0: Initial rotation (2-vector).
        t: Array of time levels, starting at the initial time.
        step: The one-step method, e.g. rk4_step or trapezoid_step.

    Returns:
        Array of omega at the time levels, shape (len(t), 2).
    """
    omega = np.zeros((len(t), 2))
    omega[0] = omega_0
    for n in range(len(t) - 1):
        omega[n + 1] = step(f, t[n], omega[n], t[n + 1] - t[n])
    return omega


def sea_level(state, constants, load, t, omega, /, *, ocean=True, rtol=1e-10):
    """The sea level at time t: one more solve, with zeta(t) and omega(t) (SHGrid)."""
    solve = solve_instantaneous(state, constants, load.value(t), omega, ocean=ocean, rtol=rtol)
    return solve["sea_level"]


#=======================================================================
# Transform and quadrature helpers
#=======================================================================

def _expand(model, data):
    """Forward transform of grid values to a coefficient array."""
    return model.expand_field(SHGrid.from_array(data, grid=model.grid)).coeffs


def _synthesise(model, coeffs, *, as_grid=False):
    """Inverse transform of a coefficient array to grid values (or an SHGrid)."""
    clm = SHCoeffs.from_array(coeffs, normalization=model.normalization, csphase=model.csphase)
    grid = model.expand_coefficient(clm)
    return grid if as_grid else grid.data


def _integrate(model, data):
    """Surface integral of grid values."""
    return model.integrate(SHGrid.from_array(data, grid=model.grid))
