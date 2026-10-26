"""
Runs the dynamic rotation solver driven by the GRACE direct loads, and the
static (equilibrium) rotation for comparison, and saves the results.

The direct load is the land load L, optionally plus the direct ocean load D
(with or without GAD). Each coefficient is interpolated in time by a cubic spline
through the monthly values, which bridges the missing months, and zeta_dot is the
spline's derivative. The
run starts in equilibrium with the load of the first month.

Needs the cached mascon coefficients (python new-code/grace/download_grace_data.py)
and, for the ocean forcings, the direct ocean load at the same degree
(python new-code/grace/grace_direct_ocean.py --lmax <lmax>).
Run from the repository root, e.g.
    python new-code/grace/grace_rotation_run.py --forcing land-ocean --lmax 64
"""

import sys
from pathlib import Path

# solver_dynamic lives one directory up, in new-code/
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time
import numpy as np

import solver_dynamic as sd
import grace_data_helper as gh


#=======================================================================
# Parameters
#=======================================================================

parser = argparse.ArgumentParser(description="Dynamic and static rotation driven by GRACE loads.")
parser.add_argument(
    "--forcing",
    choices=gh.FORCINGS,
    default="land-ocean",
    help="direct load: L, L + D, or L + D without GAD (default land-ocean)",
)
parser.add_argument("--lmax", type=int, default=64, help="truncation degree (default 64)")
parser.add_argument("--step-days", type=float, default=10.0, help="largest RK4 step in days (default 10)")
parser.add_argument(
    "--quality-factor",
    default="100",
    help="quality factor Q of the free wobble, or 'none' for no damping (default 100)",
)
parser.add_argument("--end-year", type=float, default=None, help="stop the run at this year (default: the last month)")
parser.add_argument("--output", default=None, help="output .npz (default data/grace/runs/rotation_<forcing>_lmax<lmax>_Q<Q>.npz)")
args = parser.parse_args()

lmax = args.lmax
quality_factor = None if args.quality_factor.lower() == "none" else float(args.quality_factor)
rtol = 1e-10

if args.output is None:
    output_path = gh.run_path(args.forcing, lmax, args.quality_factor)
    output_path.parent.mkdir(parents=True, exist_ok=True)
else:
    output_path = args.output

#=======================================================================
# Earth model and the direct load
#=======================================================================

state = sd.make_state(lmax)
model = state.model
constants = sd.rotation_constants(model)

time_days, dates, direct_load_coeffs = gh.direct_load_coefficients(args.forcing, lmax)

load = gh.CoefficientLoad(
    model,
    constants,
    time_days,
    direct_load_coeffs,
)

#=======================================================================
# Time levels: the months, with each interval split into steps of at most step_days
#=======================================================================

if args.end_year is not None:
    keep = 2002.0 + time_days / 365.25 <= args.end_year
    month_days = time_days[keep]
else:
    month_days = time_days
n_months = len(month_days)

t_days = [month_days[:1]]
month_index = [0]
for n in range(n_months - 1):
    n_steps = int(np.ceil((month_days[n + 1] - month_days[n]) / args.step_days))
    t_days.append(np.linspace(month_days[n], month_days[n + 1], n_steps + 1)[1:])
    month_index.append(month_index[-1] + n_steps)
t_days = np.concatenate(t_days)
t = sd.days_to_model_time(t_days, constants)
month_t = t[month_index]

print(f"lmax = {lmax}, forcing {args.forcing}, Q = {args.quality_factor}")
print(f"{n_months} months, {len(t) - 1} steps, from {dates[0]} to {dates[n_months - 1]}")

#=======================================================================
# Static rotation: in equilibrium with the load at each month
#=======================================================================

start_time = time.time()
static_omega = np.zeros((n_months, 2))
static_j = np.zeros((n_months, 2))
static_sea_level_coeffs = np.zeros((n_months, 2, lmax + 1, lmax + 1))
for n in range(n_months):
    equilibrium = sd.equilibrium_rotation(state, constants, load.value(month_t[n]), rtol=rtol)
    static_omega[n] = equilibrium["omega"]
    static_j[n] = equilibrium["j"]
    static_sea_level_coeffs[n] = gh.coefficients_from_sea_level(model, equilibrium["sea_level"])
print(f"Static rotation took {time.time() - start_time:.1f} s")

#=======================================================================
# Dynamic rotation: equilibrium start, then RK4 month by month
#=======================================================================

f = sd.forward_rhs(
    state,
    constants,
    load,
    quality_factor=quality_factor,
    rtol=rtol,
)

start_time = time.time()
omega = np.zeros((len(t), 2))
omega[0] = static_omega[0]
for n in range(n_months - 1):
    steps = slice(month_index[n], month_index[n + 1] + 1)
    omega[steps] = sd.integrate(f, omega[month_index[n]], t[steps])
    if (n + 1) % 12 == 0 or n == n_months - 2:
        pole_mas = sd.omega_to_mas(omega[month_index[n + 1]], constants)
        print(f"  {dates[n + 1]}: pole ({pole_mas[0]:7.2f}, {pole_mas[1]:7.2f}) mas, "
              f"{time.time() - start_time:6.0f} s")
print(f"Dynamic rotation took {time.time() - start_time:.1f} s; "
      f"mean inner iterations {np.mean(f.inner_iterations):.2f}")

# --- j and sea level at the months ---
dynamic_omega = omega[month_index]
dynamic_j = np.zeros((n_months, 2))
dynamic_sea_level_coeffs = np.zeros((n_months, 2, lmax + 1, lmax + 1))
for n in range(n_months):
    solve = sd.solve_instantaneous(state, constants, load.value(month_t[n]), dynamic_omega[n], rtol=rtol)
    dynamic_j[n] = solve["j"]
    dynamic_sea_level_coeffs[n] = gh.coefficients_from_sea_level(model, solve["sea_level"])

#=======================================================================
# Save
#=======================================================================

np.savez_compressed(
    output_path,
    settings=json.dumps(vars(args)),
    lmax=lmax,
    t_days=t_days,
    omega=omega,
    pole_mas=sd.omega_to_mas(omega, constants),
    month_days=month_days,
    dates=dates[:n_months].astype(str),
    dynamic_omega=dynamic_omega,
    static_omega=static_omega,
    dynamic_j=dynamic_j,
    static_j=static_j,
    dynamic_sea_level_coeffs=dynamic_sea_level_coeffs,
    static_sea_level_coeffs=static_sea_level_coeffs,
    damping=f.damping,
    rotation_frequency=constants["omega"],
    normalization=gh.CACHE_NORMALIZATION,
    csphase=gh.CACHE_CSPHASE,
)
print(f"Saved {output_path}")

#=======================================================================
# Summary: dynamic against static pole, in mas
#=======================================================================

dynamic_mas = sd.omega_to_mas(dynamic_omega, constants)
static_mas = sd.omega_to_mas(static_omega, constants)
difference_mas = dynamic_mas - static_mas
print(f"RMS pole: dynamic {np.sqrt(np.mean(dynamic_mas**2, axis=0)).round(2)}, "
      f"static {np.sqrt(np.mean(static_mas**2, axis=0)).round(2)}, "
      f"difference {np.sqrt(np.mean(difference_mas**2, axis=0)).round(2)} mas")
