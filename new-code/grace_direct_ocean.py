"""
Separates the JPL GRACE ocean mass into the direct ocean load and the induced
ocean load, for every observed month, and saves the direct ocean load.

The observed ocean load satisfies O = D + I(L + D): D is the direct ocean load
(wind- and current-driven redistribution, plus the GAD part), L the land load, and
I the induced ocean load (the ocean's equilibrium response to the direct loads),
with the rotation held at the mean-pole drift because of GRACE's pole-tide
processing. See new-notes/grace_direct_ocean.pdf.

Needs the cached mascon coefficients (python new-code/download_grace_data.py).
Run from the repository root:
    python new-code/grace_direct_ocean.py [--lmax 128]
"""

import argparse
import time
import numpy as np
from pyshtools import SHGrid

import solver_dynamic as sd
import grace_data_helper as gh


#=======================================================================
# Parameters
#=======================================================================

parser = argparse.ArgumentParser(description="Separate the direct and induced ocean loads.")
parser.add_argument("--lmax", type=int, default=64, help="truncation degree (default 64)")
args = parser.parse_args()

lmax = args.lmax
rtol = 1e-10
tolerance = 1e-8

# Include the response to the drift of the conventional mean pole
use_mean_pole = True

# Degree of the cached mascon coefficients
cache_lmax = 90

output_path = gh.DEFAULT_DATA_DIR / f"jpl_direct_ocean_lmax{lmax}.npz"

#=======================================================================
# Earth model and data
#=======================================================================

state = sd.make_state(lmax)
model = state.model
constants = sd.rotation_constants(model)
parameters = model.parameters
mascons = gh.process_jpl_mascons(gh.DEFAULT_DATA_DIR, lmax=cache_lmax)

n_months = len(mascons["time_days"])
print(f"lmax = {lmax}; {n_months} months, {mascons['dates'][0]} to {mascons['dates'][-1]}")

# Rotation held fixed in the induced load: the mean-pole drift, or none
if use_mean_pole:
    mean_pole_omega = gh.mean_pole_rotation(mascons["time_days"], constants)
else:
    mean_pole_omega = np.zeros((n_months, 2))

#=======================================================================
# Separate the direct and induced ocean loads, month by month
#=======================================================================

shape = (n_months, 2, lmax + 1, lmax + 1)
land_coeffs = np.zeros(shape)
direct_ocean_coeffs = np.zeros(shape)
direct_ocean_nogad_coeffs = np.zeros(shape)
induced_ocean_coeffs = np.zeros(shape)
ocean_data_coeffs = np.zeros(shape)
iterations = np.zeros((n_months, 2), dtype=int)
uniform_mass = np.zeros((n_months, 2))

start_time = time.time()
for n in range(n_months):

    # --- L, O and O without GAD on the model grid ---
    land_load = gh.load_from_coefficients(model, mascons["land_coeffs"][n])
    ocean_data = gh.load_from_coefficients(model, mascons["ocean_coeffs"][n])
    ocean_data_nogad = gh.load_from_coefficients(
        model,
        mascons["ocean_coeffs"][n] - mascons["ocean_gad_coeffs"][n],
    )

    # --- O = D + I(L + D), with and without GAD ---
    separation = gh.direct_ocean_load(
        state,
        constants,
        land_load,
        ocean_data,
        mean_pole_omega[n],
        rtol=rtol,
        tolerance=tolerance,
    )
    separation_nogad = gh.direct_ocean_load(
        state,
        constants,
        land_load,
        ocean_data_nogad,
        mean_pole_omega[n],
        rtol=rtol,
        tolerance=tolerance,
    )

    land_coeffs[n] = gh.coefficients_from_load(model, land_load)
    ocean_data_coeffs[n] = gh.coefficients_from_load(model, ocean_data)
    direct_ocean_coeffs[n] = gh.coefficients_from_load(model, separation["direct_ocean_load"])
    direct_ocean_nogad_coeffs[n] = gh.coefficients_from_load(model, separation_nogad["direct_ocean_load"])
    induced_ocean_coeffs[n] = gh.coefficients_from_load(model, separation["induced_ocean_load"])
    iterations[n] = [separation["iterations"], separation_nogad["iterations"]]
    uniform_mass[n] = [separation["uniform_mass"], separation_nogad["uniform_mass"]]

print(f"Separation took {time.time() - start_time:.1f} s; mean iterations {iterations.mean(axis=0)} (with, without GAD)")

#=======================================================================
# Save
#=======================================================================

np.savez_compressed(
    output_path,
    time_days=mascons["time_days"],
    dates=mascons["dates"].astype(str),
    lmax=lmax,
    direct_ocean_coeffs=direct_ocean_coeffs,
    direct_ocean_nogad_coeffs=direct_ocean_nogad_coeffs,
    induced_ocean_coeffs=induced_ocean_coeffs,
    mean_pole_omega=mean_pole_omega,
    use_mean_pole=use_mean_pole,
    iterations=iterations,
    uniform_mass_kg=uniform_mass * parameters.density_scale * parameters.length_scale**3,
    normalization=gh.CACHE_NORMALIZATION,
    csphase=gh.CACHE_CSPHASE,
)
print(f"Saved {output_path}")

#=======================================================================
# Summary: masses, sizes over the ocean, and the degree-2, order-1 parts
#=======================================================================

years = 2002.0 + mascons["time_days"] / 365.25
radius_m = parameters.mean_sea_floor_radius * parameters.length_scale
ocean_function = state.ocean_function.data


def total_mass_gt(coeffs):
    """Total mass (Gt) of each month: the degree-0 coefficient times the surface area."""
    return coeffs[:, 0, 0, 0] * 4 * np.pi * radius_m**2 / 1e12


def ocean_rms_mm(coeffs):
    """Root-mean-square over the model ocean (mm of water = kg/m^2), averaged over the months."""
    values = []
    for month in coeffs:
        grid = gh.load_from_coefficients(model, month).data * parameters.density_scale * parameters.length_scale
        values.append(model.integrate(SHGrid.from_array(ocean_function * grid**2, grid=model.grid)) / state.ocean_area)
    return np.sqrt(np.mean(values))


trend = lambda y: np.polyfit(years, y, 1)[0]
for label, coeffs in [
    ("land L", land_coeffs),
    ("ocean data O", ocean_data_coeffs),
    ("induced I(L + D)", induced_ocean_coeffs),
    ("direct ocean D", direct_ocean_coeffs),
    ("direct ocean D, no GAD", direct_ocean_nogad_coeffs),
]:
    print(f"{label:24s}: mass trend {trend(total_mass_gt(coeffs)):8.1f} Gt/yr, RMS over the ocean {ocean_rms_mm(coeffs):6.1f} mm")
print(f"uniform layer removed (GRACE non-closure): trend {trend(uniform_mass[:, 0] * parameters.density_scale * parameters.length_scale**3 / 1e12):.1f} Gt/yr "
      f"(with GAD), {trend(uniform_mass[:, 1] * parameters.density_scale * parameters.length_scale**3 / 1e12):.1f} Gt/yr (without)")

# Degree-2, order-1 coefficients drive the rotation: compare D with L
for label, coeffs in [("land L", land_coeffs), ("direct ocean D", direct_ocean_coeffs), ("direct ocean D, no GAD", direct_ocean_nogad_coeffs)]:
    c21 = np.stack([coeffs[:, 0, 2, 1], coeffs[:, 1, 2, 1]], axis=1)
    fitted = np.stack([np.polyval(np.polyfit(years, c21[:, i], 1), years) for i in range(2)], axis=1)
    detrended = c21 - fitted
    print(f"{label:24s}: (2,1) trend {[f'{trend(c21[:, i]):+.3f}' for i in range(2)]} kg/m^2/yr, "
          f"detrended RMS {np.sqrt(np.mean(detrended**2, axis=0)).round(3)} kg/m^2")
