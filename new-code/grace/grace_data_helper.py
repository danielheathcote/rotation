"""
Download, read and project the JPL GRACE/GRACE-FO mascon solutions.

Product: JPL RL06.3Mv04 mascons with the Coastline Resolution Improvement (CRI)
filter, monthly from April 2002, on a 0.5 degree grid (PO.DAAC short name
TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4). C20 and C30 come from satellite
laser ranging, degree 1 is estimated, and glacial isostatic adjustment (ICE6G-D)
has been removed. Values are equivalent water heights (cm), as anomalies relative
to the 2004.0-2009.999 mean. The monthly ocean-atmosphere dealiasing model (GAD)
has been added back over the ocean only; the file also gives it separately, along
with the land mask. Some months are missing (listed in the file's metadata).

Downloading needs a NASA Earthdata login in ~/.netrc:
    machine urs.earthdata.nasa.gov login <username> password <password>

The monthly fields are split into land and ocean parts with JPL's land mask and
projected onto spherical harmonics once, together with the GAD part of the ocean.
The cached coefficients are in kg/m^2, 4pi-normalised without the Condon-Shortley
phase, and are converted to a pyslfp load at any truncation degree by
load_from_coefficients.

The ocean values contain the induced ocean load (the ocean's equilibrium response
to the direct loads) as well as the direct ocean load. direct_ocean_load separates
the two; see new-notes/grace_direct_ocean.pdf.
"""

from pathlib import Path

import numpy as np
from pyshtools import SHCoeffs, SHGrid
from scipy.interpolate import CubicSpline, RegularGridInterpolator

import solver_dynamic as sd


JPL_MASCON_SHORT_NAME = "TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4"

# Default data directory: <repository>/data/grace
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "grace"

# 1 cm of water is 10 kg/m^2
KG_PER_M2_PER_CM = 10.0

# Normalisation of the cached coefficients
CACHE_NORMALIZATION = "4pi"
CACHE_CSPHASE = 1

# IERS (2018) secular pole, used as the mean pole in the GRACE pole-tide corrections:
# x = a + b (t - 2000), y = a + b (t - 2000), in mas with t in years
IERS_SECULAR_POLE_X = (55.0, 1.677)
IERS_SECULAR_POLE_Y = (320.5, 3.460)

# The GRACE anomalies are relative to the 2004.0-2009.999 mean
BASELINE_YEARS = (2004.0, 2010.0)

MAS_TO_RADIANS = np.pi / (180.0 * 3600.0e3)


#=======================================================================
# Download
#=======================================================================

def download_jpl_mascons(data_dir=DEFAULT_DATA_DIR):
    """
    Downloads the JPL CRI mascon file with earthaccess, using the Earthdata login
    in ~/.netrc.

    Args:
        data_dir: Directory for the downloaded files.

    Returns:
        List of paths of the downloaded files.
    """
    import earthaccess

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    auth = earthaccess.login(strategy="netrc")
    assert auth.authenticated, "Earthdata login failed: check the urs.earthdata.nasa.gov entry in ~/.netrc"

    granules = earthaccess.search_data(short_name=JPL_MASCON_SHORT_NAME)
    assert len(granules) > 0, f"No granules found for {JPL_MASCON_SHORT_NAME}"
    paths = earthaccess.download(granules, local_path=str(data_dir))
    return [Path(path) for path in paths]


#=======================================================================
# Read
#=======================================================================

def find_mascon_file(data_dir=DEFAULT_DATA_DIR):
    """
    Locates the mascon data file in data_dir.

    Args:
        data_dir: Directory holding the downloaded file.

    Returns:
        Path of the data file.
    """
    data_dir = Path(data_dir)
    data_files = sorted(data_dir.glob("GRCTellus.JPL.*MSCNv04CRI*.nc"))
    assert len(data_files) == 1, f"Expected one mascon data file in {data_dir}, found {[p.name for p in data_files]}"
    return data_files[0]


def read_jpl_mascons(data_dir=DEFAULT_DATA_DIR):
    """
    Reads the monthly mascon fields, the GAD fields and the land mask.

    Args:
        data_dir: Directory holding the downloaded file.

    Returns:
        Dict with time_days (mid-month times, days since 2002-01-01), dates
        (numpy datetime64), lat and lon (cell centres, degrees), lwe_cm (equivalent
        water height in cm, shape (n_months, n_lat, n_lon)), gad_cm (the GAD part,
        same shape), land_mask (1 on land, 0 on the ocean) and months_missing.
    """
    import xarray as xr

    with xr.open_dataset(find_mascon_file(data_dir)) as data:
        lat = data["lat"].values
        lon = data["lon"].values
        dates = data["time"].values
        lwe_cm = data["lwe_thickness"].values
        gad_cm = data["GAD"].values
        land_mask = data["land_mask"].values
        months_missing = data.attrs.get("months_missing", "")
        assert data["lwe_thickness"].attrs.get("units") == "cm", "Expected lwe_thickness in cm"
        assert data["GAD"].attrs.get("units") == "cm", "Expected GAD in cm"

    assert lwe_cm.shape[1:] == (lat.size, lon.size), f"Unexpected data shape {lwe_cm.shape}"
    assert np.all(np.isfinite(lwe_cm)), "lwe_thickness contains missing values"

    time_days = (dates - np.datetime64("2002-01-01")) / np.timedelta64(1, "D")
    return {
        "time_days": time_days,
        "dates": dates,
        "lat": lat,
        "lon": lon,
        "lwe_cm": lwe_cm,
        "gad_cm": gad_cm,
        "land_mask": land_mask,
        "months_missing": months_missing.split(";") if months_missing else [],
    }


#=======================================================================
# Projection onto spherical harmonics
#=======================================================================

def grid_to_coefficients(lat, lon, field, lmax):
    """
    Spherical harmonic coefficients of a field given on a cell-centred
    latitude-longitude grid. The field is interpolated onto the Driscoll-Healy
    grid of the same spacing, which pyshtools expands exactly.

    Args:
        lat: Cell-centre latitudes in degrees, ascending or descending.
        lon: Cell-centre longitudes in degrees, equally spaced over 360 degrees.
        field: Values, shape (lat.size, lon.size).
        lmax: Truncation degree of the returned coefficients.

    Returns:
        Coefficient array, shape (2, lmax + 1, lmax + 1), in CACHE_NORMALIZATION.
    """
    order = np.argsort(lat)
    lat = np.asarray(lat)[order]
    field = np.asarray(field)[order]
    spacing = lon[1] - lon[0]
    assert np.isclose(lon.size * spacing, 360.0), "Longitudes must cover the sphere at equal spacing"

    # Pad periodically in longitude, and with the polar-row means at the poles
    lon_padded = np.concatenate([[lon[-1] - 360.0], lon, [lon[0] + 360.0]])
    field_padded = np.concatenate([field[:, -1:], field, field[:, :1]], axis=1)
    lat_padded = np.concatenate([[-90.0], lat, [90.0]])
    south = np.full((1, lon_padded.size), field[0].mean())
    north = np.full((1, lon_padded.size), field[-1].mean())
    field_padded = np.concatenate([south, field_padded, north], axis=0)
    interpolator = RegularGridInterpolator((lat_padded, lon_padded), field_padded)

    # Driscoll-Healy grid (sampling 2) with the same spacing: rows from 90 to -90 + spacing
    n = int(round(180.0 / spacing))
    lat_dh = 90.0 - spacing * np.arange(n)
    lon_dh = spacing * np.arange(2 * n)
    lat_mesh, lon_mesh = np.meshgrid(lat_dh, lon_dh, indexing="ij")
    field_dh = interpolator((lat_mesh, lon_mesh % 360.0))

    grid = SHGrid.from_array(field_dh, grid="DH")
    clm = grid.expand(normalization=CACHE_NORMALIZATION, csphase=CACHE_CSPHASE, lmax_calc=lmax)
    return clm.coeffs


def process_jpl_mascons(data_dir=DEFAULT_DATA_DIR, /, *, lmax=90, cache_name=None):
    """
    Splits each month into land and ocean parts with the land mask, converts to
    kg/m^2 and projects both onto spherical harmonics, together with the GAD part
    of the ocean (which is included in the ocean part). The result is cached in
    data_dir and reused on later calls.

    Args:
        data_dir: Directory holding the downloaded files.
        lmax: Truncation degree of the cached coefficients.
        cache_name: File name of the cache (default jpl_mascons_lmax<lmax>.npz).

    Returns:
        Dict with time_days, dates, land_coeffs, ocean_coeffs and ocean_gad_coeffs
        (each of shape (n_months, 2, lmax + 1, lmax + 1), kg/m^2), months_missing,
        normalization and csphase.
    """
    data_dir = Path(data_dir)
    cache_path = data_dir / (cache_name or f"jpl_mascons_lmax{lmax}.npz")
    if cache_path.exists():
        return load_mascon_coefficients(cache_path)

    mascons = read_jpl_mascons(data_dir)
    land_mask = mascons["land_mask"]
    n_months = mascons["lwe_cm"].shape[0]
    ocean_mask = 1 - land_mask
    land_coeffs = np.zeros((n_months, 2, lmax + 1, lmax + 1))
    ocean_coeffs = np.zeros((n_months, 2, lmax + 1, lmax + 1))
    ocean_gad_coeffs = np.zeros((n_months, 2, lmax + 1, lmax + 1))

    for n in range(n_months):
        mass_kg_m2 = KG_PER_M2_PER_CM * mascons["lwe_cm"][n]
        gad_kg_m2 = KG_PER_M2_PER_CM * np.nan_to_num(mascons["gad_cm"][n])
        land_coeffs[n] = grid_to_coefficients(mascons["lat"], mascons["lon"], land_mask * mass_kg_m2, lmax)
        ocean_coeffs[n] = grid_to_coefficients(mascons["lat"], mascons["lon"], ocean_mask * mass_kg_m2, lmax)
        ocean_gad_coeffs[n] = grid_to_coefficients(mascons["lat"], mascons["lon"], ocean_mask * gad_kg_m2, lmax)

    np.savez_compressed(
        cache_path,
        time_days=mascons["time_days"],
        dates=mascons["dates"].astype("datetime64[D]").astype(str),
        land_coeffs=land_coeffs,
        ocean_coeffs=ocean_coeffs,
        ocean_gad_coeffs=ocean_gad_coeffs,
        months_missing=np.array(mascons["months_missing"]),
        normalization=CACHE_NORMALIZATION,
        csphase=CACHE_CSPHASE,
    )
    return load_mascon_coefficients(cache_path)


def load_mascon_coefficients(cache_path):
    """
    Reads cached mascon coefficients written by process_jpl_mascons.

    Args:
        cache_path: Path of the .npz cache.

    Returns:
        Dict as returned by process_jpl_mascons.
    """
    with np.load(cache_path) as cache:
        return {
            "time_days": cache["time_days"],
            "dates": cache["dates"].astype("datetime64[D]"),
            "land_coeffs": cache["land_coeffs"],
            "ocean_coeffs": cache["ocean_coeffs"],
            "ocean_gad_coeffs": cache["ocean_gad_coeffs"],
            "months_missing": list(cache["months_missing"]),
            "normalization": str(cache["normalization"]),
            "csphase": int(cache["csphase"]),
        }


#=======================================================================
# Conversion between cached coefficients and pyslfp loads
#=======================================================================

def load_from_coefficients(model, coeffs, /, *, normalization=CACHE_NORMALIZATION, csphase=CACHE_CSPHASE):
    """
    A pyslfp load grid from cached coefficients (kg/m^2), converted to
    non-dimensional units. The coefficients are truncated to the model's degree,
    or padded with zeros if the model's degree is higher.

    Args:
        model: The pyslfp EarthModel.
        coeffs: Coefficient array, shape (2, L + 1, L + 1).
        normalization, csphase: Conventions of coeffs.

    Returns:
        The load as an SHGrid on the model grid.
    """
    # Mass per area in kg/m^2 to pyslfp units (density scale x length scale)
    parameters = model.parameters
    scaled = coeffs / (parameters.density_scale * parameters.length_scale)
    return field_from_coefficients(model, scaled, normalization=normalization, csphase=csphase)


def field_from_coefficients(model, coeffs, /, *, normalization=CACHE_NORMALIZATION, csphase=CACHE_CSPHASE):
    """
    A field on the model grid from coefficients in the cache conventions, keeping
    their units (e.g. kg/m^2 or metres). Truncated or padded with zeros to the
    model's degree.

    Args:
        model: The pyslfp EarthModel.
        coeffs: Coefficient array, shape (2, L + 1, L + 1).
        normalization, csphase: Conventions of coeffs.

    Returns:
        The field as an SHGrid on the model grid.
    """
    padded = np.zeros((2, model.lmax + 1, model.lmax + 1))
    lmax_common = min(model.lmax, coeffs.shape[1] - 1)
    padded[:, : lmax_common + 1, : lmax_common + 1] = coeffs[:, : lmax_common + 1, : lmax_common + 1]
    clm = SHCoeffs.from_array(padded, normalization=normalization, csphase=csphase)
    clm = clm.convert(normalization=model.normalization, csphase=model.csphase)
    return model.expand_coefficient(clm)


def coefficients_from_load(model, load):
    """
    Coefficients (kg/m^2, cache conventions) of a pyslfp load grid: the inverse
    of load_from_coefficients.

    Args:
        model: The pyslfp EarthModel.
        load: The load as an SHGrid on the model grid (non-dimensional).

    Returns:
        Coefficient array, shape (2, model.lmax + 1, model.lmax + 1).
    """
    clm = model.expand_field(load)
    clm = clm.convert(normalization=CACHE_NORMALIZATION, csphase=CACHE_CSPHASE)
    parameters = model.parameters
    return clm.coeffs * parameters.density_scale * parameters.length_scale


#=======================================================================
# The direct ocean load
#=======================================================================

def mean_pole_rotation(time_days, constants):
    """
    The rotation corresponding to the drift of the conventional mean pole, relative
    to its mean over the baseline. The GRACE pole-tide corrections use the observed
    pole minus the mean pole, so the ocean data keep the response to this rotation.

    Args:
        time_days: Times in days since 2002-01-01.
        constants: Dict from solver_dynamic.rotation_constants.

    Returns:
        Equatorial rotations, shape (n, 2), in pyslfp units, with
        omega / Omega = (x - x_ref, -(y - y_ref)) (IERS y is positive towards 90 W).
    """
    years = 2002.0 + np.asarray(time_days) / 365.25
    reference_year = 0.5 * (BASELINE_YEARS[0] + BASELINE_YEARS[1])   # the mean of a linear drift
    x_mas = IERS_SECULAR_POLE_X[1] * (years - reference_year)
    y_mas = IERS_SECULAR_POLE_Y[1] * (years - reference_year)
    return constants["omega"] * MAS_TO_RADIANS * np.stack([x_mas, -y_mas], axis=-1)


def induced_ocean_load(state, constants, direct_load, omega, /, *, sea_level_guess=None, rtol=1e-10):
    """
    The induced ocean load rho_w C xi: the ocean's equilibrium response (eustatic
    rise plus self-attraction and loading) to a direct load, with the rotation
    held at omega.

    Args:
        state: The background EarthState.
        constants: Dict from solver_dynamic.rotation_constants.
        direct_load: The direct load (SHGrid, model units).
        omega: The rotation held fixed (2-vector).
        sea_level_guess: Grid values of xi from an earlier solve (warm start).
        rtol: Tolerance of the sea-level solve.

    Returns:
        Tuple (induced load as an SHGrid, grid values of xi for the next warm start).
    """
    solve = sd.solve_instantaneous(
        state,
        constants,
        direct_load,
        omega,
        sea_level_guess=sea_level_guess,
        rtol=rtol,
    )
    sea_level = solve["sea_level"].data
    induced = constants["water_density"] * state.ocean_function.data * sea_level
    return SHGrid.from_array(induced, grid=state.model.grid), sea_level


def direct_ocean_load(
    state,
    constants,
    land_load,
    ocean_data,
    omega,
    /,
    *,
    rtol=1e-10,
    tolerance=1e-8,
    max_iterations=50,
):
    """
    Separates the observed ocean load O into the direct ocean load D and the
    induced ocean load, O = D + I(L + D), where L is the land load and I the
    induced ocean load with the rotation held at omega. Iterates
    D <- O - I(L + D), removing D's uniform ocean-mean layer each time: GRACE's
    mass budget does not quite close, and that excess can only be a uniform
    ocean layer, which the sea-level equation cancels anyway.

    Args:
        state: The background EarthState.
        constants: Dict from solver_dynamic.rotation_constants.
        land_load: The land load L (SHGrid, model units).
        ocean_data: The observed ocean load O (SHGrid, model units).
        omega: The rotation held fixed (2-vector), e.g. from mean_pole_rotation.
        rtol: Tolerance of each sea-level solve.
        tolerance: Relative tolerance on the change in D.
        max_iterations: Iteration limit.

    Returns:
        Dict with direct_ocean_load (D) and induced_ocean_load (I(L + D)) as
        SHGrids, the mass removed as a uniform layer (model units), and iterations.
    """
    model = state.model
    ocean_function = state.ocean_function.data
    grid = model.grid

    direct = np.zeros_like(ocean_data.data)
    sea_level_guess = None
    converged = False
    for count in range(1, max_iterations + 1):

        # I(L + D), then D = O - I(L + D)
        total = SHGrid.from_array(land_load.data + direct, grid=grid)
        induced, sea_level_guess = induced_ocean_load(
            state,
            constants,
            total,
            omega,
            sea_level_guess=sea_level_guess,
            rtol=rtol,
        )
        direct_new = ocean_data.data - induced.data

        # Remove the uniform ocean layer, so that D carries no net mass
        uniform_mass = model.integrate(SHGrid.from_array(direct_new, grid=grid))
        direct_new -= uniform_mass / state.ocean_area * ocean_function

        change = np.max(np.abs(direct_new - direct))
        direct = direct_new
        if change <= tolerance * np.max(np.abs(direct)):
            converged = True
            break

    assert converged, f"direct_ocean_load did not converge in {max_iterations} iterations"
    return {
        "direct_ocean_load": SHGrid.from_array(direct, grid=grid),
        "induced_ocean_load": induced,
        "uniform_mass": uniform_mass,
        "iterations": count,
    }


def load_direct_ocean_coefficients(cache_path):
    """
    Reads the direct ocean load written by grace_direct_ocean.py.

    Args:
        cache_path: Path of the .npz file.

    Returns:
        Dict of its arrays (coefficients in kg/m^2, cache conventions).
    """
    with np.load(cache_path) as cache:
        return {key: cache[key] for key in cache.files}


def coefficients_from_sea_level(model, sea_level):
    """
    Coefficients (metres, cache conventions) of a pyslfp sea-level grid.

    Args:
        model: The pyslfp EarthModel.
        sea_level: The sea level as an SHGrid on the model grid (non-dimensional).

    Returns:
        Coefficient array, shape (2, model.lmax + 1, model.lmax + 1).
    """
    clm = model.expand_field(sea_level)
    clm = clm.convert(normalization=CACHE_NORMALIZATION, csphase=CACHE_CSPHASE)
    return clm.coeffs * model.parameters.length_scale


#=======================================================================
# The direct load of each forcing, and the saved runs
#=======================================================================

FORCINGS = ("land", "land-ocean", "land-ocean-nogad")


def direct_load_coefficients(forcing, lmax, /, *, data_dir=DEFAULT_DATA_DIR):
    """
    The monthly direct load of a forcing: the land load L, plus the direct ocean
    load D for the ocean forcings (with or without GAD).

    Args:
        forcing: "land", "land-ocean" or "land-ocean-nogad".
        lmax: Truncation degree. The ocean forcings need the direct ocean load
            saved at this degree by grace_direct_ocean.py.
        data_dir: Directory of the cached coefficients.

    Returns:
        Tuple (time_days, dates, coefficients in kg/m^2 of shape (n_months, 2, lmax + 1, lmax + 1)).
    """
    assert forcing in FORCINGS, f"Unknown forcing {forcing!r}; choose from {FORCINGS}"
    mascons = load_mascon_coefficients(data_dir / "jpl_mascons_lmax90.npz")
    time_days = mascons["time_days"]
    assert mascons["land_coeffs"].shape[2] > lmax, (
        f"The mascon cache only reaches degree {mascons['land_coeffs'].shape[2] - 1}; "
        f"rerun download_grace_data.py with a higher lmax"
    )
    coeffs = mascons["land_coeffs"][:, :, : lmax + 1, : lmax + 1]

    # --- Add the direct ocean load ---
    if forcing != "land":
        direct_ocean = load_direct_ocean_coefficients(data_dir / f"jpl_direct_ocean_lmax{lmax}.npz")
        assert np.allclose(direct_ocean["time_days"], time_days), (
            "The direct ocean load and the mascons have different months"
        )
        key = "direct_ocean_coeffs" if forcing == "land-ocean" else "direct_ocean_nogad_coeffs"
        coeffs = coeffs + direct_ocean[key]
    return time_days, mascons["dates"], coeffs


def run_path(forcing, lmax, quality_factor, /, *, data_dir=DEFAULT_DATA_DIR):
    """
    Path of a run saved by grace_rotation_run.py.

    Args:
        forcing: "land", "land-ocean" or "land-ocean-nogad".
        lmax: Truncation degree.
        quality_factor: Q as given on the command line (e.g. "100" or "none").
        data_dir: Directory of the cached coefficients.

    Returns:
        The Path of the .npz file.
    """
    return data_dir / "runs" / f"rotation_{forcing}_lmax{lmax}_Q{quality_factor}.npz"


def load_run(forcing, lmax, quality_factor, /, *, data_dir=DEFAULT_DATA_DIR):
    """
    Reads a run saved by grace_rotation_run.py.

    Args:
        forcing, lmax, quality_factor, data_dir: As for run_path.

    Returns:
        Dict of its arrays, with dates as numpy datetime64.
    """
    with np.load(run_path(forcing, lmax, quality_factor, data_dir=data_dir)) as run:
        result = {key: run[key] for key in run.files}
    result["dates"] = result["dates"].astype("datetime64[D]")
    return result


#=======================================================================
# A time-varying load from the monthly coefficients
#=======================================================================

class CoefficientLoad(sd.Load):
    """
    A direct load zeta(x, t) from monthly coefficients (kg/m^2), for the dynamic
    solver. Each coefficient is interpolated by a natural cubic spline through the
    monthly values, which also bridges the missing months; zeta_dot is the spline's
    derivative.
    """

    def __init__(self, model, constants, time_days, coeffs, /):
        """
        Args:
            model: The pyslfp EarthModel.
            constants: Dict from solver_dynamic.rotation_constants.
            time_days: Times of the months (days since 2002-01-01).
            coeffs: Coefficients, shape (n_months, 2, L + 1, L + 1).
        """
        assert coeffs.shape[0] == len(time_days), (
            f"{coeffs.shape[0]} coefficient sets for {len(time_days)} times"
        )
        self.model = model
        self.constants = constants
        self.coeffs_shape = coeffs.shape[1:]
        self.spline = CubicSpline(
            time_days,
            coeffs.reshape(len(time_days), -1),
            axis=0,
            bc_type="natural",
        )
        self.spline_rate = self.spline.derivative()

        # Days per unit of model time, to turn d/d(days) into d/dt
        self.days_per_time_unit = self.constants["time_scale"] / 86400.0

    def coefficients(self, t):
        """The interpolated coefficients (kg/m^2) at model time t."""
        days = sd.model_time_to_days(t, self.constants)
        return self.spline(days).reshape(self.coeffs_shape)

    def value(self, t):
        return load_from_coefficients(self.model, self.coefficients(t))

    def rate(self, t):
        days = sd.model_time_to_days(t, self.constants)
        coeffs_rate = self.spline_rate(days).reshape(self.coeffs_shape) * self.days_per_time_unit
        return load_from_coefficients(self.model, coeffs_rate)
