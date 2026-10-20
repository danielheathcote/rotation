"""
Analysis utilities for examples_dynamic.ipynb: the free-wobble period of a run,
and the least-squares separation of a forced response from the free wobble.
"""

import numpy as np


def wobble_period_days(t_days, omega):
    """
    Mean period of a free wobble from the unwrapped polar angle of omega,
    using whole turns only.

    Args:
        t_days: Times in days.
        omega: Equatorial rotation, shape (len(t_days), 2).

    Returns:
        The period in days.
    """
    angle = np.unwrap(np.arctan2(omega[:, 1], omega[:, 0]))
    turns = int(np.floor((angle[-1] - angle[0]) / (2 * np.pi)))
    assert turns >= 1, f"Run is shorter than one wobble period ({turns} turns)"
    t_end = np.interp(angle[0] + 2 * np.pi * turns, angle, t_days)
    return (t_end - t_days[0]) / turns


def fit_forced_response(t_days, omega, nu_per_day, sigma_per_day):
    """
    Least-squares fit of m = omega_1 + i omega_2 by
    c_1 e^{i nu t} + c_2 e^{-i nu t} + c_3 e^{i sigma t} + c_4 e^{-i sigma* t},
    separating the forced (nu) and free (sigma) parts. The negative-frequency
    terms allow for elliptical motion. A damped free wobble has a complex
    sigma, whose positive imaginary part makes both free terms decay.

    Args:
        t_days: Times in days.
        omega: Equatorial rotation, shape (len(t_days), 2).
        nu_per_day: Forcing frequency (rad/day), positive.
        sigma_per_day: Free-wobble frequency (rad/day), real or complex.

    Returns:
        The four complex coefficients (c_1, c_2, c_3, c_4).
    """
    m = omega[:, 0] + 1j * omega[:, 1]
    basis = np.stack(
        [
            np.exp(1j * nu_per_day * t_days),
            np.exp(-1j * nu_per_day * t_days),
            np.exp(1j * sigma_per_day * t_days),
            np.exp(-1j * np.conj(sigma_per_day) * t_days),
        ],
        axis=1,
    )
    coefficients, *_ = np.linalg.lstsq(basis, m, rcond=None)
    return coefficients


def coastline_segments(state):
    """
    Coastlines of the model's ocean function, as the 0.5 contour of C.

    Args:
        state: The pyslfp EarthState.

    Returns:
        List of (n, 2) arrays of (longitude, latitude) in degrees.
    """
    import contourpy

    generator = contourpy.contour_generator(state.lons(), state.lats(), state.ocean_function.data)
    return generator.lines(0.5)


def lonlat_to_xyz(lon, lat, radius=1.0):
    """
    Cartesian coordinates in the Earth frame (x towards 0 E, y towards 90 E, z north).

    Args:
        lon, lat: Longitude and latitude in degrees.
        radius: Radius of the sphere.

    Returns:
        Tuple of arrays (x, y, z).
    """
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    return (
        radius * np.cos(lat_rad) * np.cos(lon_rad),
        radius * np.cos(lat_rad) * np.sin(lon_rad),
        radius * np.sin(lat_rad),
    )


def rotation_axis_direction(omega, omega_reference, exaggeration):
    """
    Unit vectors along the rotation axis Omega e_3 + omega, in the Earth frame,
    with the tilt from e_3 multiplied by an exaggeration factor.

    Args:
        omega: Equatorial rotation perturbations, shape (n, 2).
        omega_reference: The reference rotation rate Omega.
        exaggeration: Factor applied to the tilt angle |omega| / Omega.

    Returns:
        Array of unit vectors, shape (n, 3).
    """
    omega = np.atleast_2d(omega)
    tilt = exaggeration * np.linalg.norm(omega, axis=1) / omega_reference
    azimuth = np.arctan2(omega[:, 1], omega[:, 0])
    return np.stack(
        [np.sin(tilt) * np.cos(azimuth), np.sin(tilt) * np.sin(azimuth), np.cos(tilt)],
        axis=1,
    )


def remove_degrees(model, field, degrees):
    """
    Removes whole spherical-harmonic degrees from a field: expand, set every
    coefficient of those degrees (all orders m) to zero, synthesise.

    Args:
        model: The pyslfp EarthModel.
        field: The field (SHGrid), defined everywhere on the sphere.
        degrees: The degrees l to remove, e.g. [2].

    Returns:
        The filtered field (SHGrid).
    """
    from pyshtools import SHCoeffs

    coeffs = model.expand_field(field).coeffs.copy()
    coeffs[:, list(degrees), :] = 0.0
    clm = SHCoeffs.from_array(coeffs, normalization=model.normalization, csphase=model.csphase)
    return model.expand_coefficient(clm)


def sea_level_animation(state, sea_levels_mm, titles, pole_mas, t_days, *, frames_per_second):
    """
    Animation of one or more sea-level maps (Robinson projection, model
    coastlines) beside the pole position, all advancing together.

    Args:
        state: The pyslfp EarthState.
        sea_levels_mm: List of sea-level arrays (mm), each of shape
            (n_frames, n_lat, n_lon), NaN over land. Each map has its own
            colour scale.
        titles: List of map titles, one per array; the time is appended.
        pole_mas: Pole position at each frame (mas), shape (n_frames, 2).
        t_days: Time of each frame (days).
        frames_per_second: Playback rate.

    Returns:
        The matplotlib FuncAnimation.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from matplotlib.animation import FuncAnimation

    assert len(sea_levels_mm) == len(titles), (
        f"{len(sea_levels_mm)} sea-level arrays but {len(titles)} titles"
    )
    lons, lats = state.lons(), state.lats()
    n_maps = len(sea_levels_mm)

    fig = plt.figure(figsize=(6.5 * n_maps + 5, 4.6))
    grid = fig.add_gridspec(1, n_maps + 1, width_ratios=[1.0] * n_maps + [0.65], wspace=0.3)

    # --- Sea-level maps ---
    meshes = []
    map_titles = []
    for i, sea_level_mm in enumerate(sea_levels_mm):
        ax_map = fig.add_subplot(grid[i], projection=ccrs.Robinson())
        limit = np.nanmax(np.abs(sea_level_mm))
        mesh = ax_map.pcolormesh(
            lons,
            lats,
            sea_level_mm[0],
            transform=ccrs.PlateCarree(),
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            shading="auto",
        )
        ax_map.contour(lons, lats, state.ocean_function.data, levels=[0.5], colors="k", linewidths=0.5,
                       transform=ccrs.PlateCarree())
        ax_map.set_global()
        fig.colorbar(mesh, ax=ax_map, orientation="horizontal", pad=0.05, fraction=0.05, label=r"$\xi$ (mm)")
        meshes.append(mesh)
        map_titles.append(ax_map.set_title(""))

    # --- Pole position ---
    ax_pole = fig.add_subplot(grid[n_maps])
    ax_pole.plot(pole_mas[:, 0], pole_mas[:, 1], color="grey", lw=0.8)
    marker, = ax_pole.plot([], [], "ko")
    arrow = ax_pole.annotate("", xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="k"))
    ax_pole.set_xlabel(r"$\omega_1/\Omega$ (mas)")
    ax_pole.set_ylabel(r"$\omega_2/\Omega$ (mas)")
    ax_pole.set_aspect("equal")
    ax_pole.grid(True, alpha=0.5)
    ax_pole.set_title("Pole position")

    def update(n):
        for mesh, map_title, sea_level_mm, title in zip(meshes, map_titles, sea_levels_mm, titles):
            mesh.set_array(sea_level_mm[n].ravel())
            map_title.set_text(f"{title}, t = {t_days[n]:.0f} d")
        marker.set_data([pole_mas[n, 0]], [pole_mas[n, 1]])
        arrow.xy = (pole_mas[n, 0], pole_mas[n, 1])
        return (*meshes, *map_titles, marker, arrow)

    animation = FuncAnimation(fig, update, frames=len(t_days), interval=1000 / frames_per_second, blit=False)
    plt.close(fig)
    return animation


def wobble_quality_factor(t_days, omega, period_days):
    """
    Quality factor Q of a decaying free wobble, from a least-squares fit of
    log|omega| against time: |omega| ~ exp(-t / tau), so Q = (2 pi / T) tau / 2.

    Args:
        t_days: Times in days.
        omega: Equatorial rotation, shape (len(t_days), 2).
        period_days: The wobble period T in days.

    Returns:
        Q (infinite if the amplitude does not decay).
    """
    slope = np.polyfit(t_days, np.log(np.linalg.norm(omega, axis=1)), 1)[0]
    if slope >= 0:
        return np.inf
    return (2 * np.pi / period_days) / (2 * abs(slope))
