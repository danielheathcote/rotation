"""
Plotting utilities for the GRACE rotation runs: fields on the model grid, gaps in
the record, a complex amplitude spectrum of the pole, and the map animations
used by the animation notebooks.
"""

import numpy as np

import grace_data_helper as gh


def years_from_days(time_days):
    """Decimal years from days since 2002-01-01."""
    return 2002.0 + np.asarray(time_days) / 365.25


def gap_spans(time_days, *, min_gap_days=45.0):
    """
    The gaps in the monthly record, for shading on time series.

    Args:
        time_days: Times of the observed months (days since 2002-01-01).
        min_gap_days: Spacing above which two months count as a gap.

    Returns:
        List of (start, end) in decimal years.
    """
    years = years_from_days(time_days)
    spacing = np.diff(time_days)
    return [(years[i], years[i + 1]) for i in np.where(spacing > min_gap_days)[0]]


def fields_from_coefficients(model, coeffs, *, scale=1.0, mask=None):
    """
    Grid values of a series of coefficient arrays, in the units of the coefficients.

    Args:
        model: The pyslfp EarthModel.
        coeffs: Coefficients, shape (n, 2, L + 1, L + 1), cache conventions.
        scale: Factor applied to the values (e.g. 1e3 for metres to mm).
        mask: Grid of 1 and NaN applied to every field (e.g. the ocean), or None.

    Returns:
        Array of shape (n, n_lat, n_lon).
    """
    fields = np.array([gh.field_from_coefficients(model, c).data for c in coeffs]) * scale
    if mask is not None:
        fields = fields * mask
    return fields


def ocean_mask(state):
    """1 over the model ocean and NaN over land, for masking sea-level maps."""
    return np.where(state.ocean_function.data > 0, 1.0, np.nan)


def trend_map(years, fields):
    """
    Least-squares linear trend at every grid point.

    Args:
        years: Times in years, shape (n,).
        fields: Grid values, shape (n, n_lat, n_lon) (NaN allowed).

    Returns:
        The trend per year, shape (n_lat, n_lon).
    """
    flat = fields.reshape(len(years), -1)
    slope = np.polyfit(years, np.nan_to_num(flat), 1)[0]
    slope[np.isnan(flat[0])] = np.nan
    return slope.reshape(fields.shape[1:])


def detrend(years, values):
    """Removes a least-squares straight line along the first axis."""
    flat = values.reshape(len(years), -1)
    coefficients = np.polyfit(years, np.nan_to_num(flat), 1)
    fitted = np.outer(years, coefficients[0]) + coefficients[1]
    return (flat - fitted).reshape(values.shape)


def robust_limit(values, *, percentile=99.0):
    """A symmetric colour limit that ignores the most extreme values."""
    return np.nanpercentile(np.abs(values), percentile)


def complex_amplitude_spectrum(years, pole_mas, frequencies_cpy):
    """
    Amplitude spectrum of the complex pole m = m_1 + i m_2 from unevenly spaced
    samples, after removing a straight line: |sum_n m_n e^{-2 pi i f t_n}| / N.
    Positive frequencies are prograde (anticlockwise, like the Chandler wobble),
    negative ones retrograde. The gaps leak some power into side lobes.

    Args:
        years: Sample times in years.
        pole_mas: Pole position, shape (n, 2).
        frequencies_cpy: Frequencies in cycles per year.

    Returns:
        The amplitude (mas) at each frequency.
    """
    m = detrend(years, pole_mas)
    m = m[:, 0] + 1j * m[:, 1]
    phase = np.exp(-2j * np.pi * np.outer(frequencies_cpy, years))
    return np.abs(phase @ m) / len(years)


def load_norm(load_mm, *, linear_mm):
    """
    A symmetric colour scale for the load: linear within +-linear_mm, logarithmic
    beyond, up to the largest value. This shows the ocean and land hydrology
    (tens of mm) and the ice sheets (metres) on one map.

    Args:
        load_mm: Load values (mm water equivalent), any shape.
        linear_mm: Half-width of the linear part.

    Returns:
        A matplotlib SymLogNorm.
    """
    from matplotlib.colors import SymLogNorm

    limit = np.nanmax(np.abs(load_mm))
    return SymLogNorm(linthresh=linear_mm, linscale=1.0, vmin=-limit, vmax=limit)


def colorbar_ticks(norm):
    """
    Ticks for a colour bar: for a SymLogNorm, 0 and the powers of ten from its
    linear threshold up to its limit (with both signs); otherwise None (automatic).
    """
    from matplotlib.colors import SymLogNorm

    if not isinstance(norm, SymLogNorm):
        return None
    powers = 10.0 ** np.arange(np.ceil(np.log10(norm.linthresh)), np.floor(np.log10(norm.vmax)) + 1)
    return np.concatenate([-powers[::-1], [0.0], powers])


def symmetric_norm(values, *, percentile=99.0):
    """A linear symmetric colour scale whose limit ignores the most extreme values."""
    from matplotlib.colors import Normalize

    limit = robust_limit(values, percentile=percentile)
    return Normalize(vmin=-limit, vmax=limit)


def frame_index(t_days, month_days):
    """Index of each month in the time steps of a run (the months are time levels)."""
    index = np.searchsorted(t_days, month_days)
    assert np.allclose(t_days[index], month_days), "The months are not time levels of the run"
    return index


def map_animation(state, fields, titles, labels, cmaps, norms, dates, pole_path_mas, frames, *,
                  pole_static_mas=None, frames_per_second):
    """
    Animation of one or more maps (Robinson projection, model coastlines) beside
    the pole path, all advancing month by month.

    Args:
        state: The pyslfp EarthState.
        fields: List of arrays, each of shape (n_frames, n_lat, n_lon) (NaN where not shown).
        titles: List of map titles; the date is appended.
        labels: List of colour-bar labels.
        cmaps: List of colour map names.
        norms: List of matplotlib colour norms.
        dates: Date of each frame (numpy datetime64).
        pole_path_mas: Dynamic pole at every time step (mas), shape (n_steps, 2).
        frames: Index into pole_path_mas of each frame.
        pole_static_mas: Static pole at each frame, or None.
        frames_per_second: Playback rate.

    Returns:
        The matplotlib FuncAnimation.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from matplotlib.animation import FuncAnimation

    assert len(fields) == len(titles) == len(labels) == len(cmaps) == len(norms), (
        "fields, titles, labels, cmaps and norms must have the same length"
    )
    lons, lats = state.lons(), state.lats()
    n_maps = len(fields)

    fig = plt.figure(figsize=(5.6 * n_maps + 3.8, 4.2))
    grid = fig.add_gridspec(1, n_maps + 1, width_ratios=[1.0] * n_maps + [0.62], wspace=0.22,
                            left=0.02, right=0.98, top=0.9, bottom=0.16)

    # --- Maps ---
    meshes = []
    map_titles = []
    for i in range(n_maps):
        ax_map = fig.add_subplot(grid[i], projection=ccrs.Robinson())
        mesh = ax_map.pcolormesh(
            lons,
            lats,
            fields[i][0],
            transform=ccrs.PlateCarree(),
            cmap=cmaps[i],
            norm=norms[i],
            shading="auto",
        )
        ax_map.contour(lons, lats, state.ocean_function.data, levels=[0.5], colors="k", linewidths=0.5,
                       transform=ccrs.PlateCarree())
        ax_map.set_global()
        colorbar = fig.colorbar(mesh, ax=ax_map, orientation="horizontal", pad=0.04, fraction=0.06, shrink=0.85,
                                extend="both", label=labels[i], ticks=colorbar_ticks(norms[i]))
        if colorbar_ticks(norms[i]) is not None:
            colorbar.ax.set_xticklabels([f"{tick:g}" for tick in colorbar_ticks(norms[i])])
        meshes.append(mesh)
        map_titles.append(ax_map.set_title(""))

    # --- Pole path ---
    ax_pole = fig.add_subplot(grid[n_maps])
    if pole_static_mas is not None:
        ax_pole.plot(pole_static_mas[:, 0], pole_static_mas[:, 1], "r--", lw=0.6, alpha=0.7, label="static")
        static_marker, = ax_pole.plot([], [], "ro", ms=4)
    ax_pole.plot(pole_path_mas[:, 0], pole_path_mas[:, 1], color="lightgrey", lw=0.6)
    trace, = ax_pole.plot([], [], "k", lw=1.0, label="dynamic")
    marker, = ax_pole.plot([], [], "ko")
    ax_pole.set_xlabel(r"$\omega_1/\Omega$ (mas)")
    ax_pole.set_ylabel(r"$\omega_2/\Omega$ (mas)")
    ax_pole.set_aspect("equal")
    ax_pole.grid(True, alpha=0.5)
    ax_pole.legend(loc="upper left", fontsize=8)
    ax_pole.set_title("Pole position")

    def update(n):
        for mesh, map_title, field, title in zip(meshes, map_titles, fields, titles):
            mesh.set_array(field[n].ravel())
            map_title.set_text(f"{title}, {str(dates[n])[:7]}")
        trace.set_data(pole_path_mas[: frames[n] + 1, 0], pole_path_mas[: frames[n] + 1, 1])
        marker.set_data([pole_path_mas[frames[n], 0]], [pole_path_mas[frames[n], 1]])
        artists = [*meshes, *map_titles, trace, marker]
        if pole_static_mas is not None:
            static_marker.set_data([pole_static_mas[n, 0]], [pole_static_mas[n, 1]])
            artists.append(static_marker)
        return artists

    animation = FuncAnimation(fig, update, frames=len(dates), interval=1000 / frames_per_second, blit=False)
    plt.close(fig)
    return animation
