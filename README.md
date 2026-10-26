# rotation

Theory notes on dynamic rotational feedbacks in the elastic sea-level (fingerprint) problem,
extending Al-Attar et al. (2024) from a static to a dynamically rotating Earth.

## Layout

- `notes/` — LaTeX notes and their PDFs.
  - `rotation_dynamic_elastic` — main note ("N"). Working.
  - `rotation_jdot` — j̇ kept, angular-momentum (ℓ) formulation. Working.
  - `rotation_wdot` — the same problem as an ω̇ = f(ω, t) form for any time integrator. Working.
  - `gyroscopic_matrix_N`, `section7_derivations` — supporting derivations.
  - `on-ice/` — earlier notes, not currently in use.
- `new-notes/` — `rotation_dynamic`: standalone master note (j̇ kept, ω̇ fixed-point solver,
  analytic solutions, reciprocity and adjoint).
- `new-code/` — `solver_dynamic.py`: forward solver of `new-notes/rotation_dynamic_2vector`;
  `examples_dynamic.ipynb` (with `examples_dynamic_helper.py`): examples.
- `data/` — downloaded data (gitignored; see below).
- `papers/` — reference papers (Heathcote et al. 2026 submission; Al-Attar et al. 2023/2024).
- `todo.md` — current plan.

Build a note with `latexmk -pdf <name>.tex` in `notes/`.

## Data

GRACE/GRACE-FO loads come from the JPL RL06.3Mv04 CRI mascons. They need a NASA Earthdata login in
`~/.netrc` (`machine urs.earthdata.nasa.gov login <username> password <password>`). Then, from the
repository root,

    python new-code/grace/download_grace_data.py

downloads them into `data/grace/` and caches their land and ocean parts as spherical harmonic
coefficients (`new-code/grace/grace_data_helper.py`).

The direct ocean load (the ocean mass change that is not the ocean's response to loading) and the
dynamic runs follow:

    python new-code/grace/grace_direct_ocean.py --lmax 64
    python new-code/grace/grace_rotation_run.py --forcing land-ocean --lmax 64

The runs are saved in `data/grace/runs/`; `--help` lists the options.
The notebooks in `new-code/grace/` plot them (`grace_plots.ipynb`) and animate them
(`animation_grace_*.ipynb`, writing GIFs to `new-code/grace/animations/`).
