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
- `papers/` — reference papers (Heathcote et al. 2026 submission; Al-Attar et al. 2023/2024).
- `todo.md` — current plan.

Build a note with `latexmk -pdf <name>.tex` in `notes/`.
