# Dynamic rotation — plan

Status as of 2 October 2026.

## Documents

Working (in `notes/`):
- `rotation_dynamic_elastic.pdf` (main note, "N"): the leading-order dynamic theory, with j̇ dropped.
  Covers the fluid core (via A_w), damping (γ), the generalised problem and reciprocity.
- `rotation_jdot.pdf`: the narrow problem with j̇, h and Γ kept, using the ℓ formulation.
  Includes the ocean; no fluid core and no damping.

Useful (in `notes/`):
- `gyroscopic_matrix_N.pdf`: how the fluid core and damping enter the matrix N.
- `section7_derivations.pdf`: derivations of N §7 and of the damped form in N §8.1.

On ice (in `notes/on-ice/`): `rotation_alattar_notation`, `rotation_forward_problem`.

## Plan

1. **Dan:** get to grips with the equatorial angular-momentum (ℓ) formulation of the problem.
   Write a basic solver based on pyslfp, and record the exact algorithm implemented.
   - Starting point: `notes/rotation_jdot.pdf`, §5–7. It gives the trapezoid step on ℓ, the
     algorithm box, the pyslfp changes and the initial conditions.
   - Checks: the test cases in `notes/rotation_jdot.pdf` §8.

2. Apply the general case (j̇ and ḣ included) to the main document (N). Probably make a new
   version of N that uses ℓ as the state variable.

3. Create a full version with damping and a fluid core.

## Real forcing from GRACE (started 22 October 2026)

Data: JPL GRACE/GRACE-FO mascons, RL06.3Mv04, CRI-filtered (0.5° grid, monthly from April 2002).
C20/C30 from SLR, degree 1 estimated, GIA (ICE6G-D) removed. Anomalies relative to 2004.0–2009.999.
Download needs a NASA Earthdata login in `~/.netrc`.

1. Done: download and load helper, `new-code/grace_data_helper.py` and `new-code/download_grace_data.py`.
2. Done: the direct ocean load D, `new-code/grace_direct_ocean.py`, explained in
   `new-notes/grace_direct_ocean.pdf`. O = D + I(L + D; mean-pole rotation), solved by iteration. The combined
   direct load L + D depends only on the GRACE total (up to a harmless uniform ocean layer).
   Saved with and without GAD in `data/grace/jpl_direct_ocean_lmax64.npz`.
   Mean pole: linear (Ries 2017 = IERS 2018 secular pole), per the JPL RL06 standards. GAD: keep (recommended);
   the removed uniform layer equals JPL TN-12 (mean atmospheric mass over the ocean).
3. Done: `new-code/grace_rotation_run.py` (forcing land / land-ocean / land-ocean-nogad; saved in
   `data/grace/runs/`). Original plan: the dynamic runs driven by L + D (and by L alone for comparison), with a `CoefficientLoad`
   (cubic-spline interpolation, zeta_dot from its derivative) and an equilibrium start. Compare dynamic and
   static rotation and sea level.
4. NEXT (immediate): make some plots of the runs in `data/grace/runs/` (dynamic vs static pole, the three
   forcings, sea-level differences). The load is the natural cubic spline through the monthly values, with no
   smoothing (`new-notes/grace_time_interpolation.pdf`); the 2017-18 gap is bridged by it, and that is accepted.
5. Parked: the angular momentum h of atmospheric winds and ocean currents (the runs are mass-driven only).
