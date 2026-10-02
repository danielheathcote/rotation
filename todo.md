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
