"""
Downloads the JPL GRACE/GRACE-FO CRI mascons and caches their land and ocean
parts as spherical harmonic coefficients (see grace_data_helper.py).

Needs a NASA Earthdata login in ~/.netrc:
    machine urs.earthdata.nasa.gov login <username> password <password>

Run from the repository root:
    python new-code/grace/download_grace_data.py
"""

import sys
from pathlib import Path

# solver_dynamic lives one directory up, in new-code/
sys.path.append(str(Path(__file__).resolve().parents[1]))

import grace_data_helper as gh


#=======================================================================
# Parameters
#=======================================================================

data_dir = gh.DEFAULT_DATA_DIR

# Truncation degree of the cached coefficients (models of any lower degree can use them)
lmax = 90

#=======================================================================
# Download and process
#=======================================================================

paths = gh.download_jpl_mascons(data_dir)
print(f"Downloaded {len(paths)} files to {data_dir}")

mascons = gh.process_jpl_mascons(data_dir, lmax=lmax)
print(f"Cached {len(mascons['time_days'])} months, {mascons['dates'][0]} to {mascons['dates'][-1]}, "
      f"to degree {lmax}")
