#!/usr/bin/env python3
"""
Regrid ORAS5 temperature and salinity onto a MITgcm C‑grid.

This script performs both horizontal regridding from the native ORAS5
curvilinear grid to the MITgcm T‑point grid and vertical interpolation
from ORAS5 depth levels to the MITgcm RC levels.  It can optionally
reuse pre‑computed xESMF weights and will write both a CF‑compliant
NetCDF file and MITgcm binary files containing the regridded fields.

The workflow is as follows:

* Open the ORAS5 salinity and temperature NetCDF files using
  ``xarray.open_dataset``.  Missing values (``missing_value`` or
  ``_FillValue`` attributes) are converted to NaNs.  Times are
  decoded using cftime when necessary.  The longitude coordinate is
  shifted to match the longitude range of the MITgcm grid (0–360 or
  −180–180).  Data are transposed so that the horizontal dimensions
  (y, x) appear last and depth precedes them, i.e. (time, deptht,
  y, x).

* Open the MITgcm grid file and read ``XC``, ``YC``, ``RC`` and
  ``HFacC``.  These define the target longitude/latitude grid,
  vertical coordinates and the open‑water fraction mask.

* Construct two xESMF regridders: a bilinear regridder to perform
  first pass horizontal interpolation and a nearest‑neighbour regridder
  used only to fill any NaNs created during the bilinear step.  The
  ``periodic`` flag is set automatically based on the longitude range
  of the target grid.  If a weights file is provided via
  ``--weights`` it is used to load or store the bilinear weights.

* Horizontally regrid the ORAS5 variables at the requested time
  index.  xESMF handles any number of extra dimensions so depth
  levels and time are preserved.  After bilinear interpolation,
  remaining NaNs are replaced by nearest‑neighbour values only at
  those locations where the bilinear output is NaN.

* Vertically interpolate from ORAS5 depth levels (positive down) to
  MITgcm levels (positive up).  The absolute values of ``RC``
  therefore give depths in metres increasing downward.  For each
  horizontal location with open water (``HFacC>0``), a monotonic
  PCHIP interpolator is built from the valid ORAS5 depths and values.
  Interpolation is evaluated at each MITgcm depth.  Extrapolation
  below the deepest valid ORAS5 level is clamped so that the bottom
  value is replicated.  Cells above the shallowest ORAS5 level are
  also assigned the surface value.  Dry cells (``HFacC==0``) remain
  NaN in the NetCDF output and are set to zero in the MITgcm
  binaries.

* Apply the HFacC mask to ensure that any remaining values in dry
  cells are set to NaN.  For partial cells (0<HFacC<1) the
  interpolated values are retained; no scaling is applied because
  these fields represent state (temperature and salinity) rather
  than fluxes.

* Write a NetCDF file containing the final fields with appropriate
  coordinates and attributes.  The NetCDF file uses compression,
  little‑endian float32, and NaN fill values.  Finally write the
  MITgcm binary files for temperature and salinity using big‑endian
  float32 with X as the fastest dimension, as required by MITgcm.

Example usage::

    python regrid_oras5_to_mitgcm.py \
      --oras-s-file vosaline_control_monthly_highres_3D_195808_CONS_v0.1.nc \
      --oras-t-file votemper_control_monthly_highres_3D_195808_CONS_v0.1.nc \
      --grid-file grid.t001.nc \
      --time-index 0 \
      --out-nc ORAS5_on_MITgcm_Tgrid.nc \
      --out-bin-theta THETA_init.bin \
      --out-bin-salt SALT_init.bin \
      --weights weights_oras5_to_mitgcm_bilinear.nc

"""

import argparse
import logging
import os
from typing import Tuple, Optional

import numpy as np
import xarray as xr

try:
    import dask.array  # noqa: F401
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

import xesmf as xe
from scipy.interpolate import interp1d


def configure_logging(verbosity: int = 1) -> None:
    """Configure a basic logger for console output.

    Parameters
    ----------
    verbosity : int, default 1
        Verbosity level: 0 = warnings only, 1 = info, 2 = debug.
    """
    level_map = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    level = level_map.get(verbosity, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_oras(
    file_path: str,
    variable_name: str,
    chunks: Optional[dict] = None,
    use_dask_chunks: bool = True,
) -> xr.DataArray:
    """Load an ORAS5 variable as an xarray DataArray.

    Parameters
    ----------
    file_path : str
        Path to the ORAS5 NetCDF file.
    variable_name : str
        Name of the variable to load (e.g. 'vosaline' or 'votemper').
    chunks : dict, optional
        Dictionary of chunks for dask.  If None, a default chunking
        scheme is used when dask is available.
    use_dask_chunks : bool, default True
        If False, disable chunking and load the dataset without dask
        chunks to avoid performance warnings from mismatched chunks.

    Returns
    -------
    xr.DataArray
        DataArray containing the requested variable with time and
        depth dimensions ordered as (time, depth, y, x).  Missing
        values are converted to NaN.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"ORAS5 file not found: {file_path}. "
            "Please verify the path passed via --oras-s-file/--oras-t-file."
        )
    if chunks is None and DASK_AVAILABLE and use_dask_chunks:
        # Reasonable defaults; these can be tuned based on typical ORAS5
        chunks = {"x": 200, "y": 200, "deptht": 75}
    elif not use_dask_chunks:
        # Disable dask chunking entirely (load as a single chunk).
        chunks = None
    logging.info(f"Opening ORAS5 file {file_path} for variable {variable_name}")
    ds = xr.open_dataset(
        file_path,
        decode_times=True,
        chunks=chunks,
        mask_and_scale=True,
    )
    if variable_name not in ds:
        raise KeyError(f"Variable {variable_name} not found in {file_path}")
    da = ds[variable_name]
    # Convert fill values to NaN
    fill_value = None
    # Identify possible missing_value/_FillValue attributes
    for attr_name in ["_FillValue", "missing_value"]:
        if attr_name in da.attrs:
            fill_value = da.attrs.get(attr_name)
            break
    if fill_value is not None:
        da = da.where(da != fill_value)
    # Ensure dimension order: time, depth, y, x
    dims = list(da.dims)
    # Use names present in the dataset
    time_dim = next((d for d in dims if "time" in d), None)
    depth_dim = next((d for d in dims if d.lower().startswith("depth")), None)
    y_dim = next((d for d in dims if d.lower() in ("y", "lat", "latitude", "nav_lat")), None)
    x_dim = next((d for d in dims if d.lower() in ("x", "lon", "longitude", "nav_lon")), None)
    if any(d is None for d in [time_dim, depth_dim, y_dim, x_dim]):
        raise ValueError("Could not identify all required dimensions in ORAS5 file")
    da = da.transpose(time_dim, depth_dim, y_dim, x_dim)
    # Remove any length‑one time dimension via selection (time index applied later)
    return da


def load_grid(grid_file: str) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Load the MITgcm grid variables from a NetCDF file.

    Parameters
    ----------
    grid_file : str
        Path to the MITgcm grid file.

    Returns
    -------
    tuple of xr.DataArray
        (XC, YC, RC, HFacC) with dimensions (Y,X) for XC,YC and
        (Z,) for RC and (Z,Y,X) for HFacC.
    """
    logging.info(f"Opening MITgcm grid file {grid_file}")
    ds = xr.open_dataset(grid_file, decode_times=False)
    required = ["XC", "YC", "RC", "HFacC"]
    missing = [name for name in required if name not in ds]
    if missing:
        raise KeyError(f"Missing required grid variables in {grid_file}: {missing}")
    XC = ds["XC"]
    YC = ds["YC"]
    RC = ds["RC"]
    HFacC = ds["HFacC"]
    return XC, YC, RC, HFacC


def unify_longitude(nav_lon: xr.DataArray, target_lon: xr.DataArray) -> xr.DataArray:
    """Shift ORAS5 longitudes to match the range of the MITgcm grid.

    Parameters
    ----------
    nav_lon : xr.DataArray
        Original longitude coordinate from ORAS5 (shape (y,x)).
    target_lon : xr.DataArray
        Target longitude grid (XC) from MITgcm.

    Returns
    -------
    xr.DataArray
        Longitude values shifted to the same range as `target_lon`.
    """
    lon = nav_lon.copy()
    # Determine target range
    tgt_min = float(target_lon.min())
    tgt_max = float(target_lon.max())
    # Determine if MITgcm grid spans 0–360 or −180–180
    if tgt_min >= 0 and tgt_max <= 360:
        # Shift to 0–360
        logging.info("Shifting ORAS5 longitudes to 0–360 domain to match target grid")
        # Use modulo to wrap negative values
        lon_vals = np.mod(lon, 360)
        lon = xr.DataArray(lon_vals, dims=lon.dims, coords=lon.coords, attrs=lon.attrs)
    elif tgt_min < 0 and tgt_max <= 180:
        # Shift to −180–180
        logging.info("Shifting ORAS5 longitudes to −180–180 domain to match target grid")
        # Convert any values >180 down by 360
        lon_vals = ((lon + 180) % 360) - 180
        lon = xr.DataArray(lon_vals, dims=lon.dims, coords=lon.coords, attrs=lon.attrs)
    else:
        # Leave unchanged; target grid may already be curvilinear with unusual range
        logging.info("Leaving ORAS5 longitudes unchanged")
    return lon


def build_regridder(
    nav_lon: xr.DataArray,
    nav_lat: xr.DataArray,
    XC: xr.DataArray,
    YC: xr.DataArray,
    method: str = "bilinear",
    periodic: Optional[bool] = None,
    weights_path: Optional[str] = None,
    ignore_degenerate: bool = True,
) -> xe.Regridder:
    """Build an xESMF regridder between the ORAS5 and MITgcm grids.

    Parameters
    ----------
    nav_lon, nav_lat : xr.DataArray
        Source longitude and latitude fields with shape (y, x).
    XC, YC : xr.DataArray
        Target longitude and latitude fields with shape (Y, X).
    method : str, default 'bilinear'
        Regridding method to use.  Should be 'bilinear' or 'nearest_s2d'.
    periodic : bool, optional
        Whether to treat the grid as periodic in longitude.  If None,
        periodicity is inferred: True when the target longitude range
        covers a full 360°.
    weights_path : str, optional
        If provided, path to a weight file for reusing weights.  The file
        is read when existing and written otherwise.
    ignore_degenerate : bool, default True
        If True, instruct ESMF to ignore degenerate cells in the source
        grid (e.g., NaN or duplicated coordinates) when building weights.

    Returns
    -------
    xe.Regridder
        Configured regridder object.
    """
    # Build minimal xarray datasets required by xESMF
    src_lon = np.asfortranarray(nav_lon.data)
    src_lat = np.asfortranarray(nav_lat.data)
    src_mask = np.isfinite(src_lon) & np.isfinite(src_lat)
    ds_in = xr.Dataset({
        "lon": (nav_lon.dims, src_lon),
        "lat": (nav_lat.dims, src_lat),
        "mask": (nav_lon.dims, src_mask.astype(np.int8)),
    })
    ds_out = xr.Dataset({
        "lon": (XC.dims, np.asfortranarray(XC.data)),
        "lat": (YC.dims, np.asfortranarray(YC.data)),
    })
    if periodic is None:
        # Determine periodicity: True if full 360° coverage
        lon_min = float(XC.min())
        lon_max = float(XC.max())
        periodic = (lon_max - lon_min) >= 360 - 1e-6
    reuse = False
    filename = None
    if weights_path:
        filename = weights_path
        reuse = os.path.exists(weights_path)
    logging.info(
        f"Building regridder (method={method}, periodic={periodic}, weights file={filename}, reuse={reuse})"
    )
    regridder = xe.Regridder(
        ds_in,
        ds_out,
        method,
        periodic=periodic,
        filename=filename,
        reuse_weights=reuse,
        ignore_degenerate=ignore_degenerate,
    )
    return regridder


def subset_oras_to_target(
    sal_da: xr.DataArray,
    temp_da: xr.DataArray,
    nav_lon: xr.DataArray,
    nav_lat: xr.DataArray,
    XC: xr.DataArray,
    YC: xr.DataArray,
    margin_deg: float = 1.0,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Subset ORAS5 data to a bounding box around the target grid.

    Parameters
    ----------
    sal_da, temp_da : xr.DataArray
        ORAS5 salinity and temperature (time, depth, y, x).
    nav_lon, nav_lat : xr.DataArray
        ORAS5 longitude/latitude on the native grid (y, x).
    XC, YC : xr.DataArray
        MITgcm target longitude/latitude grids (Y, X).
    margin_deg : float, default 1.0
        Degrees to expand the target bounding box on each side.

    Returns
    -------
    tuple of xr.DataArray
        (sal_subset, temp_subset, nav_lon_subset, nav_lat_subset).
    """
    lon_min = float(XC.min()) - margin_deg
    lon_max = float(XC.max()) + margin_deg
    lat_min = float(YC.min()) - margin_deg
    lat_max = float(YC.max()) + margin_deg
    mask = (
        (nav_lon >= lon_min)
        & (nav_lon <= lon_max)
        & (nav_lat >= lat_min)
        & (nav_lat <= lat_max)
    )
    if not np.any(mask):
        logging.warning("No ORAS5 points found within target bounding box; skipping subsetting.")
        return sal_da, temp_da, nav_lon, nav_lat
    y_idx, x_idx = np.where(mask)
    y_min, y_max = int(y_idx.min()), int(y_idx.max())
    x_min, x_max = int(x_idx.min()), int(x_idx.max())
    y_dim, x_dim = nav_lon.dims
    logging.info(
        "Subsetting ORAS5 grid to y=%s:%s, x=%s:%s (margin=%s°)",
        y_min,
        y_max,
        x_min,
        x_max,
        margin_deg,
    )
    subset_sel = {y_dim: slice(y_min, y_max + 1), x_dim: slice(x_min, x_max + 1)}
    sal_sub = sal_da.isel(subset_sel)
    temp_sub = temp_da.isel(subset_sel)
    nav_lon_sub = nav_lon.isel(subset_sel)
    nav_lat_sub = nav_lat.isel(subset_sel)
    return sal_sub, temp_sub, nav_lon_sub, nav_lat_sub


def regrid_horizontal(
    data: xr.DataArray,
    bilinear: xe.Regridder,
    nearest: xe.Regridder,
) -> xr.DataArray:
    """Perform horizontal interpolation on a 3D field (depth, y, x).

    The function interpolates the rightmost two dimensions (y, x) of
    `data` onto the MITgcm grid using bilinear interpolation and then
    fills any NaNs using nearest neighbour interpolation.  Extra
    dimensions (such as depth) are preserved.

    Parameters
    ----------
    data : xr.DataArray
        Input data with shape (..., y, x).
    bilinear : xe.Regridder
        Bilinear regridder object created by :func:`build_regridder`.
    nearest : xe.Regridder
        Nearest neighbour regridder used for filling NaNs.

    Returns
    -------
    xr.DataArray
        Data horizontally regridded with shape (..., Y, X).
    """
    # xESMF expects the horizontal dims to be the last two; ensure this
    orig_dims = data.dims
    y_dim, x_dim = orig_dims[-2:]
    transposed = data
    if (y_dim, x_dim) != (data.dims[-2], data.dims[-1]):
        # unlikely but safe
        transposed = data.transpose(..., y_dim, x_dim)
    # Apply bilinear regridding
    logging.info("Performing bilinear horizontal regridding")
    bilinear_data = bilinear(transposed, skipna=True)
    # Identify NaNs left after bilinear step
    nan_mask = np.isnan(bilinear_data)
    # Use nearest neighbour only on NaNs
    if nan_mask.any():
        logging.info("Filling coastal holes with nearest neighbour regridding")
        nearest_data = nearest(transposed, skipna=True)
        filled = bilinear_data.where(~nan_mask, nearest_data)
    else:
        filled = bilinear_data
    # Preserve original name and attributes
    filled = filled.rename(data.name)
    return filled


def interpolate_column(
    z_oras: np.ndarray,
    values: np.ndarray,
    z_mit: np.ndarray,
    clamp_bottom: bool = True,
) -> np.ndarray:
    """Interpolate a single vertical profile.

    A helper for :func:`interp_vertical`.  Given a one‑dimensional
    vertical profile defined at depths ``z_oras`` with values ``values``,
    return the values interpolated at depths ``z_mit``.  The function
    removes NaNs, performs linear interpolation, and clamps the
    extrapolation at the bottom to the deepest valid value.

    Parameters
    ----------
    z_oras : np.ndarray
        1D array of ORAS5 depths (positive down).
    values : np.ndarray
        1D array of values corresponding to z_oras (same length).
    z_mit : np.ndarray
        1D array of MITgcm depths (positive down, derived from abs(RC)).
    clamp_bottom : bool, default True
        If True, extend the deepest value to all deeper levels beyond
        the last valid z_oras.

    Returns
    -------
    np.ndarray
        Interpolated values at z_mit (length len(z_mit)).  Positions
        where interpolation is not possible because all input values
        are NaN are filled with NaN.
    """
    # Remove NaNs
    valid = ~np.isnan(values)
    if not np.any(valid):
        return np.full_like(z_mit, np.nan, dtype=float)
    z_valid = z_oras[valid]
    v_valid = values[valid]
    sort_idx = np.argsort(z_valid)
    z_sorted = z_valid[sort_idx]
    v_sorted = v_valid[sort_idx]
    # If there's only one valid point, replicate it everywhere
    if v_sorted.size == 1:
        out = np.full_like(z_mit, v_sorted[0], dtype=float)
    else:
        try:
            interp_func = PchipInterpolator(z_sorted, v_sorted, extrapolate=True)
        except Exception:
            # Fallback to linear
            interp_func = interp1d(
                z_sorted,
                v_sorted,
                kind="linear",
                bounds_error=False,
                fill_value=(v_sorted[0], v_sorted[-1]),
            )
        out = interp_func(z_mit)
    if clamp_bottom:
        # Clamp values below the deepest observation to the deepest value
        deepest_depth = z_sorted[-1]
        deepest_val = v_sorted[-1]
        mask = z_mit > deepest_depth
        out = np.where(mask, deepest_val, out)
        # Clamp above the shallowest to the shallowest value
        shallowest_depth = z_sorted[0]
        shallowest_val = v_sorted[0]
        mask_top = z_mit < shallowest_depth
        out = np.where(mask_top, shallowest_val, out)
    return out


def interp_vertical(
    data: xr.DataArray,
    z_oras: xr.DataArray,
    RC: xr.DataArray,
    HFacC: xr.DataArray,
    varname: str,
) -> xr.DataArray:
    """Interpolate a horizontally regridded field onto MITgcm vertical levels.

    Parameters
    ----------
    data : xr.DataArray
        Horizontally regridded data with dimensions (deptht, Y, X).
    z_oras : xr.DataArray
        ORAS5 depth coordinate (positive down).
    RC : xr.DataArray
        MITgcm RC coordinate (positive up; negative values indicate
        depth).  The absolute value is used to compute depths (positive down).
    HFacC : xr.DataArray
        MITgcm mask with shape (Z, Y, X).  Values of 0 represent dry
        cells.
    varname : str
        Name to assign to the resulting DataArray.

    Returns
    -------
    xr.DataArray
        Vertically interpolated data with dimensions (Z, Y, X).
    """
    logging.info(f"Performing vertical interpolation for {varname}")
    z_mit = np.abs(RC.values).astype(float)
    z_oras_vals = z_oras.values.astype(float)
    depth_dim = data.dims[0]
    # Convert to numpy for speed
    arr = data.values  # shape (depth, Y, X)
    ny, nx = arr.shape[1], arr.shape[2]
    nz = z_mit.size
    out = np.full((nz, ny, nx), np.nan, dtype=np.float32)
    for j in range(ny):
        for i in range(nx):
            # Only interpolate where at least one level is wet
            if not np.any(HFacC[:, j, i] > 0):
                continue
            col_values = arr[:, j, i]
            interpolated = interpolate_column(z_oras_vals, col_values, z_mit)
            out[:, j, i] = interpolated.astype(np.float32)
    coords = {
        RC.dims[0]: RC,
        data.dims[1]: data.coords[data.dims[1]],
        data.dims[2]: data.coords[data.dims[2]],
    }
    da_out = xr.DataArray(out, dims=(RC.dims[0], data.dims[1], data.dims[2]), coords=coords, name=varname)
    return da_out


def apply_hfac_mask(data: xr.DataArray, HFacC: xr.DataArray) -> xr.DataArray:
    """Apply the HFacC mask to a vertically interpolated field.

    Parameters
    ----------
    data : xr.DataArray
        Vertically interpolated data with dimensions (Z, Y, X).
    HFacC : xr.DataArray
        MITgcm mask with shape (Z, Y, X).  Zero values indicate dry
        cells; partial values are retained.

    Returns
    -------
    xr.DataArray
        Data with NaNs where HFacC == 0.
    """
    mask = HFacC == 0
    data = data.where(~mask)
    return data


def align_hfacc_to_data(data: xr.DataArray, HFacC: xr.DataArray) -> xr.DataArray:
    """Align HFacC to the data grid by slicing extra points if needed."""
    hfac_aligned = HFacC
    for dim in data.dims:
        if dim in hfac_aligned.dims:
            data_size = data.sizes[dim]
            hfac_size = hfac_aligned.sizes[dim]
            if hfac_size != data_size:
                if hfac_size < data_size:
                    raise ValueError(
                        f"HFacC dimension {dim} is smaller ({hfac_size}) than data ({data_size})"
                    )
                logging.warning(
                    "Aligning HFacC dim %s from %s to %s by slicing.",
                    dim,
                    hfac_size,
                    data_size,
                )
                hfac_aligned = hfac_aligned.isel({dim: slice(0, data_size)})
    return hfac_aligned


def report_stats(stage: str, da: xr.DataArray) -> None:
    """Print basic statistics for a DataArray.

    Parameters
    ----------
    stage : str
        Label describing the processing stage.
    da : xr.DataArray
        Data to summarise.
    """
    arr = da.values
    finite = np.isfinite(arr)
    total = arr.size
    finite_count = finite.sum()
    nan_count = total - finite_count
    nan_pct = 100.0 * nan_count / total
    if finite_count > 0:
        valid_vals = arr[finite]
        vmin = float(valid_vals.min())
        vmax = float(valid_vals.max())
        vmean = float(valid_vals.mean())
        logging.info(f"{stage}: min={vmin:.3f}, max={vmax:.3f}, mean={vmean:.3f}, NaN%={nan_pct:.2f}")
    else:
        logging.info(f"{stage}: all values are NaN")


def write_netcdf(out_nc: str, theta: xr.DataArray, salt: xr.DataArray) -> None:
    """Write the interpolated fields to a NetCDF file.

    Parameters
    ----------
    out_nc : str
        Output NetCDF file path.
    theta, salt : xr.DataArray
        Temperature and salinity fields with dimensions (Z, Y, X).
    """
    logging.info(f"Writing NetCDF output to {out_nc}")
    ds = xr.Dataset({
        "theta": theta.astype(np.float32),
        "salt": salt.astype(np.float32),
    })
    # Assign attributes
    ds.theta.attrs.update({"long_name": "Potential temperature", "units": "degC"})
    ds.salt.attrs.update({"long_name": "Salinity", "units": "PSU"})
    ds[theta.dims[0]].attrs.update({"long_name": "Depth", "units": "m", "positive": "up"})
    ds[theta.dims[1]].attrs.update({"long_name": "Latitude", "units": "degrees_north"})
    ds[theta.dims[2]].attrs.update({"long_name": "Longitude", "units": "degrees_east"})
    ds.attrs.update({
        "title": "ORAS5 regridded onto MITgcm T‑points",
        "source": "ORAS5 to MITgcm regridding",
        "history": f"Created by regrid_oras5_to_mitgcm script",
    })
    # Compression encoding
    encoding = {
        "theta": {"zlib": True, "complevel": 4, "_FillValue": np.nan},
        "salt": {"zlib": True, "complevel": 4, "_FillValue": np.nan},
    }
    ds.to_netcdf(out_nc, encoding=encoding)


def write_mitgcm_bin(out_path: str, data: xr.DataArray, HFacC: xr.DataArray) -> None:
    """Write a DataArray to a MITgcm binary file.

    Data are written in big‑endian float32 with X as the fastest
    dimension.  NaNs in dry cells (where HFacC==0) are replaced by
    zero; NaNs in wet cells will raise an error to ensure that the
    vertical interpolation covered all wet cells.

    Parameters
    ----------
    out_path : str
        Path to the output binary file.
    data : xr.DataArray
        Data with dimensions (Z, Y, X).
    HFacC : xr.DataArray
        Mask with dimensions matching data.
    """
    logging.info(f"Writing MITgcm binary to {out_path}")
    arr = data.values.copy()
    mask_dry = HFacC.values == 0
    # Replace NaN with 0 only in dry cells
    nan_mask = np.isnan(arr)
    if np.any(nan_mask & ~mask_dry):
        # There are NaNs in wet cells
        raise ValueError(f"Wet cells contain NaN values in {out_path}")
    arr[mask_dry] = 0.0
    # Flatten in C order (X fastest) – data is already ordered (Z,Y,X)
    arr = arr.astype('>f4')
    with open(out_path, 'wb') as f:
        arr.tofile(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regrid ORAS5 temperature and salinity onto a MITgcm grid",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--oras-s-file", required=True, help="Path to ORAS5 salinity file")
    parser.add_argument("--oras-t-file", required=True, help="Path to ORAS5 temperature file")
    parser.add_argument("--grid-file", required=True, help="Path to MITgcm grid file")
    parser.add_argument("--time-index", type=int, default=0, help="Time index to extract from ORAS5")
    parser.add_argument(
        "--oras-chunks",
        nargs=3,
        type=int,
        metavar=("Y", "X", "DEPTH"),
        help="Optional ORAS5 chunk sizes for (y x deptht).",
    )
    parser.add_argument(
        "--no-dask-chunks",
        action="store_true",
        help="Disable dask chunking when opening ORAS5 files.",
    )
    parser.add_argument(
        "--subset-margin",
        type=float,
        default=1.0,
        help="Degrees to expand target bbox when subsetting ORAS5 data.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Verbosity level: 0=warnings, 1=info, 2=debug",
    )
    parser.add_argument("--out-nc", required=True, help="Output NetCDF file path")
    parser.add_argument("--out-bin-theta", required=True, help="Output MITgcm binary path for temperature")
    parser.add_argument("--out-bin-salt", required=True, help="Output MITgcm binary path for salinity")
    parser.add_argument(
        "--output-chunks",
        nargs=2,
        type=int,
        metavar=("Y", "X"),
        help="Optional output chunk sizes for regridding (target grid Y X)",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional path to xESMF weights file for regridding",
    )
    args = parser.parse_args()
    configure_logging(args.verbose)
    # Load grid
    XC, YC, RC, HFacC = load_grid(args.grid_file)
    # Load ORAS5 variables
    # Chunking controls for ORAS5 inputs: user override or disable dask chunks.
    oras_chunks = None
    if args.oras_chunks:
        # Map CLI order (Y, X, DEPTH) to ORAS5 dimension names.
        oras_chunks = {"y": args.oras_chunks[0], "x": args.oras_chunks[1], "deptht": args.oras_chunks[2]}
    use_dask_chunks = not args.no_dask_chunks
    sal_da = load_oras(args.oras_s_file, "vosaline", chunks=oras_chunks, use_dask_chunks=use_dask_chunks)
    temp_da = load_oras(args.oras_t_file, "votemper", chunks=oras_chunks, use_dask_chunks=use_dask_chunks)
    # Align longitudes
    nav_lon = sal_da.coords.get("nav_lon")
    nav_lat = sal_da.coords.get("nav_lat")
    if nav_lon is None or nav_lat is None:
        # Some ORAS5 files may use different names
        nav_lon = sal_da.coords.get("longitude", None)
        nav_lat = sal_da.coords.get("latitude", None)
    if nav_lon is None or nav_lat is None:
        raise KeyError("Could not find nav_lon/nav_lat coordinates in ORAS5 file")
    nav_lon_shifted = unify_longitude(nav_lon, XC)
    sal_da, temp_da, nav_lon_shifted, nav_lat = subset_oras_to_target(
        sal_da,
        temp_da,
        nav_lon_shifted,
        nav_lat,
        XC,
        YC,
        margin_deg=args.subset_margin,
    )
    # Build regridders
    bilinear = build_regridder(
        nav_lon_shifted,
        nav_lat,
        XC,
        YC,
        method="bilinear",
        periodic=None,
        weights_path=args.weights,
    )
    nearest = build_regridder(
        nav_lon_shifted,
        nav_lat,
        XC,
        YC,
        method="nearest_s2d",
        periodic=None,
        weights_path=None,
    )
    # Extract the requested time index
    sal_t = sal_da.isel({sal_da.dims[0]: args.time_index})
    temp_t = temp_da.isel({temp_da.dims[0]: args.time_index})
    # Horizontal regridding
    sal_h = regrid_horizontal(sal_t, bilinear, nearest)
    temp_h = regrid_horizontal(temp_t, bilinear, nearest)
    # Report stats after horizontal interpolation
    report_stats("Salinity after horizontal regridding", sal_h)
    report_stats("Temperature after horizontal regridding", temp_h)
    # Vertical interpolation
    z_oras = sal_da.coords[next((d for d in sal_da.dims if d.lower().startswith("depth")), None)]
    sal_v = interp_vertical(sal_h, z_oras, RC, HFacC, varname="salt")
    temp_v = interp_vertical(temp_h, z_oras, RC, HFacC, varname="theta")
    # Apply HFacC mask
    HFacC_aligned = align_hfacc_to_data(sal_v, HFacC)
    sal_masked = apply_hfac_mask(sal_v, HFacC_aligned)
    temp_masked = apply_hfac_mask(temp_v, HFacC_aligned)
    # Report stats after vertical interpolation and masking
    report_stats("Salinity after vertical interpolation", sal_masked)
    report_stats("Temperature after vertical interpolation", temp_masked)
    # Verify shapes
    expected_shape = HFacC_aligned.shape
    if sal_masked.shape != expected_shape:
        raise ValueError(f"Salinity shape {sal_masked.shape} does not match expected {expected_shape}")
    if temp_masked.shape != expected_shape:
        raise ValueError(f"Temperature shape {temp_masked.shape} does not match expected {expected_shape}")
    # Write outputs
    write_netcdf(args.out_nc, temp_masked, sal_masked)
    write_mitgcm_bin(args.out_bin_theta, temp_masked, HFacC_aligned)
    write_mitgcm_bin(args.out_bin_salt, sal_masked, HFacC_aligned)
    logging.info("Regridding complete")


if __name__ == "__main__":
    main()
