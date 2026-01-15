#!/usr/bin/env python3
"""Regrid ORAS5 temperature/salinity to MITgcm T-grid.

This script performs horizontal (xESMF) and vertical (PCHIP) interpolation
from ORAS5 curvilinear lat/lon grid to MITgcm T-grid cell centers, applies
HFacC masking, and writes both NetCDF and MITgcm binary outputs.

Example:
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

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import xarray as xr
import xesmf as xe
from scipy.interpolate import PchipInterpolator


def dask_available() -> bool:
    """Return True if dask is available."""
    import importlib.util

    return importlib.util.find_spec("dask") is not None


def setup_logging() -> None:
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def replace_missing(da: xr.DataArray) -> xr.DataArray:
    """Replace missing_value and _FillValue with NaN."""
    missing_values = []
    if "missing_value" in da.attrs:
        missing_values.append(da.attrs["missing_value"])
    if "_FillValue" in da.attrs:
        missing_values.append(da.attrs["_FillValue"])
    for mv in missing_values:
        da = da.where(da != mv)
    return da


def load_oras(path: str, var_name: str, chunks: Optional[dict]) -> xr.DataArray:
    """Load ORAS5 variable as DataArray and clean missing values."""
    ds = xr.open_dataset(path, chunks=chunks, decode_times=True)
    if var_name not in ds:
        raise KeyError(f"Variable '{var_name}' not found in {path}")
    da = ds[var_name]
    if "nav_lon" in ds and "nav_lat" in ds:
        da = da.assign_coords(nav_lon=ds["nav_lon"], nav_lat=ds["nav_lat"])
    da = replace_missing(da)
    da = da.transpose("time_counter", "deptht", "y", "x")
    return da


def load_grid(path: str) -> xr.Dataset:
    """Load MITgcm grid dataset."""
    ds = xr.open_dataset(path)
    for req in ["XC", "YC", "RC", "HFacC"]:
        if req not in ds:
            raise KeyError(f"Missing '{req}' in MITgcm grid file {path}")
    return ds


def infer_lon_range(xc: xr.DataArray) -> Tuple[float, float]:
    """Infer longitude range from MITgcm XC."""
    lon_min = float(np.nanmin(xc.values))
    lon_max = float(np.nanmax(xc.values))
    return lon_min, lon_max


def normalize_lon(lon: xr.DataArray, target_range: str) -> xr.DataArray:
    """Normalize longitude to target range ('0_360' or 'neg180_180')."""
    if target_range == "0_360":
        return (lon % 360.0).where(np.isfinite(lon))
    if target_range == "neg180_180":
        return (((lon + 180.0) % 360.0) - 180.0).where(np.isfinite(lon))
    raise ValueError(f"Unknown target_range: {target_range}")


def build_regridder(
    nav_lon: xr.DataArray,
    nav_lat: xr.DataArray,
    xc: xr.DataArray,
    yc: xr.DataArray,
    weights_path: Optional[str],
) -> Tuple[xe.Regridder, xe.Regridder, bool]:
    """Build bilinear and nearest regridders."""
    lon_min = float(np.nanmin(xc.values))
    lon_max = float(np.nanmax(xc.values))
    if lon_min >= 0.0 and lon_max > 180.0:
        target_range = "0_360"
    else:
        target_range = "neg180_180"

    nav_lon = normalize_lon(nav_lon, target_range)

    span = float(np.nanmax(nav_lon.values) - np.nanmin(nav_lon.values))
    periodic = bool(span > 350.0 and target_range == "0_360")

    ds_in = xr.Dataset({"lon": nav_lon, "lat": nav_lat})
    ds_out = xr.Dataset({"lon": xc, "lat": yc})

    reuse_weights = False
    if weights_path:
        reuse_weights = os.path.exists(weights_path)

    regrid_bilinear = xe.Regridder(
        ds_in,
        ds_out,
        "bilinear",
        periodic=periodic,
        filename=weights_path,
        reuse_weights=reuse_weights,
    )

    regrid_nearest = xe.Regridder(
        ds_in,
        ds_out,
        "nearest_s2d",
        periodic=periodic,
        reuse_weights=False,
    )

    return regrid_bilinear, regrid_nearest, periodic


def log_stats(label: str, data: xr.DataArray) -> None:
    """Log min/max/mean and NaN percentage."""
    finite = np.isfinite(data.values)
    nan_pct = 100.0 * (1.0 - finite.mean())
    if finite.any():
        data_min = float(np.nanmin(data.values))
        data_max = float(np.nanmax(data.values))
        data_mean = float(np.nanmean(data.values))
        logging.info(
            "%s: min=%s max=%s mean=%s NaN%%=%.2f",
            label,
            f"{data_min:.4f}",
            f"{data_max:.4f}",
            f"{data_mean:.4f}",
            nan_pct,
        )
    else:
        logging.info("%s: all NaN", label)


def regrid_horizontal(
    da: xr.DataArray,
    regrid_bilinear: xe.Regridder,
    regrid_nearest: xe.Regridder,
) -> xr.DataArray:
    """Apply bilinear regridding and fill NaNs with nearest neighbor."""
    bilinear = regrid_bilinear(da)
    if np.isnan(bilinear).any():
        nearest = regrid_nearest(da)
        bilinear = xr.where(np.isnan(bilinear), nearest, bilinear)
    return bilinear


def interp_vertical(
    data: xr.DataArray,
    z_oras: np.ndarray,
    z_mit: np.ndarray,
    hfac: np.ndarray,
) -> xr.DataArray:
    """Interpolate vertically using PCHIP, with bottom clamp."""
    nz_mit = z_mit.size
    ny, nx = data.shape[1], data.shape[2]
    out = np.full((nz_mit, ny, nx), np.nan, dtype=np.float32)

    for j in range(ny):
        for i in range(nx):
            if np.nanmax(hfac[:, j, i]) <= 0.0:
                continue
            profile = data[:, j, i]
            valid = np.isfinite(profile)
            if valid.sum() == 0:
                continue
            if valid.sum() == 1:
                out[:, j, i] = profile[valid][0]
                continue
            z_valid = z_oras[valid]
            pchip = PchipInterpolator(z_valid, profile[valid], extrapolate=False)
            interp_vals = pchip(z_mit)
            if np.isnan(interp_vals).any():
                deep_mask = np.isnan(interp_vals) & (z_mit > z_valid.max())
                shallow_mask = np.isnan(interp_vals) & (z_mit < z_valid.min())
                interp_vals[deep_mask] = profile[valid][-1]
                interp_vals[shallow_mask] = profile[valid][0]
            out[:, j, i] = interp_vals.astype(np.float32)
    return xr.DataArray(out, dims=("Z", "Y", "X"))


def apply_hfac_mask(data: xr.DataArray, hfac: xr.DataArray) -> xr.DataArray:
    """Apply HFacC mask to data."""
    return data.where(hfac > 0.0)


def write_netcdf(
    theta: xr.DataArray,
    salt: xr.DataArray,
    xc: xr.DataArray,
    yc: xr.DataArray,
    rc: xr.DataArray,
    out_path: str,
    history: str,
    time_ref: str,
) -> None:
    """Write CF-compliant NetCDF output."""
    ny, nx = yc.shape
    ds_out = xr.Dataset(
        {
            "theta": theta.astype(np.float32),
            "salt": salt.astype(np.float32),
        },
        coords={
            "Z": ("Z", rc.values, {"long_name": "Vertical coordinate", "units": "m"}),
            "Y": ("Y", np.arange(ny), {"long_name": "Y index"}),
            "X": ("X", np.arange(nx), {"long_name": "X index"}),
            "YC": (
                ("Y", "X"),
                yc.values,
                {"long_name": "Latitude", "units": "degrees_north"},
            ),
            "XC": (
                ("Y", "X"),
                xc.values,
                {"long_name": "Longitude", "units": "degrees_east"},
            ),
        },
        attrs={
            "title": "ORAS5 regridded to MITgcm T-grid",
            "history": history,
            "source": "ORAS5 -> MITgcm",
            "time_reference": time_ref,
            "created": datetime.utcnow().isoformat() + "Z",
        },
    )

    ds_out["theta"].attrs.update({"long_name": "Potential temperature", "units": "degC"})
    ds_out["salt"].attrs.update({"long_name": "Salinity", "units": "PSU"})

    encoding = {
        "theta": {"zlib": True, "complevel": 4, "_FillValue": np.nan},
        "salt": {"zlib": True, "complevel": 4, "_FillValue": np.nan},
    }

    ds_out.to_netcdf(out_path, encoding=encoding)
    logging.info("Wrote NetCDF: %s", out_path)


def write_mitgcm_bin(data: xr.DataArray, hfac: xr.DataArray, out_path: str) -> None:
    """Write MITgcm binary file (big-endian float32)."""
    arr = data.values.copy()
    dry = hfac.values == 0.0
    wet_nan = np.isnan(arr) & (~dry)
    if np.any(wet_nan):
        logging.warning("NaN values remain in wet cells: %s", int(np.sum(wet_nan)))
    arr[np.isnan(arr) & dry] = 0.0
    arr = arr.astype(">f4")
    arr.tofile(out_path)
    logging.info(
        "Wrote MITgcm binary: %s (dtype=%s, endian=big, order=Z,Y,X)",
        out_path,
        arr.dtype,
    )


def build_uv_placeholder() -> None:
    """Placeholder for future U/V face interpolation."""
    return None


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Regrid ORAS5 temperature/salinity to MITgcm T-grid.",
    )
    parser.add_argument("--oras-s-file", required=True, help="Path to ORAS5 salinity file")
    parser.add_argument("--oras-t-file", required=True, help="Path to ORAS5 temperature file")
    parser.add_argument("--grid-file", required=True, help="Path to MITgcm grid netCDF")
    parser.add_argument("--time-index", type=int, default=0, help="Time index to use")
    parser.add_argument("--out-nc", required=True, help="Output NetCDF path")
    parser.add_argument("--out-bin-theta", required=True, help="Output MITgcm theta binary")
    parser.add_argument("--out-bin-salt", required=True, help="Output MITgcm salt binary")
    parser.add_argument("--weights", default=None, help="Path to xESMF weights file")
    args = parser.parse_args()

    setup_logging()

    chunks = {"x": 200, "y": 200, "deptht": 75} if dask_available() else None
    if chunks:
        logging.info("Dask detected; using chunks %s", chunks)
    else:
        logging.info("Dask not detected; using eager loading")

    grid = load_grid(args.grid_file)
    xc = grid["XC"]
    yc = grid["YC"]
    rc = grid["RC"]
    hfac = grid["HFacC"]

    oras_t = load_oras(args.oras_t_file, "votemper", chunks)
    oras_s = load_oras(args.oras_s_file, "vosaline", chunks)

    if args.time_index < 0 or args.time_index >= oras_t.sizes["time_counter"]:
        raise IndexError("time-index out of range for ORAS5 dataset")

    time_value = oras_t["time_counter"].isel(time_counter=args.time_index).values
    time_ref = str(time_value)

    if "nav_lon" not in oras_t.coords or "nav_lat" not in oras_t.coords:
        raise KeyError("nav_lon/nav_lat not found in ORAS5 dataset coordinates")

    nav_lon = oras_t["nav_lon"]
    nav_lat = oras_t["nav_lat"]

    regrid_bilinear, regrid_nearest, periodic = build_regridder(
        nav_lon, nav_lat, xc, yc, args.weights
    )
    logging.info("Regridder built (periodic=%s)", periodic)
    logging.info(
        "ORAS5 dims: time=%s depth=%s y=%s x=%s",
        oras_t.sizes["time_counter"],
        oras_t.sizes["deptht"],
        oras_t.sizes["y"],
        oras_t.sizes["x"],
    )
    logging.info(
        "MITgcm dims: Z=%s Y=%s X=%s",
        hfac.sizes["Z"],
        hfac.sizes["Y"],
        hfac.sizes["X"],
    )
    logging.info(
        "ORAS5 lon range: [%.3f, %.3f] lat range: [%.3f, %.3f]",
        float(np.nanmin(nav_lon.values)),
        float(np.nanmax(nav_lon.values)),
        float(np.nanmin(nav_lat.values)),
        float(np.nanmax(nav_lat.values)),
    )
    logging.info(
        "MITgcm XC range: [%.3f, %.3f] YC range: [%.3f, %.3f]",
        float(np.nanmin(xc.values)),
        float(np.nanmax(xc.values)),
        float(np.nanmin(yc.values)),
        float(np.nanmax(yc.values)),
    )

    t_sel = oras_t.isel(time_counter=args.time_index)
    s_sel = oras_s.isel(time_counter=args.time_index)

    log_stats("Temp before horizontal", t_sel)
    log_stats("Salt before horizontal", s_sel)

    t_regridded = []
    s_regridded = []
    for k in range(t_sel.sizes["deptht"]):
        t_slice = t_sel.isel(deptht=k)
        s_slice = s_sel.isel(deptht=k)
        t_regridded.append(regrid_horizontal(t_slice, regrid_bilinear, regrid_nearest))
        s_regridded.append(regrid_horizontal(s_slice, regrid_bilinear, regrid_nearest))

    t_regridded = xr.concat(t_regridded, dim="deptht")
    s_regridded = xr.concat(s_regridded, dim="deptht")

    log_stats("Temp after horizontal", t_regridded)
    log_stats("Salt after horizontal", s_regridded)

    z_oras = t_sel["deptht"].values.astype(np.float64)
    z_mit = np.abs(rc.values.astype(np.float64))

    t_vert = interp_vertical(t_regridded, z_oras, z_mit, hfac.values)
    s_vert = interp_vertical(s_regridded, z_oras, z_mit, hfac.values)

    ny, nx = yc.shape
    coords = {
        "Z": rc.values,
        "Y": np.arange(ny),
        "X": np.arange(nx),
        "YC": (("Y", "X"), yc.values),
        "XC": (("Y", "X"), xc.values),
    }
    t_vert = t_vert.assign_coords(coords)
    s_vert = s_vert.assign_coords(coords)

    log_stats("Temp after vertical", t_vert)
    log_stats("Salt after vertical", s_vert)

    t_masked = apply_hfac_mask(t_vert, hfac)
    s_masked = apply_hfac_mask(s_vert, hfac)

    log_stats("Temp after HFacC", t_masked)
    log_stats("Salt after HFacC", s_masked)

    if t_masked.shape != hfac.shape:
        raise ValueError("Final temperature shape mismatch with HFacC")
    if s_masked.shape != hfac.shape:
        raise ValueError("Final salinity shape mismatch with HFacC")

    history = " ".join(sys.argv)
    write_netcdf(t_masked, s_masked, xc, yc, rc, args.out_nc, history, time_ref)

    write_mitgcm_bin(t_masked, hfac, args.out_bin_theta)
    write_mitgcm_bin(s_masked, hfac, args.out_bin_salt)


if __name__ == "__main__":
    main()
