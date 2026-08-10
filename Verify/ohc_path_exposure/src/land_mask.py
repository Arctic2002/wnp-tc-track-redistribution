"""Self-contained GSHHG raster mask used only for source-point land flags."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def build_land_mask(shapefile: Path, domain: dict, resolution_deg: float) -> dict:
    """Rasterize GSHHG level-1 land polygons on a fixed lon/lat grid."""
    import geopandas as gpd
    from affine import Affine
    from rasterio.features import rasterize

    path = Path(shapefile)
    if not path.is_file():
        raise FileNotFoundError(path)
    west = float(domain["west"])
    east = float(domain["east"])
    south = float(domain["south"])
    north = float(domain["north"])
    resolution = float(resolution_deg)
    if resolution <= 0.0:
        raise ValueError("land-mask resolution must be positive")
    nx = int(np.ceil((east - west) / resolution))
    ny = int(np.ceil((north - south) / resolution))
    transform = Affine.translation(west, north) * Affine.scale(resolution, -resolution)
    land = gpd.read_file(path)
    if land.crs is not None and not land.crs.is_geographic:
        land = land.to_crs("EPSG:4326")
    mask = rasterize(
        ((geometry, 1) for geometry in land.geometry if geometry is not None and not geometry.is_empty),
        out_shape=(ny, nx),
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    return {
        "mask": mask,
        "west": west,
        "east": east,
        "south": south,
        "north": north,
        "resolution": resolution,
    }


def classify_points(latitude: np.ndarray, longitude: np.ndarray, mask_info: dict) -> np.ndarray:
    """Return 1 for land, 0 for ocean and -1 for points outside the mask."""
    lat = np.asarray(latitude, dtype=float)
    lon = np.asarray(longitude, dtype=float) % 360.0
    if lat.shape != lon.shape:
        raise ValueError("latitude and longitude must have identical shape")
    west = float(mask_info["west"])
    east = float(mask_info["east"])
    south = float(mask_info["south"])
    north = float(mask_info["north"])
    resolution = float(mask_info["resolution"])
    mask = np.asarray(mask_info["mask"], dtype=np.uint8)
    inside = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (lon >= west)
        & (lon <= east)
        & (lat >= south)
        & (lat <= north)
    )
    result = np.full(lat.shape, -1, dtype=np.int8)
    if not inside.any():
        return result
    column = np.floor((lon[inside] - west) / resolution).astype(int)
    row = np.floor((north - lat[inside]) / resolution).astype(int)
    column = np.clip(column, 0, mask.shape[1] - 1)
    row = np.clip(row, 0, mask.shape[0] - 1)
    result[inside] = mask[row, column].astype(np.int8)
    return result
