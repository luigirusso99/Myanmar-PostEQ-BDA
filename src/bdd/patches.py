import csv
import os
import random
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import box
from tqdm import tqdm


def rasterize_footprint(geom, out_shape, transform):
    return rasterize([(geom, 1)], out_shape=out_shape, transform=transform, fill=0, dtype="uint8")


def save_geotiff(path, array, crs, transform, dtype=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim == 2:
        count, height, width = 1, array.shape[0], array.shape[1]
    elif array.ndim == 3:
        count, height, width = array.shape
    else:
        raise ValueError("array must be 2D or 3D")

    meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "crs": crs,
        "transform": transform,
        "dtype": dtype or array.dtype,
    }
    with rasterio.open(path, "w", **meta) as dst:
        if count == 1:
            dst.write(array, 1)
        else:
            dst.write(array)


def choose_common_resolution_m(hh_ds, rgb_ds, cfg: dict) -> float:
    """Choose the output ground sampling distance for the common patch grid."""
    requested = cfg.get("common_resolution_m", None)
    if requested is not None:
        return float(requested)

    sar_res = min(abs(hh_ds.transform.a), abs(hh_ds.transform.e))
    rgb_res = min(abs(rgb_ds.transform.a), abs(rgb_ds.transform.e))
    strategy = cfg.get("common_resolution_strategy", "finest")

    if strategy == "finest":
        return min(sar_res, rgb_res)
    if strategy == "coarsest":
        return max(sar_res, rgb_res)
    if strategy == "sar":
        return sar_res
    if strategy == "rgb":
        return rgb_res

    raise ValueError(
        "common_resolution_strategy must be one of: finest, coarsest, sar, rgb"
    )


def choose_common_patch_size_px(cfg: dict, resolution_m: float) -> int:
    """Choose the patch size on the common grid."""
    if cfg.get("common_patch_px", None) is not None:
        patch_px = int(cfg["common_patch_px"])
    elif cfg.get("common_patch_size_m", None) is not None:
        patch_px = int(round(float(cfg["common_patch_size_m"]) / resolution_m))
    else:
        patch_px = int(cfg.get("patch_px_sar", 40))

    if patch_px <= 0:
        raise ValueError("common patch size must be positive")

    if patch_px % 2 == 0:
        patch_px += 1

    return patch_px


def compute_common_patch_transform(center_x: float, center_y: float, patch_px: int, resolution_m: float):
    """Build an output transform for a square patch centered on the building centroid."""
    half_size_m = patch_px * resolution_m / 2.0
    west = center_x - half_size_m
    north = center_y + half_size_m
    return from_origin(west, north, resolution_m, resolution_m)


def patch_bounds_from_transform(transform, patch_px: int):
    west = transform.c
    north = transform.f
    east = west + patch_px * transform.a
    south = north + patch_px * transform.e
    return min(west, east), min(south, north), max(west, east), max(south, north)


def dataset_covers_bounds(ds, bounds) -> bool:
    minx, miny, maxx, maxy = bounds
    return (
        minx >= ds.bounds.left
        and maxx <= ds.bounds.right
        and miny >= ds.bounds.bottom
        and maxy <= ds.bounds.top
    )


def read_dataset_on_common_grid(ds, band_indexes, dst_crs, dst_transform, patch_px: int, resampling=Resampling.bilinear):
    """Read and resample selected bands from any raster onto the common patch grid."""
    dst = np.zeros((len(band_indexes), patch_px, patch_px), dtype=np.float32)
    for out_idx, band_idx in enumerate(band_indexes):
        reproject(
            source=rasterio.band(ds, band_idx),
            destination=dst[out_idx],
            src_transform=ds.transform,
            src_crs=ds.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
        )
    return dst


def compute_valid_sar_fraction(stack: np.ndarray) -> float:
    """Compute the fraction of pixels with valid SAR information in both channels."""
    valid = np.isfinite(stack).all(axis=0)
    valid &= np.any(stack != 0, axis=0)
    return float(valid.mean())


def create_dataset_patches(cfg: dict) -> Path:
    random.seed(int(cfg.get("seed", 42)))

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    patch_px_sar = int(cfg.get("patch_px_sar", 40))
    ratio = cfg.get("ratio_intact_to_damaged", 20)
    min_sar_valid_fraction = float(cfg.get("min_sar_valid_fraction", 0.90))

    osm_layer = cfg.get("osm_layer", None)
    col_id = cfg.get("columns", {}).get("id", "osm_id")
    col_label = cfg.get("columns", {}).get("label", "damagedid")

    with rasterio.open(cfg["sar_hh_path"]) as hh_ds, rasterio.open(cfg["sar_hv_path"]) as hv_ds, rasterio.open(cfg["rgb_path"]) as rgb_ds:
        buildings = gpd.read_file(cfg["osm_path"], layer=osm_layer)
        if buildings.crs != hh_ds.crs:
            buildings = buildings.to_crs(hh_ds.crs)

        sar_fp = box(*hh_ds.bounds)
        buildings = buildings[buildings.intersects(sar_fp)].copy()

        if col_id not in buildings.columns:
            raise ValueError(f"Missing ID column: {col_id}")
        if col_label not in buildings.columns:
            raise ValueError(f"Missing label column: {col_label}")

        damaged_idx = buildings.index[buildings[col_label] == 1].tolist()
        intact_idx = buildings.index[buildings[col_label] == 0].tolist()

        if ratio is not None and int(ratio) > 0:
            random.shuffle(intact_idx)
            intact_idx = intact_idx[: len(damaged_idx) * int(ratio)]

        common_resolution_m = choose_common_resolution_m(hh_ds, rgb_ds, cfg)
        common_patch_px = choose_common_patch_size_px(cfg, common_resolution_m)
        print(f"Buildings in SAR tile -> damaged={len(damaged_idx)} | intact={len(intact_idx)}")
        print(
            "Common grid -> "
            f"resolution={common_resolution_m:.3f} m | patch={common_patch_px}x{common_patch_px} px | "
            f"size={common_patch_px * common_resolution_m:.2f} m"
        )

        rows = []
        for idx in tqdm(damaged_idx + intact_idx, desc="Writing patches", unit="building"):
            geom = buildings.geometry[idx]
            if geom.is_empty:
                continue

            centroid = geom.centroid
            cx, cy = centroid.x, centroid.y

            common_transform = compute_common_patch_transform(
                cx, cy, common_patch_px, common_resolution_m
            )
            common_bounds = patch_bounds_from_transform(common_transform, common_patch_px)
            if not dataset_covers_bounds(hh_ds, common_bounds):
                continue

            if rgb_ds.crs != hh_ds.crs:
                patch_geom = gpd.GeoSeries(
                    [box(*common_bounds)], crs=hh_ds.crs
                ).to_crs(rgb_ds.crs).geometry.iloc[0]
                rgb_bounds = patch_geom.bounds
            else:
                rgb_bounds = common_bounds

            if not dataset_covers_bounds(rgb_ds, rgb_bounds):
                continue

            stack = np.concatenate(
                [
                    read_dataset_on_common_grid(
                        hh_ds,
                        [1],
                        hh_ds.crs,
                        common_transform,
                        common_patch_px,
                        resampling=Resampling.bilinear,
                    ),
                    read_dataset_on_common_grid(
                        hv_ds,
                        [1],
                        hh_ds.crs,
                        common_transform,
                        common_patch_px,
                        resampling=Resampling.bilinear,
                    ),
                ],
                axis=0,
            )
            sar_valid_fraction = compute_valid_sar_fraction(stack)
            if sar_valid_fraction < min_sar_valid_fraction:
                continue

            mask = rasterize_footprint(
                geom,
                out_shape=(common_patch_px, common_patch_px),
                transform=common_transform,
            )
            rgb_patch = read_dataset_on_common_grid(
                rgb_ds,
                [1, 2, 3],
                hh_ds.crs,
                common_transform,
                common_patch_px,
                resampling=Resampling.bilinear,
            )

            label = int(buildings.loc[idx, col_label])
            prefix = "D" if label == 1 else "I"
            sample_id = f"{prefix}_{buildings.loc[idx, col_id]}"

            save_geotiff(
                out_dir / f"{sample_id}_SAR.tif",
                stack,
                hh_ds.crs,
                common_transform,
                dtype=np.float32,
            )
            save_geotiff(
                out_dir / f"{sample_id}_SARftp.tif",
                mask.astype(np.uint8),
                hh_ds.crs,
                common_transform,
                dtype=np.uint8,
            )
            save_geotiff(
                out_dir / f"{sample_id}_RGB.tif",
                rgb_patch,
                hh_ds.crs,
                common_transform,
                dtype=np.float32,
            )

            minx, miny, maxx, maxy = common_bounds
            rows.append([sample_id, label, minx, miny, maxx, maxy])

    csv_path = out_dir / "patch_list.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label", "minx", "miny", "maxx", "maxy"])
        writer.writerows(rows)

    print(f"Completed. Written patches: {len(rows)}")
    print(f"Manifest: {csv_path}")
    return csv_path
