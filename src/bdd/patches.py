import csv
import os
import random
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from shapely.geometry import box
from tqdm import tqdm


def compute_sar_window(ds, cx: float, cy: float, patch_px: int):
    row, col = ds.index(cx, cy)
    half = patch_px // 2
    row0, col0 = row - half, col - half
    if row0 < 0 or col0 < 0 or row0 + patch_px > ds.height or col0 + patch_px > ds.width:
        return None
    return Window(col0, row0, patch_px, patch_px)


def read_hh_hv_stack(hh_ds, hv_ds, window):
    hh = hh_ds.read(1, window=window).astype(np.float32)
    same_grid = (
        hh_ds.transform.a == hv_ds.transform.a
        and hh_ds.transform.e == hv_ds.transform.e
        and hh_ds.crs == hv_ds.crs
    )
    if same_grid:
        hv = hv_ds.read(1, window=window).astype(np.float32)
    else:
        dst = np.zeros_like(hh, dtype=np.float32)
        win_tr = rasterio.windows.transform(window, hh_ds.transform)
        reproject(
            source=rasterio.band(hv_ds, 1),
            destination=dst,
            src_transform=hv_ds.transform,
            src_crs=hv_ds.crs,
            dst_transform=win_tr,
            dst_crs=hh_ds.crs,
            resampling=Resampling.bilinear,
        )
        hv = dst
    return np.stack([hh, hv], axis=0)


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


def physical_patch_size_m(ds, patch_px: int):
    return patch_px * abs(ds.transform.a), patch_px * abs(ds.transform.e)


def compute_window_for_same_area(ds, center_x, center_y, width_m, height_m):
    px_w = max(1, int(round(width_m / abs(ds.transform.a))))
    px_h = max(1, int(round(height_m / abs(ds.transform.e))))
    row, col = ds.index(center_x, center_y)
    row0, col0 = row - px_h // 2, col - px_w // 2
    if row0 < 0 or col0 < 0 or row0 + px_h > ds.height or col0 + px_w > ds.width:
        return None
    return Window(col0, row0, px_w, px_h)


def create_dataset_patches(cfg: dict) -> Path:
    random.seed(int(cfg.get("seed", 42)))

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    patch_px_sar = int(cfg.get("patch_px_sar", 40))
    ratio = cfg.get("ratio_intact_to_damaged", 20)
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

        print(f"Buildings in SAR tile -> damaged={len(damaged_idx)} | intact={len(intact_idx)}")
        patch_w_m, patch_h_m = physical_patch_size_m(hh_ds, patch_px_sar)

        rows = []
        for idx in tqdm(damaged_idx + intact_idx, desc="Writing patches", unit="building"):
            geom = buildings.geometry[idx]
            if geom.is_empty:
                continue

            centroid = geom.centroid
            cx, cy = centroid.x, centroid.y

            sar_win = compute_sar_window(hh_ds, cx, cy, patch_px_sar)
            if sar_win is None:
                continue

            stack = read_hh_hv_stack(hh_ds, hv_ds, sar_win)
            win_tr = rasterio.windows.transform(sar_win, hh_ds.transform)
            mask = rasterize_footprint(geom, out_shape=(stack.shape[1], stack.shape[2]), transform=win_tr)

            cx_rgb, cy_rgb = cx, cy
            if rgb_ds.crs != hh_ds.crs:
                pt = gpd.GeoSeries([centroid], crs=hh_ds.crs).to_crs(rgb_ds.crs).geometry.iloc[0]
                cx_rgb, cy_rgb = pt.x, pt.y

            rgb_win = compute_window_for_same_area(rgb_ds, cx_rgb, cy_rgb, patch_w_m, patch_h_m)
            if rgb_win is None:
                continue

            rgb_patch = rgb_ds.read([1, 2, 3], window=rgb_win)

            label = int(buildings.loc[idx, col_label])
            prefix = "D" if label == 1 else "I"
            sample_id = f"{prefix}_{buildings.loc[idx, col_id]}"

            save_geotiff(out_dir / f"{sample_id}_SAR.tif", stack, hh_ds.crs, win_tr, dtype=np.float32)
            save_geotiff(out_dir / f"{sample_id}_SARftp.tif", mask.astype(np.uint8), hh_ds.crs, win_tr, dtype=np.uint8)
            save_geotiff(
                out_dir / f"{sample_id}_RGB.tif",
                rgb_patch,
                rgb_ds.crs,
                rasterio.windows.transform(rgb_win, rgb_ds.transform),
                dtype=rgb_patch.dtype,
            )

            minx, miny, maxx, maxy = geom.bounds
            rows.append([sample_id, label, minx, miny, maxx, maxy])

    csv_path = out_dir / "patch_list.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label", "minx", "miny", "maxx", "maxy"])
        writer.writerows(rows)

    print(f"Completed. Written patches: {len(rows)}")
    print(f"Manifest: {csv_path}")
    return csv_path
