

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def strip_damage_prefix(sample_id: str) -> str:
    """Remove D_/I_ prefix from a patch/sample id to recover the original building id."""
    sample_id = str(sample_id)
    if sample_id.startswith("D_") or sample_id.startswith("I_"):
        return sample_id[2:]
    return sample_id


def load_oof_predictions(oof_path: str | Path, threshold: float) -> pd.DataFrame:
    oof_path = Path(oof_path)
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF predictions not found: {oof_path}")

    oof = pd.read_csv(oof_path)
    required_cols = {"id", "prob_damage"}
    missing_cols = required_cols - set(oof.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in OOF CSV: {sorted(missing_cols)}")

    oof = oof.copy()
    oof["building_id"] = oof["id"].map(strip_damage_prefix)
    oof["prob_damage"] = oof["prob_damage"].astype(float)
    oof["pred_label"] = (oof["prob_damage"] >= float(threshold)).astype(int)
    oof["damage_status"] = np.where(oof["pred_label"] == 1, "damaged", "intact")

    keep_cols = [
        "building_id",
        "id",
        "prob_damage",
        "pred_label",
        "damage_status",
        "fold",
        "epoch",
        "best_epoch",
        "label",
    ]
    keep_cols = [col for col in keep_cols if col in oof.columns]
    return oof[keep_cols]


def build_oof_damage_map(
    oof_path: str | Path,
    buildings_path: str | Path,
    out_path: str | Path,
    id_column: str,
    threshold: float = 0.5,
    buildings_layer: str | None = None,
    output_layer: str = "oof_damage_map",
) -> Path:
    buildings_path = Path(buildings_path)
    out_path = Path(out_path)

    if not buildings_path.exists():
        raise FileNotFoundError(f"Building layer not found: {buildings_path}")

    buildings = gpd.read_file(buildings_path, layer=buildings_layer)
    if id_column not in buildings.columns:
        raise ValueError(
            f"ID column '{id_column}' not found in building layer. "
            f"Available columns: {list(buildings.columns)}"
        )

    oof = load_oof_predictions(oof_path, threshold=threshold)

    buildings = buildings.copy()
    buildings["building_id"] = buildings[id_column].astype(str)
    oof["building_id"] = oof["building_id"].astype(str)

    merged = buildings.merge(oof, on="building_id", how="left")

    merged["evaluated"] = merged["prob_damage"].notna()
    merged["pred_label"] = merged["pred_label"].fillna(-1).astype(int)
    merged["damage_status"] = merged["damage_status"].fillna("not_evaluated")

    merged["map_class"] = merged["damage_status"].map(
        {
            "damaged": 1,
            "intact": 0,
            "not_evaluated": -1,
        }
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_file(out_path, layer=output_layer, driver="GPKG")

    n_total = len(merged)
    n_evaluated = int(merged["evaluated"].sum())
    n_damaged = int((merged["damage_status"] == "damaged").sum())
    n_intact = int((merged["damage_status"] == "intact").sum())
    n_not_evaluated = int((merged["damage_status"] == "not_evaluated").sum())

    print(f"Saved OOF damage map: {out_path}")
    print(f"Layer: {output_layer}")
    print(f"Total buildings: {n_total}")
    print(f"Evaluated buildings: {n_evaluated}")
    print(f"Predicted damaged: {n_damaged}")
    print(f"Predicted intact: {n_intact}")
    print(f"Not evaluated: {n_not_evaluated}")

    return out_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a GeoPackage damage map from OOF building-level predictions."
    )
    parser.add_argument(
        "--oof",
        required=True,
        type=str,
        help="Path to oof_predictions.csv generated after cross-validation.",
    )
    parser.add_argument(
        "--buildings",
        required=True,
        type=str,
        help="Path to the original building GeoPackage used for patch generation.",
    )
    parser.add_argument(
        "--buildings-layer",
        default=None,
        type=str,
        help="Optional layer name inside the building GeoPackage.",
    )
    parser.add_argument(
        "--id-column",
        required=True,
        type=str,
        help="ID column in the building layer matching the patch ids after removing D_/I_ prefix.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=str,
        help="Output GeoPackage path.",
    )
    parser.add_argument(
        "--output-layer",
        default="oof_damage_map",
        type=str,
        help="Layer name to write in the output GeoPackage.",
    )
    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float,
        help="Probability threshold used to convert prob_damage into damaged/intact.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_oof_damage_map(
        oof_path=args.oof,
        buildings_path=args.buildings,
        buildings_layer=args.buildings_layer,
        id_column=args.id_column,
        out_path=args.out,
        output_layer=args.output_layer,
        threshold=args.threshold,
    )

"""
Example launch; --buildings must be the original building GeoPackage used to create the patches.
Important: do not leave spaces after the trailing backslashes.

python -m src.bdd.build_oof_map \
  --oof /home/silvia/Desktop/GIGI/ASI_WGD_2026_Myanmar/BDD/results_BDD_no_cls_whts/oof_predictions.csv \
  --buildings /home/silvia/Desktop/GIGI/ASI_WGD_2026_Myanmar/BDD/data/REF/OSM_Polygons_with_dmg_clipped.gpkg \
  --id-column osm_id \
  --out /home/silvia/Desktop/GIGI/ASI_WGD_2026_Myanmar/BDD/results_BDD_no_cls_whts/oof_damage_map.gpkg \
  --threshold 0.5
"""