from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
import argparse
from sklearn import metrics
from torch.utils.data import WeightedRandomSampler
from tqdm.auto import tqdm

from src.bdd.dataset import SarRgbFootprintDataset, make_loader, make_stratified_folds
from src.bdd.model import MultiModalBuildingGuidedFusion
from src.bdd.utils import set_seed, initialize_model


def train_cross_validation(cfg: dict):
    seed = int(cfg["cross_validation"].get("seed", 42))
    set_seed(seed)

    root_dir = cfg["dataset"]["root_dir"]
    checkpoint_dir = Path(cfg["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    num_folds = int(cfg["cross_validation"].get("num_folds", 5))
    folds = make_stratified_folds(root_dir, n_splits=num_folds, seed=seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    oof_rows = []

    for fold_i, (train_ids, val_ids) in enumerate(folds, start=1):
        print(f"\n=== FOLD {fold_i}/{num_folds} ===")

        train_ds = SarRgbFootprintDataset(
            root_dir,
            ids=train_ids,
            cache=cfg["dataset"].get("cache_dataset", False),
            return_id=True,
        )
        val_ds = SarRgbFootprintDataset(root_dir, ids=val_ids, cache=False, return_id=True)

        sampler = None
        counts = np.bincount(np.array(train_ds.labels, dtype=int))
        if cfg["loss"].get("use_class_weights", False):
            weights = 1.0 / counts
            sample_weights = weights[np.array(train_ds.labels, dtype=int)]
            sampler = WeightedRandomSampler(
                sample_weights,
                len(sample_weights),
                generator=torch.Generator().manual_seed(seed),
            )

        train_loader = make_loader(
            train_ds,
            batch_size=int(cfg["training"].get("batch_size", 32)),
            num_workers=int(cfg["training"].get("num_workers", 4)),
            shuffle=True,
            sampler=sampler,
            drop_last=True,
        )
        val_loader = make_loader(
            val_ds,
            batch_size=int(cfg["training"].get("batch_size", 32)),
            num_workers=int(cfg["training"].get("num_workers", 4)),
            shuffle=False,
        )

        model = MultiModalBuildingGuidedFusion(
            embed_dim=int(cfg["model"].get("embed_dim", 128)),
            use_sar=bool(cfg["model"].get("use_sar", True)),
            use_rgb=bool(cfg["model"].get("use_rgb", True)),
        )

        model.apply(initialize_model)

        if cfg["pretraining"].get("use_sar_pretrained", False):
            ckpt = torch.load(cfg["pretraining"]["sar_pretrain_path"], map_location="cpu")
            model.sar_trunk.load_state_dict(ckpt, strict=False)

        model.to(device)

        if cfg["loss"].get("use_class_weights", False):
            neg, pos = counts
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))
        else:
            criterion = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["training"].get("learning_rate", 1e-4)),
        )
        epochs = int(cfg["training"].get("epochs", 30))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(cfg["training"].get("learning_rate", 1e-4)),
            epochs=epochs,
            steps_per_epoch=len(train_loader),
        )
        scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

        best_auc, best_epoch, patience_ctr = 0.0, -1, 0
        best_fold_oof = None
        patience = int(cfg["training"]["early_stopping"].get("patience", 5))

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0

            for (sar, rgb, ftp), labels, _ in tqdm(train_loader, desc=f"[Fold{fold_i}] Ep{epoch} Train", leave=False):
                labels = labels.float().to(device)
                sar, rgb, ftp = sar.to(device), rgb.to(device), ftp.to(device)

                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(
                        ftp=ftp,
                        sar=sar if cfg["model"].get("use_sar", True) else None,
                        rgb=rgb if cfg["model"].get("use_rgb", True) else None,
                    )
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                train_loss += loss.item()

            model.eval()
            outs, gts, sample_ids = [], [], []
            with torch.no_grad():
                for (sar, rgb, ftp), labels, ids in tqdm(val_loader, desc=f"[Fold{fold_i}] Ep{epoch} Val", leave=False):
                    sar, rgb, ftp = sar.to(device), rgb.to(device), ftp.to(device)
                    with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                        logits = model(
                            ftp=ftp,
                            sar=sar if cfg["model"].get("use_sar", True) else None,
                            rgb=rgb if cfg["model"].get("use_rgb", True) else None,
                        )
                    outs.append(torch.sigmoid(logits).cpu().numpy())
                    gts.append(labels.cpu().numpy())
                    sample_ids.extend(list(ids))

            outs = np.concatenate(outs)
            gts = np.concatenate(gts).astype(int)
            auc = metrics.roc_auc_score(gts, outs)
            ap = metrics.average_precision_score(gts, outs)

            fpr, tpr, thresholds = metrics.roc_curve(gts, outs)
            best_idx = np.argmax(tpr - fpr)
            best_threshold = float(thresholds[best_idx])

            preds_best = (outs >= best_threshold).astype(int)
            f1 = metrics.f1_score(gts, preds_best)
            precision = metrics.precision_score(gts, preds_best, zero_division=0)
            recall = metrics.recall_score(gts, preds_best, zero_division=0)

            fold_oof = pd.DataFrame(
                {
                    "id": sample_ids,
                    "label": gts,
                    "prob_damage": outs,
                    "pred_label": (
                        outs >= float(cfg["evaluation"].get("decision_threshold", 0.5))
                    ).astype(int),
                    "fold": fold_i,
                    "epoch": epoch,
                    "best_threshold": best_threshold,
                    "auprc": ap,
                }
            )
            print(
                f"Fold{fold_i} Ep{epoch} | "
                f"TrainLoss {train_loss / len(train_loader):.4f} | "
                f"AUROC {auc:.4f} | "
                f"AUPRC {ap:.4f} | "
                f"F1 {f1:.4f} | "
                f"Thr {best_threshold:.3f}"
            )

            if auc > best_auc:
                best_auc, best_epoch = auc, epoch
                torch.save(model.state_dict(), checkpoint_dir / f"fold{fold_i}_best.pth")
                best_fold_oof = fold_oof.copy()
                best_fold_oof["best_epoch"] = epoch
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(f"[Fold{fold_i}] early stop at epoch {epoch}")
                    break


        results.append(
            {
                "fold": fold_i,
                "best_epoch": best_epoch,
                "best_auroc": best_auc,
            }
        )
        if best_fold_oof is not None:
            oof_rows.append(best_fold_oof)
        torch.cuda.empty_cache()

    print("\n=== CV RESULTS ===")
    for row in results:
        print(f"Fold {row['fold']}: best epoch={row['best_epoch']}, best AUROC={row['best_auroc']:.4f}")
    print(f"Mean AUROC = {np.mean([r['best_auroc'] for r in results]):.4f} ± {np.std([r['best_auroc'] for r in results]):.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(checkpoint_dir / "cv_results.csv", index=False)

    if len(oof_rows) > 0:
        oof_df = pd.concat(oof_rows, ignore_index=True)
        manifest_path = Path(root_dir) / "patch_list.csv"
        if manifest_path.exists():
            manifest = pd.read_csv(manifest_path)
            bounds_cols = ["id", "minx", "miny", "maxx", "maxy"]
            available_cols = [col for col in bounds_cols if col in manifest.columns]
            if "id" in available_cols:
                oof_df = oof_df.merge(manifest[available_cols], on="id", how="left")
        oof_df.to_csv(checkpoint_dir / "oof_predictions.csv", index=False)
        print(f"Saved OOF predictions: {checkpoint_dir / 'oof_predictions.csv'}")
        print(f"Saved CV results: {checkpoint_dir / 'cv_results.csv'}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Train FGCA model with stratified cross-validation.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to the YAML training configuration file.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    train_cross_validation(cfg)
