from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import WeightedRandomSampler
from tqdm.auto import tqdm

from .dataset import SarRgbFootprintDataset, make_loader, make_stratified_folds
from .model import MultiModalSARFTPRGB
from .utils import set_seed, initialize_model


def train_cross_validation(cfg: dict):
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    root_dir = cfg["root_dir"]
    checkpoint_dir = Path(cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    num_folds = int(cfg.get("num_folds", 5))
    folds = make_stratified_folds(root_dir, n_splits=num_folds, seed=seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for fold_i, (train_ids, val_ids) in enumerate(folds, start=1):
        print(f"\n=== FOLD {fold_i}/{num_folds} ===")

        train_ds = SarRgbFootprintDataset(root_dir, ids=train_ids, cache=cfg.get("cache_dataset", False), return_id=True)
        val_ds = SarRgbFootprintDataset(root_dir, ids=val_ids, cache=False, return_id=True)

        sampler = None
        counts = np.bincount(np.array(train_ds.labels, dtype=int))
        if cfg.get("use_class_weights", False):
            weights = 1.0 / counts
            sample_weights = weights[np.array(train_ds.labels, dtype=int)]
            sampler = WeightedRandomSampler(
                sample_weights,
                len(sample_weights),
                generator=torch.Generator().manual_seed(seed),
            )

        train_loader = make_loader(
            train_ds,
            batch_size=int(cfg.get("batch_size", 32)),
            num_workers=int(cfg.get("num_workers", 4)),
            shuffle=True,
            sampler=sampler,
            drop_last=True,
        )
        val_loader = make_loader(
            val_ds,
            batch_size=int(cfg.get("batch_size", 32)),
            num_workers=int(cfg.get("num_workers", 4)),
            shuffle=False,
        )

        model = MultiModalSARFTPRGB(
            embed_dim=int(cfg.get("embed_dim", 128)),
            use_sar=bool(cfg.get("use_sar", True)),
            use_rgb=bool(cfg.get("use_rgb", True)),
        )
        model.apply(initialize_model)

        if cfg.get("use_sar_pretrained", False):
            ckpt = torch.load(cfg["sar_pretrain_path"], map_location="cpu")
            model.branch_sar.load_state_dict(ckpt, strict=False)

        model.to(device)

        if cfg.get("use_class_weights", False):
            neg, pos = counts
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))
        else:
            criterion = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 1e-4)))
        epochs = int(cfg.get("epochs", 30))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

        best_auc, best_epoch, patience_ctr = 0.0, -1, 0
        patience = int(cfg.get("patience", 5))

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
                        sar=sar if cfg.get("use_sar", True) else None,
                        rgb=rgb if cfg.get("use_rgb", True) else None,
                    )
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()

            model.eval()
            outs, gts = [], []
            with torch.no_grad():
                for (sar, rgb, ftp), labels, _ in tqdm(val_loader, desc=f"[Fold{fold_i}] Ep{epoch} Val", leave=False):
                    sar, rgb, ftp = sar.to(device), rgb.to(device), ftp.to(device)
                    with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                        logits = model(
                            ftp=ftp,
                            sar=sar if cfg.get("use_sar", True) else None,
                            rgb=rgb if cfg.get("use_rgb", True) else None,
                        )
                    outs.append(torch.sigmoid(logits).cpu().numpy())
                    gts.append(labels.cpu().numpy())

            outs = np.concatenate(outs)
            gts = np.concatenate(gts).astype(int)
            auc = metrics.roc_auc_score(gts, outs)
            print(f"Fold{fold_i} Ep{epoch} | TrainLoss {train_loss / len(train_loader):.4f} | ValAUROC {auc:.4f}")

            if auc > best_auc:
                best_auc, best_epoch = auc, epoch
                torch.save(model.state_dict(), checkpoint_dir / f"fold{fold_i}_best.pth")
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(f"[Fold{fold_i}] early stop at epoch {epoch}")
                    break

            scheduler.step()

        results.append({"fold": fold_i, "best_epoch": best_epoch, "best_auroc": best_auc})
        torch.cuda.empty_cache()

    print("\n=== CV RESULTS ===")
    for row in results:
        print(f"Fold {row['fold']}: best epoch={row['best_epoch']}, best AUROC={row['best_auroc']:.4f}")
    print(f"Mean AUROC = {np.mean([r['best_auroc'] for r in results]):.4f} ± {np.std([r['best_auroc'] for r in results]):.4f}")
    return results
