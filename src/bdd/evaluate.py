from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm

from .dataset import SarRgbFootprintDataset, make_loader, make_stratified_folds
from .model import MultiModalSARFTPRGB
from .metrics import binary_eval


def evaluate_cross_validation(cfg: dict):
    root_dir = cfg["root_dir"]
    checkpoint_dir = Path(cfg["checkpoint_dir"])
    folds = make_stratified_folds(root_dir, n_splits=int(cfg.get("num_folds", 5)), seed=int(cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for fold_i, (_, val_ids) in enumerate(folds, start=1):
        print(f"\n>>> Evaluating fold {fold_i}")

        model = MultiModalSARFTPRGB(
            embed_dim=int(cfg.get("embed_dim", 128)),
            use_sar=bool(cfg.get("use_sar", True)),
            use_rgb=bool(cfg.get("use_rgb", True)),
        ).to(device)
        model.load_state_dict(torch.load(checkpoint_dir / f"fold{fold_i}_best.pth", map_location=device), strict=False)
        model.eval()

        val_ds = SarRgbFootprintDataset(root_dir, ids=val_ids, cache=False, return_id=True)
        val_loader = make_loader(val_ds, int(cfg.get("batch_size", 32)), int(cfg.get("num_workers", 4)), shuffle=False)

        outs, gts = [], []
        with torch.no_grad():
            for (sar, rgb, ftp), labels, _ in tqdm(val_loader, desc=f"Fold {fold_i} [Val]"):
                sar, rgb, ftp = sar.to(device), rgb.to(device), ftp.to(device)
                logits = model(
                    ftp=ftp,
                    sar=sar if cfg.get("use_sar", True) else None,
                    rgb=rgb if cfg.get("use_rgb", True) else None,
                )
                outs.append(torch.sigmoid(logits).cpu().numpy())
                gts.append(labels.cpu().numpy())

        probs = np.concatenate(outs)
        labels = np.concatenate(gts).astype(int)
        m = binary_eval(probs, labels)
        rows.append({k: v for k, v in m.items() if k != "pred"})
        print(f"Fold {fold_i}: AUROC={m['auroc']:.3f}, Best-F1={m['best_f1']:.3f}, thr={m['selected_threshold']:.2f}, kappa={m['kappa']:.3f}")

    print("\n=== Cross-validation summary ===")
    for key in ["auroc", "f1", "best_f1", "best_threshold", "kappa"]:
        values = [r[key] for r in rows]
        print(f"{key}: {np.mean(values):.3f} ± {np.std(values):.3f}")
    return rows
