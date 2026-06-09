import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score


def compute_pr_metrics(probs: np.ndarray, labels: np.ndarray, thr: float = 0.5) -> dict:
    y_pred = (probs >= thr).astype(int)
    precision = precision_score(labels, y_pred, zero_division=0)
    recall = recall_score(labels, y_pred, zero_division=0)
    f1 = f1_score(labels, y_pred, zero_division=0)

    prec_curve, rec_curve, thr_curve = metrics.precision_recall_curve(labels, probs)
    f1_curve = 2 * prec_curve * rec_curve / (prec_curve + rec_curve + 1e-8)
    best_idx = int(np.argmax(f1_curve))
    best_thr = thr_curve[best_idx] if best_idx < len(thr_curve) else 1.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "best_threshold": float(best_thr),
        "best_precision": float(prec_curve[best_idx]),
        "best_recall": float(rec_curve[best_idx]),
        "best_f1": float(f1_curve[best_idx]),
    }


def binary_eval(probs: np.ndarray, labels: np.ndarray, thr: float | None = None) -> dict:
    auc = metrics.roc_auc_score(labels, probs)
    pr = compute_pr_metrics(probs, labels, thr=0.5)
    selected_thr = pr["best_threshold"] if thr is None else thr
    pred = (probs >= selected_thr).astype(int)
    return {
        "auroc": float(auc),
        **pr,
        "selected_threshold": float(selected_thr),
        "kappa": float(cohen_kappa_score(labels, pred)),
        "pred": pred,
    }
