"""Print final test metrics from saved Stage-3 predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


RESULTS = Path(r"D:\ALZ\results")


def main() -> int:
    y_true_path = RESULTS / "y_true_test.npy"
    y_pred_path = RESULTS / "y_pred_test.npy"
    y_proba_path = RESULTS / "y_proba_test.npy"
    missing = [p for p in [y_true_path, y_pred_path, y_proba_path] if not p.exists()]
    if missing:
        raise SystemExit("Missing prediction files: " + ", ".join(str(p) for p in missing))

    y_true = np.load(y_true_path)
    y_pred = np.load(y_pred_path)
    y_proba = np.load(y_proba_path)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 50)
    print("RESULTS (Paper targets in parentheses)")
    print("=" * 50)
    print(f"Accuracy : {acc:.4f}  (target: 0.7321)")
    print(f"AUC      : {auc:.4f}  (target: 0.8625)")
    print(f"F1       : {f1:.4f}  (target: 0.6868)")
    print(f"Precision: {prec:.4f}  (target: 0.7361)")
    print(f"Recall   : {rec:.4f}  (target: 0.6615)")
    print(f"MCC      : {mcc:.4f}  (target: 0.5299)")
    print("\nConfusion Matrix (CN / MCI / AD):")
    print(cm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
