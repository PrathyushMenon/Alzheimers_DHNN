"""Train and evaluate a pilot multimodal 3-class classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


LABEL_NAMES = ["CN", "MCI", "AD"]


def load_features(feature_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    return np.load(feature_dir / f"{split}_features.npy"), np.load(feature_dir / f"{split}_labels.npy")


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None) -> dict[str, float | list[list[int]] | None]:
    out: dict[str, float | list[list[int]] | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }
    if y_proba is not None and y_proba.shape[1] == 3 and len(np.unique(y_true)) == 3:
        out["auc_macro_ovr"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    else:
        out["auc_macro_ovr"] = None
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", default=r"D:\ALZ\features\pilot")
    parser.add_argument("--out-dir", default=r"D:\ALZ\results\pilot_model")
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train = load_features(feature_dir, "train")
    x_val, y_val = load_features(feature_dir, "val")
    x_test, y_test = load_features(feature_dir, "test")
    n_pca = min(24, x_train.shape[0] - 1, x_train.shape[1])

    models = {
        "logreg_balanced": LogisticRegression(max_iter=5000, class_weight="balanced", C=0.25, random_state=42),
        "svc_rbf_balanced": SVC(C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=42),
        "random_forest_balanced": RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42, min_samples_leaf=2),
        "extra_trees_balanced": ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=42, min_samples_leaf=2),
    }

    rows = []
    fitted = {}
    for name, clf in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=n_pca, random_state=42)), ("clf", clf)])
        pipe.fit(x_train, y_train)
        pred = pipe.predict(x_val)
        proba = pipe.predict_proba(x_val) if hasattr(pipe, "predict_proba") else None
        m = metrics_dict(y_val, pred, proba)
        rows.append({"model": name, **{k: v for k, v in m.items() if k != "confusion_matrix"}})
        fitted[name] = pipe
        print(f"VAL {name}: acc={m['accuracy']:.4f}, bal_acc={m['balanced_accuracy']:.4f}, f1={m['f1_macro']:.4f}")

    val_df = pd.DataFrame(rows).sort_values(["balanced_accuracy", "f1_macro", "accuracy"], ascending=False)
    best_name = str(val_df.iloc[0]["model"])
    best = fitted[best_name]
    print(f"Selected model: {best_name}")

    # Refit selected model on train+val after model selection, then evaluate once on test.
    x_trainval = np.vstack([x_train, x_val])
    y_trainval = np.concatenate([y_train, y_val])
    final_model = best
    final_model.fit(x_trainval, y_trainval)
    y_pred = final_model.predict(x_test)
    y_proba = final_model.predict_proba(x_test) if hasattr(final_model, "predict_proba") else None
    test_metrics = metrics_dict(y_test, y_pred, y_proba)

    val_df.to_csv(out_dir / "validation_model_selection.tsv", sep="\t", index=False)
    np.save(out_dir / "y_true_test.npy", y_test)
    np.save(out_dir / "y_pred_test.npy", y_pred)
    if y_proba is not None:
        np.save(out_dir / "y_proba_test.npy", y_proba)
    pd.DataFrame({"label": [0, 1, 2], "diagnosis": LABEL_NAMES}).to_csv(out_dir / "label_map.tsv", sep="\t", index=False)
    (out_dir / "metrics.json").write_text(json.dumps({"selected_model": best_name, "test": test_metrics}, indent=2), encoding="utf-8")
    joblib.dump(final_model, out_dir / "best_pilot_model.joblib")

    # Also write to the canonical results filenames used by 05_evaluate.py.
    canonical = Path(r"D:\ALZ\results")
    np.save(canonical / "y_true_test.npy", y_test)
    np.save(canonical / "y_pred_test.npy", y_pred)
    if y_proba is not None:
        np.save(canonical / "y_proba_test.npy", y_proba)

    print(json.dumps({"selected_model": best_name, "test": test_metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
