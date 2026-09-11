#!/usr/bin/env python3
"""
Load ALBEF-extracted features and run TabPFN if available, otherwise fallback to RandomForest.
Outputs metrics and saves predictions to the features directory.
"""
import sys
import numpy as np
import os
from sklearn.metrics import accuracy_score, roc_auc_score

def load_npz(path):
    data = np.load(path)
    # support many npz layouts
    if 'aligned' in data:
        X = data['aligned']
    elif 'fusioned' in data:
        X = data['fusioned']
    else:
        # try first array
        X = data[list(data.files)[0]]
    return X

def main():
    if len(sys.argv) < 3:
        print('Usage: run_tabpfn.py <features_dir> <label_npz>')
        sys.exit(1)
    feat_dir = sys.argv[1]
    label_npz = sys.argv[2]

    X_train = load_npz(os.path.join(feat_dir, 'tr_feat_fusioned.npz'))['fusioned'] if os.path.exists(os.path.join(feat_dir, 'tr_feat_fusioned.npz')) else load_npz(os.path.join(feat_dir, 'tr_feat_aligned.npz'))
    y_train = np.load(os.path.join(feat_dir, 'tr_label.npz'))['label']
    X_val = load_npz(os.path.join(feat_dir, 'val_feat_fusioned.npz'))['fusioned'] if os.path.exists(os.path.join(feat_dir, 'val_feat_fusioned.npz')) else load_npz(os.path.join(feat_dir, 'val_feat_aligned.npz'))
    y_val = np.load(os.path.join(feat_dir, 'val_label.npz'))['label']
    X_test = load_npz(os.path.join(feat_dir, 'test_feat_fusioned.npz'))['fusioned'] if os.path.exists(os.path.join(feat_dir, 'test_feat_fusioned.npz')) else load_npz(os.path.join(feat_dir, 'test_feat_aligned.npz'))
    y_test = np.load(os.path.join(feat_dir, 'test_label.npz'))['label']

    # Try TabPFN
    try:
        import tabpfn
        use_tabpfn = True
    except Exception:
        use_tabpfn = False

    if use_tabpfn:
        print('Running TabPFN (may be slow)')
        clf = tabpfn.Predictor(device='cuda' if tabpfn.torch.cuda.is_available() else 'cpu')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:,1]
    else:
        print('TabPFN not available — falling back to RandomForest')
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        try:
            y_proba = clf.predict_proba(X_test)[:,1]
        except Exception:
            y_proba = None

    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {acc:.4f}')
    if y_proba is not None and len(np.unique(y_test))==2:
        try:
            auc = roc_auc_score(y_test, y_proba)
            print(f'AUC: {auc:.4f}')
        except Exception:
            pass

    np.save(os.path.join(feat_dir, 'y_pred_test.npy'), y_pred)
    if y_proba is not None:
        np.save(os.path.join(feat_dir, 'y_proba_test.npy'), y_proba)
    np.save(os.path.join(feat_dir, 'y_true_test.npy'), y_test)

if __name__ == '__main__':
    main()
