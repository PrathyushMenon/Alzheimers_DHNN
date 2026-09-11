import json
from pathlib import Path

RESULTS_FILE = Path(r"D:\ALZ\results\multimodal_ad_results.json")


def load_results(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    return json.loads(path.read_text())


def aggregate_fold_predictions(results: dict) -> tuple[list[int], list[int]]:
    y_true = []
    y_pred = []
    for fold in results.get('fold_details', []):
        y_true.extend(fold.get('y_true', []))
        y_pred.extend(fold.get('y_pred', []))
    return y_true, y_pred


def print_summary(results: dict) -> None:
    print('RESULTS SUMMARY')
    print('===============')
    print(f"Mean accuracy: {results.get('mean_accuracy')}")
    print(f"Std accuracy: {results.get('std_accuracy')}")
    print(f"Subjects: {results.get('n_subjects')}")
    print(f"Features: {results.get('n_features')}")
    print(f"Paper target accuracy: {results.get('paper_target')}")
    print(f"Achieved accuracy: {results.get('achieved_accuracy')}")
    print(f"Classifier: {results.get('classifier_used')}")
    print('Feature components:')
    for comp, info in results.get('feature_components', {}).items():
        print(f"  {comp}: {info}")


def main() -> None:
    results = load_results(RESULTS_FILE)
    print_summary(results)

    if 'fold_details' in results:
        y_true, y_pred = aggregate_fold_predictions(results)
        print('\nFold-level details:')
        for fold in results['fold_details']:
            print(f"  Fold {fold.get('fold')}: accuracy={fold.get('accuracy')}, balanced_accuracy={fold.get('balanced_accuracy')}, f1_score={fold.get('f1_score')}")
        print(f"\nCombined samples from all folds: {len(y_true)}")
    else:
        print('No fold_details available in results JSON.')


if __name__ == '__main__':
    main()
