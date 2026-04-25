from __future__ import annotations

import pandas as pd

from src.models.metrics import compute_classification_metrics, resolve_benign_label
from src.models.registry import load_bundle
from src.utils.io import load_yaml, save_json
from src.visualization.plots import save_confusion_matrix


def evaluate_model(bundle, X: pd.DataFrame, y: pd.Series, benign_label: str):
    Xt = bundle.preprocessor.transform(X[bundle.feature_columns])
    pred_idx = bundle.model.predict(Xt)
    pred = bundle.label_encoder.inverse_transform(pred_idx)
    prob = bundle.model.predict_proba(Xt)
    labels = list(bundle.label_encoder.classes_)
    for label in y.astype(str).unique():
        if label not in labels:
            labels.append(label)
    return compute_classification_metrics(
        y.astype(str).to_numpy(), pred, labels, benign_label, prob
    )


def main() -> None:
    cfg = load_yaml("configs/config.yaml")
    target = cfg["schema"]["target_column"]

    test_df = pd.read_parquet(f"{cfg['paths']['processed_dir']}/test_split.parquet")
    if target not in test_df.columns:
        raise ValueError(f"Configured target column '{target}' not found in test split file.")

    X_test, y_test = test_df.drop(columns=[target]), test_df[target]

    baseline = load_bundle(cfg["paths"]["models_dir"], "baseline_model")
    stronger = load_bundle(cfg["paths"]["models_dir"], "stronger_model")
    benign_label = resolve_benign_label(cfg["schema"]["benign_labels"], list(stronger.label_encoder.classes_))

    baseline_m = evaluate_model(baseline, X_test, y_test, benign_label)
    stronger_m = evaluate_model(stronger, X_test, y_test, benign_label)

    save_json(baseline_m, f"{cfg['paths']['reports_metrics_dir']}/baseline_test_metrics.json")
    save_json(stronger_m, f"{cfg['paths']['reports_metrics_dir']}/stronger_test_metrics.json")

    comparison = pd.DataFrame(
        [
            {
                "model": "baseline",
                "accuracy": baseline_m["accuracy"],
                "macro_f1": baseline_m["macro_f1"],
                "weighted_f1": baseline_m["weighted_f1"],
                "fpr_benign": baseline_m["false_positive_rate_on_benign"],
            },
            {
                "model": "stronger",
                "accuracy": stronger_m["accuracy"],
                "macro_f1": stronger_m["macro_f1"],
                "weighted_f1": stronger_m["weighted_f1"],
                "fpr_benign": stronger_m["false_positive_rate_on_benign"],
            },
        ]
    )
    comparison.to_csv(f"{cfg['paths']['reports_metrics_dir']}/model_comparison.csv", index=False)

    save_confusion_matrix(
        baseline_m["confusion_matrix"],
        baseline_m["labels_used"],
        f"{cfg['paths']['reports_figures_dir']}/cm_baseline.png",
        "Baseline Confusion Matrix",
    )
    save_confusion_matrix(
        stronger_m["confusion_matrix"],
        stronger_m["labels_used"],
        f"{cfg['paths']['reports_figures_dir']}/cm_stronger.png",
        "Stronger Model Confusion Matrix",
    )


if __name__ == "__main__":
    main()
