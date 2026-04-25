from __future__ import annotations

import pandas as pd

from src.models.metrics import resolve_benign_label
from src.models.registry import load_bundle
from src.monitoring.alert_rate import compute_alert_rate
from src.monitoring.drift import (
    compute_feature_drift_report,
    jensen_shannon_divergence_from_counts,
)
from src.monitoring.reporting import build_monitoring_report
from src.serving.service import InferenceService
from src.utils.io import load_yaml, save_json
from src.visualization.dashboards import plot_drift_over_batches


def main() -> None:
    cfg = load_yaml("configs/config.yaml")
    target = cfg["schema"]["target_column"]
    benign_candidates = cfg["schema"]["benign_labels"]

    train_df = pd.read_parquet(f"{cfg['paths']['processed_dir']}/train_split.parquet")
    test_df = pd.read_parquet(f"{cfg['paths']['processed_dir']}/test_split.parquet")
    if target not in train_df.columns or target not in test_df.columns:
        raise ValueError(f"Configured target column '{target}' missing from train/test split files.")

    bundle = load_bundle(cfg["paths"]["models_dir"], "stronger_model")
    service = InferenceService(bundle, top_k=cfg["serving"]["top_k_probabilities"])

    feature_cols = bundle.feature_columns
    batch_size = cfg["streaming"]["batch_size"]
    benign_label = resolve_benign_label(benign_candidates, list(bundle.label_encoder.classes_))

    baseline_pred = service.predict_records(train_df[feature_cols].to_dict(orient="records"))
    baseline_pred_labels = pd.Series([p["predicted_label"] for p in baseline_pred])
    baseline_counts = baseline_pred_labels.value_counts().to_dict()
    baseline_alert_rate = compute_alert_rate(baseline_pred_labels, [benign_label])

    outputs = []
    batch_summaries = []
    per_batch_drift = []
    running_pred_labels: list[str] = []

    for i in range(0, len(test_df), batch_size):
        batch = test_df.iloc[i : i + batch_size]
        records = batch[feature_cols].to_dict(orient="records")
        preds = service.predict_records(records)
        batch_labels = pd.Series([p["predicted_label"] for p in preds])
        running_pred_labels.extend(batch_labels.tolist())

        batch_summaries.append(
            {
                "batch_id": i // batch_size,
                "batch_start": i,
                "batch_end": min(i + batch_size, len(test_df)),
                "batch_size": len(batch),
                "pred_counts": batch_labels.value_counts().to_dict(),
                "alert_rate": compute_alert_rate(batch_labels, [benign_label]),
            }
        )

        cumulative_df = test_df.iloc[: min(i + batch_size, len(test_df))]
        feature_drift_batch = compute_feature_drift_report(
            baseline_df=train_df[feature_cols],
            current_df=cumulative_df[feature_cols],
            numeric_columns=feature_cols,
            psi_threshold=cfg["monitoring"]["psi_threshold"],
        )
        running_counts = pd.Series(running_pred_labels).value_counts().to_dict()
        jsd = jensen_shannon_divergence_from_counts(baseline_counts, running_counts)
        per_batch_drift.append(
            {
                "batch_id": i // batch_size,
                "psi_per_feature": {
                    feature: values["psi"]
                    for feature, values in feature_drift_batch["per_feature"].items()
                },
                "jsd": jsd,
            }
        )
        outputs.extend(preds)

    plot_drift_over_batches(per_batch_drift)

    pred_labels = pd.Series([o["predicted_label"] for o in outputs])
    current_counts = pred_labels.value_counts().to_dict()
    current_alert_rate = compute_alert_rate(pred_labels, [benign_label])

    feature_drift = compute_feature_drift_report(
        baseline_df=train_df[feature_cols],
        current_df=test_df[feature_cols],
        numeric_columns=feature_cols,
        psi_threshold=cfg["monitoring"]["psi_threshold"],
    )
    pred_alert_report = build_monitoring_report(
        baseline_pred_counts=baseline_counts,
        current_pred_counts=current_counts,
        baseline_alert_rate=baseline_alert_rate,
        current_alert_rate=current_alert_rate,
        jsd_threshold=cfg["monitoring"]["jsd_threshold"],
        alert_spike_threshold=cfg["monitoring"]["alert_rate_spike_threshold"],
    )

    report = {
        "benign_label_used": benign_label,
        "n_batches": (len(test_df) + batch_size - 1) // batch_size,
        "n_predictions": len(outputs),
        "prediction_counts": current_counts,
        "batch_summaries": batch_summaries,
        "per_batch_drift": per_batch_drift,
        "feature_drift": feature_drift,
        "prediction_and_alert_monitoring": pred_alert_report,
    }
    save_json(report, f"{cfg['paths']['reports_metrics_dir']}/stream_monitoring_report.json")
    pd.DataFrame(outputs).to_csv(f"{cfg['paths']['reports_metrics_dir']}/stream_predictions.csv", index=False)


if __name__ == "__main__":
    main()
