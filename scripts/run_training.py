from __future__ import annotations

import logging

import pandas as pd

from src.data.load_data import load_dataset
from src.data.sampling import sample_dataframe
from src.data.split_data import split_dataset
from src.features.preprocess import fit_preprocessor, transform_features
from src.models.metrics import compute_classification_metrics, resolve_benign_label
from src.models.registry import ModelBundle, save_bundle
from src.models.train_baseline import train_baseline_model
from src.models.train_tree_model import train_tree_model
from src.utils.io import load_yaml, save_json
from src.utils.logging_utils import configure_logging
from src.utils.seed import set_seed


def _evaluate(model, prep, le, X: pd.DataFrame, y: pd.Series, benign_label: str):
    Xt = transform_features(prep, X)
    pred_idx = model.predict(Xt)
    pred = le.inverse_transform(pred_idx)
    prob = model.predict_proba(Xt) if hasattr(model, "predict_proba") else None
    labels = list(le.classes_)
    for label in y.astype(str).unique():
        if label not in labels:
            labels.append(label)
    return compute_classification_metrics(y.astype(str).to_numpy(), pred, labels, benign_label, prob)


def main() -> None:
    configure_logging()
    cfg = load_yaml("configs/config.yaml")
    set_seed(cfg["project"]["random_seed"])
    logger = logging.getLogger("run_training")
    split_cfg = cfg.get("splitting", cfg.get("split", {}))

    target = cfg["schema"]["target_column"]
    df = load_dataset(cfg["paths"]["raw_data_path"])
    if target not in df.columns:
        raise ValueError(f"Configured target column '{target}' not found in dataset.")

    if cfg["sampling"]["enabled"]:
        df = sample_dataframe(
            df,
            cfg["sampling"]["max_rows"],
            cfg["sampling"]["strategy"],
            target,
            cfg["project"]["random_seed"],
        )

    split = split_dataset(
        df=df,
        target_column=target,
        train_size=split_cfg["train_size"],
        val_size=split_cfg["val_size"],
        test_size=split_cfg["test_size"],
        random_seed=cfg["project"]["random_seed"],
        timestamp_column=cfg["schema"]["timestamp_column"],
        group_column=cfg["schema"]["group_column"],
        min_class_count=split_cfg.get("min_class_count", 2),
    )

    train_df, val_df, test_df = split["train"], split["val"], split["test"]

    excluded_feature_columns = set(cfg["schema"].get("id_columns", []))
    if cfg["schema"].get("timestamp_column"):
        excluded_feature_columns.add(cfg["schema"]["timestamp_column"])
    if cfg["schema"].get("group_column"):
        excluded_feature_columns.add(cfg["schema"]["group_column"])

    candidate_features = [c for c in train_df.columns if c != target and c not in excluded_feature_columns]
    X_train, y_train = train_df[candidate_features], train_df[target].astype(str)
    X_val, y_val = val_df[candidate_features], val_df[target].astype(str)

    prep_art = fit_preprocessor(
        X_train, y_train, cfg["preprocessing"]["imputer_strategy"], cfg["preprocessing"]["scale_numeric"]
    )

    X_train_t = transform_features(prep_art.feature_pipeline, X_train)
    y_train_enc = prep_art.label_encoder.transform(y_train)

    baseline_model = train_baseline_model(X_train_t, y_train_enc, cfg["models"]["baseline"]["params"])
    tree_model = train_tree_model(X_train_t, y_train_enc, cfg["models"]["stronger"]["params"])

    benign_label = resolve_benign_label(cfg["schema"]["benign_labels"], list(prep_art.label_encoder.classes_))
    baseline_metrics = _evaluate(
        baseline_model, prep_art.feature_pipeline, prep_art.label_encoder, X_val, y_val, benign_label
    )
    stronger_metrics = _evaluate(
        tree_model, prep_art.feature_pipeline, prep_art.label_encoder, X_val, y_val, benign_label
    )

    save_bundle(
        ModelBundle(
            "baseline_model",
            baseline_model,
            prep_art.feature_pipeline,
            prep_art.label_encoder,
            prep_art.feature_columns,
        ),
        cfg["paths"]["models_dir"],
    )
    save_bundle(
        ModelBundle(
            "stronger_model",
            tree_model,
            prep_art.feature_pipeline,
            prep_art.label_encoder,
            prep_art.feature_columns,
        ),
        cfg["paths"]["models_dir"],
    )

    train_manifest = {
        "split_strategy": split["split_strategy"],
        "rare_class_filter": split["rare_class_filter"],
        "stratification_warnings": split["stratification_warnings"],
        "rows": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "excluded_feature_columns": sorted(excluded_feature_columns),
        "n_model_features": len(prep_art.feature_columns),
        "benign_label_used": benign_label,
        "baseline_val_metrics": baseline_metrics,
        "stronger_val_metrics": stronger_metrics,
    }
    save_json(train_manifest, f"{cfg['paths']['reports_metrics_dir']}/training_summary.json")

    test_df.to_parquet(f"{cfg['paths']['processed_dir']}/test_split.parquet", index=False)
    train_df.to_parquet(f"{cfg['paths']['processed_dir']}/train_split.parquet", index=False)
    logger.info("Training complete and artifacts saved.")


if __name__ == "__main__":
    main()
