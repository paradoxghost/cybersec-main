"""Leakage-aware data split logic."""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

logger = logging.getLogger(__name__)


def _format_class_label(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    return str(value)


def build_rare_class_filter_report(
    df: pd.DataFrame,
    target_column: str,
    min_class_count: int,
) -> dict[str, Any]:
    class_counts = df[target_column].value_counts(dropna=False)
    rare_counts = class_counts[class_counts < min_class_count]
    removed_classes = [
        {"class_label": _format_class_label(label), "count": int(count)}
        for label, count in rare_counts.items()
    ]
    rows_removed = int(rare_counts.sum())

    return {
        "min_class_count": int(min_class_count),
        "rows_before": int(len(df)),
        "rows_after": int(len(df) - rows_removed),
        "rows_removed": rows_removed,
        "removed_classes": removed_classes,
    }


def _filter_rare_classes(
    df: pd.DataFrame,
    target_column: str,
    min_class_count: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if min_class_count < 1:
        raise ValueError("min_class_count must be at least 1.")

    report = build_rare_class_filter_report(df, target_column, min_class_count)
    if report["rows_removed"] == 0:
        return df.copy(), report

    class_counts = df[target_column].value_counts(dropna=False)
    rare_labels = class_counts[class_counts < min_class_count].index
    filtered_df = df.loc[~df[target_column].isin(rare_labels)].copy()

    logger.warning(
        "Removed %s rows from %s rare classes before stratified splitting: %s",
        report["rows_removed"],
        len(report["removed_classes"]),
        report["removed_classes"],
    )
    return filtered_df, report


def _safe_train_test_split(
    df: pd.DataFrame,
    *,
    train_size: float | None = None,
    test_size: float | None = None,
    stratify: pd.Series | None = None,
    random_state: int,
    split_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, bool, str | None]:
    try:
        left_df, right_df = train_test_split(
            df,
            train_size=train_size,
            test_size=test_size,
            stratify=stratify,
            random_state=random_state,
        )
        return left_df, right_df, True, None
    except ValueError as exc:
        warning = (
            f"Stratified {split_name} split failed after rare-class filtering "
            f"({exc}). Falling back to a non-stratified split."
        )
        logger.warning(warning)
        left_df, right_df = train_test_split(
            df,
            train_size=train_size,
            test_size=test_size,
            stratify=None,
            random_state=random_state,
        )
        return left_df, right_df, False, warning


def split_dataset(
    df: pd.DataFrame,
    target_column: str,
    train_size: float,
    val_size: float,
    test_size: float,
    random_seed: int,
    timestamp_column: str | None = None,
    group_column: str | None = None,
    min_class_count: int = 2,
) -> dict[str, Any]:
    if abs((train_size + val_size + test_size) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found before splitting.")

    df, rare_class_filter = _filter_rare_classes(df, target_column, min_class_count)
    if df.empty:
        raise ValueError(
            "Rare-class filtering removed all rows. Lower splitting.min_class_count "
            "or use a larger sample before training."
        )
    y = df[target_column]

    if timestamp_column and timestamp_column in df.columns:
        sorted_df = df.sort_values(timestamp_column).reset_index(drop=True)
        n = len(sorted_df)
        train_end = int(n * train_size)
        val_end = int(n * (train_size + val_size))
        return {
            "train": sorted_df.iloc[:train_end].copy(),
            "val": sorted_df.iloc[train_end:val_end].copy(),
            "test": sorted_df.iloc[val_end:].copy(),
            "split_strategy": "time_aware",
            "rare_class_filter": rare_class_filter,
            "stratification_warnings": [],
        }

    if group_column and group_column in df.columns:
        gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_seed)
        train_idx, hold_idx = next(gss.split(df, y=y, groups=df[group_column]))
        train_df = df.iloc[train_idx].copy()
        hold_df = df.iloc[hold_idx].copy()

        hold_ratio = test_size / (test_size + val_size)
        gss2 = GroupShuffleSplit(n_splits=1, train_size=(1 - hold_ratio), random_state=random_seed)
        val_idx, test_idx = next(gss2.split(hold_df, groups=hold_df[group_column]))
        return {
            "train": train_df,
            "val": hold_df.iloc[val_idx].copy(),
            "test": hold_df.iloc[test_idx].copy(),
            "split_strategy": "group_aware",
            "rare_class_filter": rare_class_filter,
            "stratification_warnings": [],
        }

    warnings = []
    train_df, hold_df, train_stratified, warning = _safe_train_test_split(
        df,
        train_size=train_size,
        stratify=df[target_column],
        random_state=random_seed,
        split_name="train/holdout",
    )
    if warning:
        warnings.append(warning)

    hold_test_ratio = test_size / (test_size + val_size)
    val_df, test_df, hold_stratified, warning = _safe_train_test_split(
        hold_df,
        test_size=hold_test_ratio,
        stratify=hold_df[target_column],
        random_state=random_seed,
        split_name="validation/test",
    )
    if warning:
        warnings.append(warning)

    split_strategy = "stratified_random"
    if not train_stratified and not hold_stratified:
        split_strategy = "random_fallback"
    elif not train_stratified or not hold_stratified:
        split_strategy = "partially_stratified_random_fallback"

    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
        "split_strategy": split_strategy,
        "rare_class_filter": rare_class_filter,
        "stratification_warnings": warnings,
    }
