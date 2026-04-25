from __future__ import annotations

import pandas as pd


def sample_dataframe(
    df: pd.DataFrame,
    max_rows: int | None,
    strategy: str,
    target_column: str,
    random_seed: int | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Return a sampled dataframe while preserving all original columns,
    including the target column.
    """
    if df.empty:
        raise ValueError("Cannot sample an empty dataframe.")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found before sampling.")

    seed = random_seed if random_seed is not None else random_state
    if seed is None:
        seed = 42

    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        sampled_df = df.copy()

    elif strategy == "random":
        sampled_df = df.sample(n=max_rows, random_state=seed)

    elif strategy == "stratified":
        sampled_indices = []
        class_counts = df[target_column].value_counts(dropna=False)

        for class_value, class_count in class_counts.items():
            class_rows = df[df[target_column] == class_value]

            n_class = max(1, round(max_rows * class_count / len(df)))
            n_class = min(n_class, len(class_rows))

            sampled_class_indices = class_rows.sample(
                n=n_class,
                random_state=seed,
                replace=False,
            ).index.tolist()

            sampled_indices.extend(sampled_class_indices)

        sampled_df = df.loc[sampled_indices]

        if len(sampled_df) > max_rows:
            sampled_df = sampled_df.sample(n=max_rows, random_state=seed)

        sampled_df = sampled_df.sample(frac=1.0, random_state=seed)

    else:
        raise ValueError(
            f"Unsupported sampling strategy '{strategy}'. "
            "Use 'random' or 'stratified'."
        )

    sampled_df = sampled_df.reset_index(drop=True)

    if target_column not in sampled_df.columns:
        raise ValueError(
            f"Sampling removed target column '{target_column}'. "
            "The sampled dataframe must preserve the target column."
        )

    return sampled_df