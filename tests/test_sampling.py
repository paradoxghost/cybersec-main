import pandas as pd

from src.data.sampling import sample_dataframe


def test_sampling_preserves_target_and_all_columns(synthetic_df):
    df = synthetic_df.rename(columns={"label": "Label"})
    sampled = sample_dataframe(
        df,
        max_rows=120,
        strategy="stratified",
        target_column="Label",
        random_state=42,
    )

    assert "Label" in sampled.columns
    assert set(sampled.columns) == set(df.columns)
    assert isinstance(sampled, pd.DataFrame)


def test_random_sampling_preserves_target_column(synthetic_df):
    sampled = sample_dataframe(
        synthetic_df,
        max_rows=120,
        strategy="random",
        target_column="label",
        random_state=42,
    )
    assert "label" in sampled.columns
