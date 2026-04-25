from src.data.validate_data import validate_dataset


def test_validate_dataset_has_expected_fields(synthetic_df):
    report = validate_dataset(synthetic_df, target_column="label")
    assert report["n_rows"] == len(synthetic_df)
    assert "class_distribution" in report
    assert "missing_values" in report
