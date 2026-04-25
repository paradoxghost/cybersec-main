import pandas as pd

from src.data.split_data import split_dataset


def test_split_filters_classes_below_min_count():
    df = pd.DataFrame(
        {
            "feature": range(9),
            "Label": ["Benign"] * 4 + ["DDoS"] * 4 + ["UPLOADING_ATTACK"],
        }
    )

    split = split_dataset(
        df,
        target_column="Label",
        train_size=0.5,
        val_size=0.25,
        test_size=0.25,
        random_seed=42,
        min_class_count=2,
    )

    combined = pd.concat([split["train"], split["val"], split["test"]], ignore_index=True)
    assert "UPLOADING_ATTACK" not in set(combined["Label"])
    assert split["rare_class_filter"]["rows_removed"] == 1
    assert split["rare_class_filter"]["removed_classes"] == [
        {"class_label": "UPLOADING_ATTACK", "count": 1}
    ]


def test_split_falls_back_when_remaining_classes_cannot_be_stratified():
    df = pd.DataFrame(
        {
            "feature": range(4),
            "Label": ["Benign", "Benign", "DDoS", "DDoS"],
        }
    )

    split = split_dataset(
        df,
        target_column="Label",
        train_size=0.5,
        val_size=0.25,
        test_size=0.25,
        random_seed=42,
        min_class_count=2,
    )

    assert split["split_strategy"] == "partially_stratified_random_fallback"
    assert split["stratification_warnings"]
    assert sum(len(split[name]) for name in ["train", "val", "test"]) == len(df)
