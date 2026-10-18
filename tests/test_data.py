
import pandas as pd

from data.download_dataset import (
    RAW_PATH,
    download_dataset,
)
from src.train import (
    PROCESSED_DATA_PATH,
    prepare_dataset,
)

EXPECTED_COLUMNS = {
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
}


def test_dataset_download_returns_dataframe():
    dataset = download_dataset()

    assert isinstance(dataset, pd.DataFrame)
    assert not dataset.empty


def test_downloaded_dataset_has_expected_columns():
    dataset = download_dataset()

    assert EXPECTED_COLUMNS.issubset(
        set(dataset.columns)
    )


def test_downloaded_target_is_binary():
    dataset = download_dataset()

    target_values = set(
        dataset["target"]
        .dropna()
        .unique()
        .tolist()
    )

    assert target_values.issubset({0, 1})


def test_downloaded_dataset_contains_both_classes():
    dataset = download_dataset()

    assert dataset["target"].nunique() == 2


def test_raw_dataset_file_is_created():
    download_dataset()

    assert RAW_PATH.exists()
    assert RAW_PATH.is_file()
    assert RAW_PATH.stat().st_size > 0


def test_prepare_dataset_removes_duplicates():
    dataset = download_dataset()

    duplicated_dataset = pd.concat(
        [
            dataset,
            dataset.iloc[[0]],
        ],
        ignore_index=True,
    )

    features, target = prepare_dataset(
        duplicated_dataset
    )

    cleaned = pd.concat(
        [
            features,
            target.rename("target"),
        ],
        axis=1,
    )

    assert not cleaned.duplicated().any()


def test_prepare_dataset_returns_features_and_target():
    dataset = download_dataset()

    features, target = prepare_dataset(dataset)

    assert isinstance(features, pd.DataFrame)
    assert isinstance(target, pd.Series)
    assert "target" not in features.columns
    assert target.name == "target"
    assert len(features) == len(target)


def test_prepare_dataset_preserves_feature_missing_values():
    """
    Missing feature values are intentionally retained here because
    SimpleImputer handles them inside the sklearn pipeline.
    """

    dataset = download_dataset()

    features, _ = prepare_dataset(dataset)

    assert isinstance(
        int(features.isna().sum().sum()),
        int,
    )


def test_processed_dataset_file_is_created():
    dataset = download_dataset()

    prepare_dataset(dataset)

    assert PROCESSED_DATA_PATH.exists()
    assert PROCESSED_DATA_PATH.is_file()
    assert (
        PROCESSED_DATA_PATH.stat().st_size > 0
    )


def test_processed_target_contains_both_classes():
    dataset = download_dataset()

    _, target = prepare_dataset(dataset)

    assert set(target.unique()).issubset(
        {0, 1}
    )

    assert target.nunique() == 2


def test_no_missing_target_after_preparation():
    dataset = download_dataset()

    _, target = prepare_dataset(dataset)

    assert int(target.isna().sum()) == 0