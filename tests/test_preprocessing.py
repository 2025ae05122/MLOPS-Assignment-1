
import joblib
import numpy as np
import pandas as pd
import pytest

from src.train import (
    CATEGORICAL_FEATURES,
    MODEL_PATH,
    build_preprocessor,
    prepare_dataset,
)

EXPECTED_FEATURES = [
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
]


def create_test_dataset() -> pd.DataFrame:
    """Create a small dataset containing missing values and duplicates."""

    dataset = pd.DataFrame(
        [
            {
                "age": 63,
                "sex": 1,
                "cp": 1,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 2,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 3,
                "ca": 0,
                "thal": 6,
                "target": 1,
            },
            {
                "age": 41,
                "sex": 0,
                "cp": 2,
                "trestbps": "?",
                "chol": 204,
                "fbs": 0,
                "restecg": 0,
                "thalach": 172,
                "exang": 0,
                "oldpeak": 1.4,
                "slope": 1,
                "ca": None,
                "thal": 3,
                "target": 0,
            },
            {
                "age": 63,
                "sex": 1,
                "cp": 1,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 2,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 3,
                "ca": 0,
                "thal": 6,
                "target": 1,
            },
        ]
    )

    return dataset


def test_prepare_dataset_removes_duplicates():
    dataset = create_test_dataset()

    features, target = prepare_dataset(dataset)

    prepared = pd.concat(
        [
            features,
            target.rename("target"),
        ],
        axis=1,
    )

    assert not prepared.duplicated().any()
    assert len(prepared) == 2


def test_prepare_dataset_converts_missing_markers_to_nan():
    dataset = create_test_dataset()

    features, _ = prepare_dataset(dataset)

    assert pd.isna(
        features.loc[
            features["age"] == 41,
            "trestbps",
        ]
    ).all()


def test_prepare_dataset_removes_target_from_features():
    dataset = create_test_dataset()

    features, target = prepare_dataset(dataset)

    assert "target" not in features.columns
    assert target.name == "target"


def test_prepare_dataset_returns_binary_target():
    dataset = create_test_dataset()

    _, target = prepare_dataset(dataset)

    assert set(target.unique()).issubset({0, 1})
    assert target.nunique() == 2


def test_build_preprocessor_handles_missing_values():
    dataset = create_test_dataset()

    features, target = prepare_dataset(dataset)

    categorical_features = [
        feature
        for feature in CATEGORICAL_FEATURES
        if feature in features.columns
    ]

    numerical_features = [
        feature
        for feature in features.columns
        if feature not in categorical_features
    ]

    preprocessor = build_preprocessor(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )

    transformed = preprocessor.fit_transform(
        features,
        target,
    )

    assert transformed.shape[0] == len(features)
    assert transformed.shape[1] > len(EXPECTED_FEATURES)
    assert int(np.isnan(transformed).sum()) == 0


def test_preprocessor_produces_feature_names():
    dataset = create_test_dataset()

    features, target = prepare_dataset(dataset)

    categorical_features = [
        feature
        for feature in CATEGORICAL_FEATURES
        if feature in features.columns
    ]

    numerical_features = [
        feature
        for feature in features.columns
        if feature not in categorical_features
    ]

    preprocessor = build_preprocessor(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )

    preprocessor.fit(features, target)

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    assert len(feature_names) > 0
    assert any(
        "age" in feature_name
        for feature_name in feature_names
    )


def test_saved_pipeline_contains_preprocessor():
    if not MODEL_PATH.exists():
        pytest.skip(
            "Model artefact does not exist. "
            "Run python src/train.py first."
        )

    pipeline = joblib.load(MODEL_PATH)

    assert hasattr(pipeline, "named_steps")
    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_saved_pipeline_handles_raw_input_with_missing_values():
    if not MODEL_PATH.exists():
        pytest.skip(
            "Model artefact does not exist. "
            "Run python src/train.py first."
        )

    pipeline = joblib.load(MODEL_PATH)

    raw_input = pd.DataFrame(
        [
            {
                "age": 63,
                "sex": 1,
                "cp": 1,
                "trestbps": None,
                "chol": 233,
                "fbs": 1,
                "restecg": 2,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 3,
                "ca": None,
                "thal": 6,
            }
        ]
    )

    prediction = pipeline.predict(raw_input)
    probability = pipeline.predict_proba(raw_input)

    assert prediction.shape == (1,)
    assert probability.shape == (1, 2)
    assert int(prediction[0]) in {0, 1}
    assert 0.0 <= float(probability[0, 1]) <= 1.0