import json

import joblib
import pandas as pd
import pytest

from src.train import (
    METADATA_PATH,
    MODEL_PATH,
)

SAMPLE_INPUT = pd.DataFrame(
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
        }
    ]
)


def require_model_artifacts() -> None:
    if not MODEL_PATH.exists():
        pytest.fail(
            "Model artefact is missing. "
            "Run python src/train.py before pytest."
        )

    if not METADATA_PATH.exists():
        pytest.fail(
            "Model metadata is missing. "
            "Run python src/train.py before pytest."
        )


def test_model_and_metadata_files_exist():
    require_model_artifacts()

    assert MODEL_PATH.is_file()
    assert MODEL_PATH.stat().st_size > 0

    assert METADATA_PATH.is_file()
    assert METADATA_PATH.stat().st_size > 0


def test_saved_object_is_complete_pipeline():
    require_model_artifacts()

    pipeline = joblib.load(MODEL_PATH)

    assert hasattr(pipeline, "named_steps")
    assert set(pipeline.named_steps) == {
        "preprocessor",
        "model",
    }


def test_saved_pipeline_generates_prediction():
    require_model_artifacts()

    pipeline = joblib.load(MODEL_PATH)

    prediction = pipeline.predict(SAMPLE_INPUT)

    assert prediction.shape == (1,)
    assert int(prediction[0]) in {0, 1}


def test_saved_pipeline_generates_probability():
    require_model_artifacts()

    pipeline = joblib.load(MODEL_PATH)

    probability = pipeline.predict_proba(
        SAMPLE_INPUT
    )

    assert probability.shape == (1, 2)
    assert 0.0 <= float(probability[0, 1]) <= 1.0


def test_model_metadata_contains_required_keys():
    require_model_artifacts()

    metadata = json.loads(
        METADATA_PATH.read_text(
            encoding="utf-8"
        )
    )

    required_keys = {
        "project",
        "task",
        "selected_model",
        "selection_rule",
        "best_parameters",
        "best_cross_validation_metrics",
        "held_out_test_metrics",
        "model_comparison",
        "random_seed",
        "numerical_features",
        "categorical_features",
        "pipeline_steps",
        "preprocessing",
        "model_path",
    }

    assert required_keys.issubset(
        metadata.keys()
    )


def test_metadata_contains_required_metrics():
    require_model_artifacts()

    metadata = json.loads(
        METADATA_PATH.read_text(
            encoding="utf-8"
        )
    )

    required_metrics = {
        "accuracy",
        "precision",
        "recall",
        "roc_auc",
    }

    assert required_metrics.issubset(
        metadata[
            "best_cross_validation_metrics"
        ].keys()
    )

    assert required_metrics.issubset(
        metadata[
            "held_out_test_metrics"
        ].keys()
    )


def test_selected_model_is_supported():
    require_model_artifacts()

    metadata = json.loads(
        METADATA_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert metadata["selected_model"] in {
        "Logistic Regression",
        "Random Forest",
    }