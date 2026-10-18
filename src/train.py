import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from data.download_dataset import download_dataset

SEED = 42

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIRECTORY = ROOT / "artifacts"
PROCESSED_DATA_DIRECTORY = ROOT / "data" / "processed"

MODEL_PATH = (
    ARTIFACT_DIRECTORY /
    "heart_disease_pipeline.joblib"
)

METADATA_PATH = (
    ARTIFACT_DIRECTORY /
    "model_metadata.json"
)

PROCESSED_DATA_PATH = (
    PROCESSED_DATA_DIRECTORY /
    "heart_disease_cleaned.csv"
)


CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "thal",
]


def create_directories() -> None:
    """Create directories required by model training."""

    ARTIFACT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROCESSED_DATA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


def prepare_dataset(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Validate and prepare features and target.

    The function:
    - replaces common missing-value markers with NaN;
    - converts all columns to numeric;
    - removes duplicate records;
    - removes rows with a missing target;
    - saves the cleaned, unencoded dataset;
    - returns raw feature columns because preprocessing is handled
      inside the sklearn model pipeline.
    """

    required_columns = set(
        CATEGORICAL_FEATURES
        + [
            "age",
            "trestbps",
            "chol",
            "thalach",
            "oldpeak",
            "ca",
            "target",
        ]
    )

    missing_columns = (
        required_columns - set(dataset.columns)
    )

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    cleaned = dataset.copy()

    cleaned = cleaned.replace(
        ["?", "NA", "N/A", "null", "None", ""],
        np.nan,
    )

    for column in cleaned.columns:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

    cleaned = (
        cleaned
        .drop_duplicates()
        .dropna(subset=["target"])
        .reset_index(drop=True)
    )

    cleaned["target"] = (
        cleaned["target"].astype(int)
    )

    if not set(
        cleaned["target"].unique()
    ).issubset({0, 1}):
        raise ValueError(
            "Target must contain only binary values 0 and 1."
        )

    if cleaned["target"].nunique() != 2:
        raise ValueError(
            "Target must contain both classes."
        )

    cleaned.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
    )

    features = cleaned.drop(columns=["target"])
    target = cleaned["target"]

    return features, target


def build_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    Create numerical and categorical preprocessing pipelines.

    Numerical:
    - median imputation;
    - StandardScaler.

    Categorical:
    - most-frequent imputation;
    - one-hot encoding.
    """

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_model_configurations(
    preprocessor: ColumnTransformer,
) -> dict:
    """Define the two classifiers and tuning grids."""

    return {
        "Logistic Regression": {
            "pipeline": Pipeline(
                steps=[
                    (
                        "preprocessor",
                        clone(preprocessor),
                    ),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=SEED,
                        ),
                    ),
                ]
            ),
            "parameter_grid": {
                "model__C": [
                    0.01,
                    0.1,
                    1,
                    10,
                ],
                "model__solver": [
                    "liblinear",
                    "lbfgs",
                ],
            },
        },
        "Random Forest": {
            "pipeline": Pipeline(
                steps=[
                    (
                        "preprocessor",
                        clone(preprocessor),
                    ),
                    (
                        "model",
                        RandomForestClassifier(
                            class_weight="balanced",
                            random_state=SEED,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            "parameter_grid": {
                "model__n_estimators": [
                    100,
                    200,
                    400,
                ],
                "model__max_depth": [
                    None,
                    5,
                    10,
                ],
                "model__min_samples_split": [
                    2,
                    5,
                    10,
                ],
            },
        },
    }


def train() -> dict:
    """
    Download data, train and tune two models, select the best model,
    evaluate it and save the complete reusable pipeline.
    """

    np.random.seed(SEED)
    create_directories()

    print("=" * 80)
    print("AUTOMATED MODEL TRAINING")
    print("=" * 80)

    # ---------------------------------------------------------------
    # 1. Download and prepare data
    # ---------------------------------------------------------------

    dataset = download_dataset()

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

    X_train, X_test, y_train, y_test = (
        train_test_split(
            features,
            target,
            test_size=0.20,
            random_state=SEED,
            stratify=target,
        )
    )

    print("\nDATA PREPARATION")
    print("-" * 80)
    print(f"Total records       : {len(features)}")
    print(f"Training records    : {len(X_train)}")
    print(f"Testing records     : {len(X_test)}")
    print(f"Numerical features  : {numerical_features}")
    print(f"Categorical features: {categorical_features}")
    print(f"Cleaned data saved  : {PROCESSED_DATA_PATH}")

    # ---------------------------------------------------------------
    # 2. Create preprocessing and model configurations
    # ---------------------------------------------------------------

    preprocessor = build_preprocessor(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )

    model_configurations = (
        build_model_configurations(preprocessor)
    )

    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
    }

    # ---------------------------------------------------------------
    # 3. Tune and evaluate both models with cross-validation
    # ---------------------------------------------------------------

    model_results = []
    completed_searches = {}

    for model_name, configuration in (
        model_configurations.items()
    ):
        print("\n" + "-" * 80)
        print(f"TRAINING AND TUNING: {model_name}")
        print("-" * 80)

        search = GridSearchCV(
            estimator=configuration["pipeline"],
            param_grid=configuration[
                "parameter_grid"
            ],
            scoring=scoring,
            refit="roc_auc",
            cv=cross_validation,
            n_jobs=-1,
            return_train_score=False,
            error_score="raise",
        )

        search.fit(X_train, y_train)

        completed_searches[model_name] = search

        best_index = search.best_index_

        result = {
            "model": model_name,
            "best_parameters":
                search.best_params_,
            "mean_cv_accuracy": float(
                search.cv_results_[
                    "mean_test_accuracy"
                ][best_index]
            ),
            "mean_cv_precision": float(
                search.cv_results_[
                    "mean_test_precision"
                ][best_index]
            ),
            "mean_cv_recall": float(
                search.cv_results_[
                    "mean_test_recall"
                ][best_index]
            ),
            "mean_cv_roc_auc": float(
                search.cv_results_[
                    "mean_test_roc_auc"
                ][best_index]
            ),
            "cv_roc_auc_std": float(
                search.cv_results_[
                    "std_test_roc_auc"
                ][best_index]
            ),
            "parameter_combinations": len(
                search.cv_results_["params"]
            ),
        }

        model_results.append(result)

        print(
            "Best parameters     : "
            f"{result['best_parameters']}"
        )
        print(
            "Mean CV accuracy    : "
            f"{result['mean_cv_accuracy']:.4f}"
        )
        print(
            "Mean CV precision   : "
            f"{result['mean_cv_precision']:.4f}"
        )
        print(
            "Mean CV recall      : "
            f"{result['mean_cv_recall']:.4f}"
        )
        print(
            "Mean CV ROC-AUC     : "
            f"{result['mean_cv_roc_auc']:.4f}"
        )

    # ---------------------------------------------------------------
    # 4. Select final model
    # ---------------------------------------------------------------

    model_results.sort(
        key=lambda result: result[
            "mean_cv_roc_auc"
        ],
        reverse=True,
    )

    best_result = model_results[0]
    best_model_name = best_result["model"]

    best_model = completed_searches[
        best_model_name
    ].best_estimator_

    # ---------------------------------------------------------------
    # 5. Evaluate selected model on held-out test set
    # ---------------------------------------------------------------

    test_predictions = best_model.predict(
        X_test
    )

    test_probabilities = (
        best_model.predict_proba(X_test)[:, 1]
    )

    test_metrics = {
        "accuracy": float(
            accuracy_score(
                y_test,
                test_predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_test,
                test_predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                test_predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_test,
                test_probabilities,
            )
        ),
    }

    # ---------------------------------------------------------------
    # 6. Save complete fitted pipeline and metadata
    # ---------------------------------------------------------------

    joblib.dump(
        best_model,
        MODEL_PATH,
    )

    fitted_preprocessor = (
        best_model.named_steps["preprocessor"]
    )

    transformed_feature_names = (
        fitted_preprocessor
        .get_feature_names_out()
        .tolist()
    )

    metadata = {
        "project":
            "Heart Disease End-to-End MLOps",
        "task":
            "Binary heart-disease classification",
        "selected_model":
            best_model_name,
        "selection_rule":
            "Highest mean stratified 5-fold CV ROC-AUC",
        "best_parameters":
            best_result["best_parameters"],
        "best_cross_validation_metrics": {
            "accuracy":
                best_result["mean_cv_accuracy"],
            "precision":
                best_result["mean_cv_precision"],
            "recall":
                best_result["mean_cv_recall"],
            "roc_auc":
                best_result["mean_cv_roc_auc"],
            "roc_auc_std":
                best_result["cv_roc_auc_std"],
        },
        "held_out_test_metrics":
            test_metrics,
        "model_comparison":
            model_results,
        "random_seed":
            SEED,
        "training_records":
            int(len(X_train)),
        "testing_records":
            int(len(X_test)),
        "original_features":
            features.columns.tolist(),
        "numerical_features":
            numerical_features,
        "categorical_features":
            categorical_features,
        "transformed_feature_names":
            transformed_feature_names,
        "pipeline_steps":
            list(best_model.named_steps.keys()),
        "preprocessing": {
            "numerical_imputation":
                "median",
            "numerical_scaling":
                "StandardScaler",
            "categorical_imputation":
                "most_frequent",
            "categorical_encoding":
                "OneHotEncoder(handle_unknown='ignore')",
        },
        "model_path":
            str(MODEL_PATH.relative_to(ROOT)),
        "cleaned_data_path":
            str(
                PROCESSED_DATA_PATH.relative_to(
                    ROOT
                )
            ),
        "disclaimer": (
            "Educational use only; "
            "not a medical diagnostic device."
        ),
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------------
    # 7. Reload verification
    # ---------------------------------------------------------------

    reloaded_model = joblib.load(MODEL_PATH)

    verification_rows = X_test.head(
        min(10, len(X_test))
    )

    original_predictions = best_model.predict(
        verification_rows
    )

    reloaded_predictions = (
        reloaded_model.predict(
            verification_rows
        )
    )

    if not np.array_equal(
        original_predictions,
        reloaded_predictions,
    ):
        raise AssertionError(
            "Reloaded model predictions do not match."
        )

    # ---------------------------------------------------------------
    # 8. Print final CI-visible result
    # ---------------------------------------------------------------

    run_summary = {
        "selected_model": best_model_name,
        "best_parameters": best_result["best_parameters"],
        "best_cv_roc_auc": best_result["mean_cv_roc_auc"],
        "held_out_test_metrics": test_metrics,
        "model_path": str(MODEL_PATH),
        "metadata_path": str(METADATA_PATH),
        "cleaned_data_path": str(PROCESSED_DATA_PATH),
        "reload_verification": "PASS",
    }

    print("\n" + "=" * 80)
    print("AUTOMATED TRAINING RESULT: SUCCESS")
    print("=" * 80)

    print(
        json.dumps(
            run_summary,
            indent=2,
            default=str,
        )
    )

    return run_summary
if __name__ == "__main__":
    train()