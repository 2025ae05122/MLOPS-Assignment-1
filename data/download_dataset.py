import os
from pathlib import Path

import certifi
import pandas as pd
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[1]
RAW_DIRECTORY = ROOT / "data" / "raw"
RAW_PATH = RAW_DIRECTORY / "heart_disease_raw.csv"

CUSTOM_CA_ENV = "UCI_CA_BUNDLE"


def configure_ca_bundle() -> None:
    """
    Configure an optional trusted CA bundle for corporate networks.

    Local corporate machine:
        Set UCI_CA_BUNDLE to a combined public and corporate CA bundle.

    GitHub Actions:
        No variable is required. The standard certifi bundle is used.
    """

    custom_ca_path = os.getenv(CUSTOM_CA_ENV)

    if not custom_ca_path:
        print(
            "Certificate bundle: standard certifi CA bundle"
        )
        return

    ca_path = Path(custom_ca_path).expanduser().resolve()

    if not ca_path.exists():
        raise FileNotFoundError(
            f"{CUSTOM_CA_ENV} points to a missing file: "
            f"{ca_path}"
        )

    if not ca_path.is_file():
        raise ValueError(
            f"{CUSTOM_CA_ENV} must point to a PEM file: "
            f"{ca_path}"
        )

    # ucimlrepo internally calls certifi.where().
    # Redirect that call to the explicitly trusted combined bundle.
    certifi.where = lambda: str(ca_path)

    print(f"Certificate bundle: {ca_path}")


def download_dataset() -> pd.DataFrame:
    """
    Download UCI Heart Disease dataset ID 45 and save its raw CSV.

    This function always downloads the dataset from UCI, satisfying
    the assignment's explicit dataset-acquisition requirement.
    """

    RAW_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    configure_ca_bundle()

    print("Downloading UCI Heart Disease dataset ID 45...")

    try:
        heart = fetch_ucirepo(id=45)

    except Exception as exc:
        raise ConnectionError(
            "Unable to download UCI Heart Disease dataset ID 45.\n"
            "If running on a corporate network, obtain the approved "
            "corporate root/intermediate CA certificate from IT, create "
            "a combined CA bundle, and set UCI_CA_BUNDLE to that file.\n"
            "SSL verification has not been disabled."
        ) from exc

    dataset = pd.concat(
        [
            heart.data.features.copy(),
            heart.data.targets.copy(),
        ],
        axis=1,
    )

    original_target = heart.data.targets.columns[0]

    dataset = dataset.rename(
        columns={original_target: "target"}
    )

    target_numeric = pd.to_numeric(
        dataset["target"],
        errors="coerce",
    )

    if target_numeric.isna().any():
        raise ValueError(
            "The downloaded target contains invalid or missing values."
        )

    dataset["target"] = (
        target_numeric > 0
    ).astype(int)

    dataset.to_csv(
        RAW_PATH,
        index=False,
    )

    print("Dataset downloaded successfully.")
    print(f"Rows        : {len(dataset)}")
    print(f"Columns     : {dataset.shape[1]}")
    print("Target      : target")
    print(f"Saved to    : {RAW_PATH}")

    return dataset


if __name__ == "__main__":
    download_dataset()