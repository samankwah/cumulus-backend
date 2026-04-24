"""Dataset assembly utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from cumulus.preprocessing.cleaning import standardize_frame
from cumulus.preprocessing.features import create_features


REQUIRED_TRAINING_COLUMNS = {"location_id", "time", "rainfall_mm"}
REQUIRED_INFERENCE_COLUMNS = {"location_id", "time"}


@dataclass
class SplitDataset:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def date_ranges(self) -> dict[str, dict[str, Any]]:
        return {
            "train": summarize_time_window(self.train),
            "validation": summarize_time_window(self.validation),
            "test": summarize_time_window(self.test),
        }


def build_training_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = standardize_frame(df)
    featured = create_features(cleaned, history_column="rainfall_mm")
    _validate_required_columns(featured, REQUIRED_TRAINING_COLUMNS, dataset_name="training")
    return featured


def build_inference_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = standardize_frame(df)
    featured = create_features(cleaned, history_column="precip_mm")
    _validate_required_columns(featured, REQUIRED_INFERENCE_COLUMNS, dataset_name="inference")
    return featured


def split_by_time(
    df: pd.DataFrame,
    validation_fraction: float,
    test_fraction: float,
) -> SplitDataset:
    if len(df) < 3:
        raise ValueError("Time-based training requires at least 3 rows to form train, validation, and test splits.")

    ordered = df.sort_values("time").reset_index(drop=True)
    total = len(ordered)
    validation_size, test_size = _resolve_split_sizes(
        total,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    train_end = max(1, total - validation_size - test_size)
    validation_end = min(total, train_end + validation_size)
    return SplitDataset(
        train=ordered.iloc[:train_end].reset_index(drop=True),
        validation=ordered.iloc[train_end:validation_end].reset_index(drop=True),
        test=ordered.iloc[validation_end:].reset_index(drop=True),
    )


def summarize_time_window(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"start": None, "end": None, "rows": 0, "locations": 0}
    ordered = df.sort_values("time")
    return {
        "start": ordered["time"].iloc[0].isoformat(),
        "end": ordered["time"].iloc[-1].isoformat(),
        "rows": int(len(ordered)),
        "locations": int(ordered["location_id"].nunique()) if "location_id" in ordered.columns else 0,
    }


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: set[str],
    dataset_name: str,
) -> None:
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required {dataset_name} dataset columns: {missing_text}")


def _resolve_split_sizes(
    total: int,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[int, int]:
    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("validation_fraction and test_fraction must be non-negative.")

    validation_size = max(1, int(total * validation_fraction))
    test_size = max(1, int(total * test_fraction))

    while validation_size + test_size > total - 1:
        if validation_size >= test_size and validation_size > 1:
            validation_size -= 1
            continue
        if test_size > 1:
            test_size -= 1
            continue
        break

    if validation_size + test_size > total - 1:
        raise ValueError("validation_fraction and test_fraction leave no rows for training.")

    return validation_size, test_size
