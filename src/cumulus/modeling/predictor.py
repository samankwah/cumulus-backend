"""Prediction utilities."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cumulus.modeling.registry import load_active_model
from cumulus.preprocessing.dataset_builder import build_inference_dataset
from cumulus.settings import Settings
from cumulus.utils.io import read_json


def load_model_bundle(
    settings: Settings,
    *,
    forecast_source: str | None = None,
) -> tuple[object, dict[str, object], dict[str, object]]:
    registry = load_active_model(Path(settings.model_artifact_dir), forecast_source=forecast_source)
    return _load_model_bundle_cached(
        str(registry["model_path"]),
        str(registry["metadata_path"]),
        str(registry["bias_path"]),
    )


@lru_cache(maxsize=8)
def _load_model_bundle_cached(
    model_path: str,
    metadata_path: str,
    bias_path: str,
) -> tuple[object, dict[str, object], dict[str, object]]:
    model = joblib.load(model_path)
    metadata = read_json(Path(metadata_path))
    bias = read_json(Path(bias_path))
    return model, metadata, bias


def clear_model_bundle_cache() -> None:
    _load_model_bundle_cached.cache_clear()


def apply_bias(raw_predictions: np.ndarray, bias_payload: dict[str, object]) -> np.ndarray:
    if bias_payload.get("method") == "mean_bias":
        scale_factor = float(bias_payload.get("scale_factor", 1.0))
        return np.clip(raw_predictions * scale_factor, a_min=0.0, a_max=None)

    predicted_quantiles = np.asarray(bias_payload.get("predicted_quantiles", []), dtype=float)
    observed_quantiles = np.asarray(bias_payload.get("observed_quantiles", []), dtype=float)
    if len(predicted_quantiles) == 0 or len(observed_quantiles) == 0:
        return np.clip(raw_predictions, a_min=0.0, a_max=None)
    return np.clip(np.interp(raw_predictions, predicted_quantiles, observed_quantiles), a_min=0.0, a_max=None)


def predict_dataframe(
    forecast_df: pd.DataFrame,
    settings: Settings,
    *,
    forecast_source: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    model, metadata, bias_payload = load_model_bundle(settings, forecast_source=forecast_source)
    frame = build_inference_dataset(forecast_df)
    feature_columns = metadata["feature_columns"]
    for column in feature_columns:
        if column not in frame.columns:
            frame[column] = 0.0
    raw_predictions = model.predict(frame[feature_columns])
    corrected_predictions = apply_bias(raw_predictions, bias_payload)
    frame["rainfall_raw_mm"] = np.clip(raw_predictions, a_min=0.0, a_max=None)
    frame["rainfall_corrected_mm"] = corrected_predictions
    if forecast_source is not None:
        metadata = {**metadata, "forecast_source": forecast_source}
    return frame, metadata
