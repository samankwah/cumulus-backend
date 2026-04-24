"""Training logic for the baseline rainfall model."""

from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from cumulus.evaluation.metrics import compute_regression_metrics
from cumulus.modeling.bias_correction import MeanBiasCorrector, QuantileMappingCorrector, build_bias_corrector
from cumulus.modeling.registry import save_active_model
from cumulus.preprocessing.dataset_builder import build_training_dataset, split_by_time
from cumulus.settings import Settings
from cumulus.utils.io import write_json

PRIMARY_METRIC_NAMES = ("rmse", "mae")
COMPARISON_METHODS = ("raw", "mean_bias", "quantile_mapping")
PERCENTILE_LEVELS = (
    ("p01", 0.01),
    ("p05", 0.05),
    ("p10", 0.10),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
    ("p95", 0.95),
    ("p99", 0.99),
)
CDF_POINT_COUNT = 101


def train_baseline_model(
    merged_df: pd.DataFrame,
    settings: Settings,
    *,
    forecast_source: str | None = None,
) -> dict[str, object]:
    dataset = build_training_dataset(merged_df)
    splits = split_by_time(
        dataset,
        validation_fraction=settings.train.validation_fraction,
        test_fraction=settings.train.test_fraction,
    )
    split_ranges = splits.date_ranges()

    feature_columns = [column for column in settings.feature_columns if column in dataset.columns]
    if not feature_columns:
        raise ValueError("No configured feature columns are present in the training dataset.")

    model = RandomForestRegressor(**settings.model.params.model_dump())
    model.fit(splits.train[feature_columns], splits.train[settings.model.target_column])

    validation_predictions = model.predict(splits.validation[feature_columns])
    validation_targets = splits.validation[settings.model.target_column].to_numpy()
    mean_bias_corrector = MeanBiasCorrector().fit(validation_predictions, validation_targets)
    quantile_mapping_corrector = QuantileMappingCorrector(
        quantile_count=settings.bias_correction.quantile_count,
    ).fit(validation_predictions, validation_targets)
    selected_corrector = build_bias_corrector(
        validation_predictions,
        validation_targets,
        min_samples=settings.bias_correction.calibration_min_samples,
        quantile_count=settings.bias_correction.quantile_count,
    )

    test_raw_predictions = model.predict(splits.test[feature_columns])
    test_targets = splits.test[settings.model.target_column].to_numpy()
    method_predictions = {
        "raw": np.clip(np.asarray(test_raw_predictions, dtype=float), a_min=0.0, a_max=None),
        "mean_bias": mean_bias_corrector.transform(test_raw_predictions),
        "quantile_mapping": quantile_mapping_corrector.transform(test_raw_predictions),
    }
    selected_method = selected_corrector.to_dict()["method"]
    selected_predictions = method_predictions[selected_method]
    method_metrics = {
        method: compute_regression_metrics(test_targets, predictions)
        for method, predictions in method_predictions.items()
    }
    metrics = method_metrics[selected_method]
    raw_metrics = method_metrics["raw"]
    bias_comparison = _build_bias_comparison(
        method_metrics=method_metrics,
        selected_method=selected_method,
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    source_tag = f"{forecast_source}_" if forecast_source else ""
    model_version = f"rf_rainfall_{source_tag}{timestamp}"
    model_dir = Path(settings.model_artifact_dir) / model_version
    bias_dir = Path(settings.bias_artifact_dir) / model_version
    evaluation_dir = Path(settings.evaluation_dir) / model_version
    model_dir.mkdir(parents=True, exist_ok=True)
    bias_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    feature_importances = _serialize_feature_importances(feature_columns, model.feature_importances_)
    bias_payload = selected_corrector.to_dict()
    predictions_path = evaluation_dir / "test_predictions.csv"
    metrics_path = evaluation_dir / "metrics.json"
    distribution_summary_path = evaluation_dir / "distribution_summary.csv"
    distribution_summary_json_path = evaluation_dir / "distribution_summary.json"
    predictions_frame = _build_predictions_frame(
        splits.test,
        target_column=settings.model.target_column,
        method_predictions=method_predictions,
        selected_method=selected_method,
    )
    predictions_frame.to_csv(predictions_path, index=False)
    distribution_summary_frame, distribution_summary_payload = _build_distribution_summary(
        observed=test_targets,
        method_predictions=method_predictions,
    )
    distribution_summary_frame.to_csv(distribution_summary_path, index=False)

    metrics_payload = {
        "model_version": model_version,
        "bias_method": bias_payload["method"],
        "selected_bias_method": selected_method,
        "primary_metrics": {
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
        },
        "metrics": metrics,
        "raw_metrics": raw_metrics,
        "methods": {method: {"metrics": values} for method, values in method_metrics.items()},
        "improvement_vs_raw": bias_comparison["improvement_vs_raw"],
        "date_ranges": split_ranges,
    }
    write_json(metrics_path, metrics_payload)
    write_json(
        distribution_summary_json_path,
        {
            "model_version": model_version,
            "selected_bias_method": selected_method,
            **distribution_summary_payload,
        },
    )

    artifact = {
        "model_version": model_version,
        "forecast_source": forecast_source or "configured",
        "feature_columns": feature_columns,
        "target_column": settings.model.target_column,
        "estimator": settings.model.estimator,
        "bias_method": bias_payload["method"],
        "selected_bias_method": selected_method,
        "spatial_resolution_km": settings.data_pipeline.target_resolution_km,
        "metrics": metrics,
        "raw_metrics": raw_metrics,
        "bias_comparison": bias_comparison,
        "date_ranges": split_ranges,
        "feature_importances": feature_importances,
        "top_feature_importances": feature_importances[:10],
        "evaluation_paths": {
            "metrics_json": str(metrics_path),
            "predictions_csv": str(predictions_path),
            "distribution_summary_csv": str(distribution_summary_path),
            "distribution_summary_json": str(distribution_summary_json_path),
        },
        "created_at": timestamp,
    }
    joblib.dump(model, model_dir / "model.joblib")
    write_json(model_dir / "metadata.json", artifact)
    write_json(bias_dir / "bias.json", bias_payload)
    save_active_model(
        Path(settings.model_artifact_dir),
        {
            "model_version": model_version,
            "forecast_source": forecast_source or "configured",
            "model_path": str(model_dir / "model.joblib"),
            "metadata_path": str(model_dir / "metadata.json"),
            "bias_path": str(bias_dir / "bias.json"),
        },
        forecast_source=forecast_source,
    )
    return {
        "model_version": model_version,
        "forecast_source": forecast_source or "configured",
        "bias_method": bias_payload["method"],
        "selected_bias_method": selected_method,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "metrics": metrics,
        "raw_metrics": raw_metrics,
        "bias_comparison": bias_comparison,
        "date_ranges": split_ranges,
        "evaluation_paths": artifact["evaluation_paths"],
    }


def _serialize_feature_importances(
    feature_columns: list[str],
    importances: Any,
) -> list[dict[str, float | str]]:
    ranked = [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in zip(feature_columns, np.asarray(importances, dtype=float), strict=True)
    ]
    ranked.sort(key=lambda item: item["importance"], reverse=True)
    return ranked


def _build_predictions_frame(
    test_df: pd.DataFrame,
    target_column: str,
    method_predictions: dict[str, np.ndarray],
    selected_method: str,
) -> pd.DataFrame:
    frame = test_df[["location_id", "time", target_column]].copy()
    frame = frame.rename(columns={target_column: "rainfall_mm"})
    frame["rainfall_raw_mm"] = np.asarray(method_predictions["raw"], dtype=float)
    frame["rainfall_mean_bias_mm"] = np.asarray(method_predictions["mean_bias"], dtype=float)
    frame["rainfall_quantile_mapping_mm"] = np.asarray(method_predictions["quantile_mapping"], dtype=float)
    frame["rainfall_corrected_mm"] = np.asarray(method_predictions[selected_method], dtype=float)
    frame["residual_raw_mm"] = frame["rainfall_raw_mm"] - frame["rainfall_mm"]
    frame["residual_mean_bias_mm"] = frame["rainfall_mean_bias_mm"] - frame["rainfall_mm"]
    frame["residual_quantile_mapping_mm"] = frame["rainfall_quantile_mapping_mm"] - frame["rainfall_mm"]
    frame["residual_mm"] = frame["rainfall_corrected_mm"] - frame["rainfall_mm"]
    return frame


def _build_bias_comparison(
    method_metrics: dict[str, dict[str, float]],
    selected_method: str,
) -> dict[str, object]:
    methods = {
        method: {"metrics": metrics}
        for method, metrics in method_metrics.items()
    }
    improvement_vs_raw = {
        method: {
            "rmse_delta": float(method_metrics["raw"]["rmse"] - metrics["rmse"]),
            "mae_delta": float(method_metrics["raw"]["mae"] - metrics["mae"]),
            "mean_bias_delta": float(abs(method_metrics["raw"]["bias"]) - abs(metrics["bias"])),
        }
        for method, metrics in method_metrics.items()
        if method != "raw"
    }
    return {
        "selected_bias_method": selected_method,
        "methods": methods,
        "improvement_vs_raw": improvement_vs_raw,
        "primary_metrics": list(PRIMARY_METRIC_NAMES),
    }


def _build_distribution_summary(
    observed: np.ndarray,
    method_predictions: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, object]]:
    observed_array = np.asarray(observed, dtype=float)
    quantile_rows: list[dict[str, float | str]] = []
    for label, level in PERCENTILE_LEVELS:
        row: dict[str, float | str] = {
            "summary_type": "percentile",
            "statistic": label,
            "probability": float(level),
            "rainfall_mm": None,
            "observed": float(np.quantile(observed_array, level)),
        }
        for method in COMPARISON_METHODS:
            row[method] = float(np.quantile(np.asarray(method_predictions[method], dtype=float), level))
        quantile_rows.append(row)

    all_values = [observed_array, *(np.asarray(method_predictions[method], dtype=float) for method in COMPARISON_METHODS)]
    max_value = max(float(np.max(values)) for values in all_values) if all_values else 0.0
    cdf_grid = np.linspace(0.0, max_value, num=CDF_POINT_COUNT) if max_value > 0 else np.zeros(CDF_POINT_COUNT, dtype=float)
    cdf_rows: list[dict[str, float | str]] = []
    for rainfall_mm in cdf_grid:
        row = {
            "summary_type": "ecdf",
            "statistic": "",
            "probability": None,
            "rainfall_mm": float(rainfall_mm),
            "observed": float(np.mean(observed_array <= rainfall_mm)),
        }
        for method in COMPARISON_METHODS:
            row[method] = float(np.mean(np.asarray(method_predictions[method], dtype=float) <= rainfall_mm))
        cdf_rows.append(row)

    summary_frame = pd.DataFrame(quantile_rows + cdf_rows)
    payload = {
        "quantiles": [
            {
                "percentile": row["statistic"],
                "probability": row["probability"],
                "observed": row["observed"],
                "raw": row["raw"],
                "mean_bias": row["mean_bias"],
                "quantile_mapping": row["quantile_mapping"],
            }
            for row in quantile_rows
        ],
        "ecdf": [
            {
                "rainfall_mm": row["rainfall_mm"],
                "observed": row["observed"],
                "raw": row["raw"],
                "mean_bias": row["mean_bias"],
                "quantile_mapping": row["quantile_mapping"],
            }
            for row in cdf_rows
        ],
    }
    return summary_frame, payload
