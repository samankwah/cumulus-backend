"""Evaluation metrics for rainfall and farmer-advisory systems."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_fscore_support, r2_score

DEFAULT_CALIBRATION_BANDS = (
    ("low", 0.0, 1.0),
    ("medium", 1.0, 10.0),
    ("high", 10.0, None),
)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    wet_day_threshold: float = 1.0,
    heavy_rain_threshold: float = 10.0,
    dry_day_threshold: float = 1.0,
    calibration_bands: Sequence[tuple[str, float, float | None]] = DEFAULT_CALIBRATION_BANDS,
) -> dict[str, Any]:
    """Compute rainfall metrics for both continuous error and operational events."""

    true_values = np.asarray(y_true, dtype=float)
    predicted_values = np.asarray(y_pred, dtype=float)
    wet_true = (true_values >= wet_day_threshold).astype(int)
    wet_pred = (predicted_values >= wet_day_threshold).astype(int)
    wet_precision, wet_recall, wet_f1, _ = precision_recall_fscore_support(
        wet_true,
        wet_pred,
        average="binary",
        zero_division=0,
    )
    heavy_true = (true_values >= heavy_rain_threshold).astype(int)
    heavy_pred = (predicted_values >= heavy_rain_threshold).astype(int)
    _, heavy_recall, _, _ = precision_recall_fscore_support(
        heavy_true,
        heavy_pred,
        average="binary",
        zero_division=0,
    )
    rmse = float(np.sqrt(mean_squared_error(true_values, predicted_values)))
    mae = float(mean_absolute_error(true_values, predicted_values))
    bias = float(np.mean(predicted_values - true_values))
    correlation = _pearson_correlation(true_values, predicted_values)
    observed_longest_dry_run = _longest_dry_run(true_values, dry_day_threshold)
    predicted_longest_dry_run = _longest_dry_run(predicted_values, dry_day_threshold)
    try:
        r2 = float(r2_score(true_values, predicted_values))
    except ValueError:
        r2 = 0.0
    if not np.isfinite(r2):
        r2 = 0.0
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "bias": bias,
        "correlation": correlation,
        "wet_day_precision": float(wet_precision),
        "wet_day_recall": float(wet_recall),
        "wet_day_f1": float(wet_f1),
        "heavy_rain_recall": float(heavy_recall),
        "observed_longest_dry_run_days": float(observed_longest_dry_run),
        "predicted_longest_dry_run_days": float(predicted_longest_dry_run),
        "dry_spell_detection_accuracy": float(predicted_longest_dry_run == observed_longest_dry_run),
        "calibration_by_rainfall_band": compute_calibration_by_rainfall_band(
            true_values,
            predicted_values,
            calibration_bands=calibration_bands,
        ),
    }


def compute_calibration_by_rainfall_band(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    calibration_bands: Sequence[tuple[str, float, float | None]] = DEFAULT_CALIBRATION_BANDS,
) -> dict[str, dict[str, float]]:
    """Compare observed and predicted rainfall frequencies by rainfall band."""

    true_values = np.asarray(y_true, dtype=float)
    predicted_values = np.asarray(y_pred, dtype=float)
    calibration: dict[str, dict[str, float]] = {}
    for label, lower, upper in calibration_bands:
        observed_mask = _band_mask(true_values, lower, upper)
        predicted_mask = _band_mask(predicted_values, lower, upper)
        observed_frequency = float(np.mean(observed_mask)) if len(true_values) else 0.0
        predicted_frequency = float(np.mean(predicted_mask)) if len(predicted_values) else 0.0
        calibration[label] = {
            "observed_frequency": observed_frequency,
            "predicted_frequency": predicted_frequency,
            "frequency_gap": predicted_frequency - observed_frequency,
        }
    return calibration


def compute_advisory_decision_metrics(
    df: pd.DataFrame,
    *,
    recommended_action_col: str = "recommended_action",
    executed_planting_col: str = "planting_executed",
    establishment_success_col: str = "establishment_success",
    observed_window_success_col: str = "observed_window_success",
    delay_avoided_failure_col: str = "delay_avoided_failure",
    advisory_type_col: str = "advisory_type",
    cohort_col: str = "cohort",
    advisory_cohort_label: str = "advisory",
    baseline_cohort_label: str = "baseline",
) -> dict[str, Any]:
    """Compute advisory-decision metrics with planting defaults and optional type splits."""

    metrics = _compute_planting_decision_core_metrics(
        df,
        recommended_action_col=recommended_action_col,
        executed_planting_col=executed_planting_col,
        establishment_success_col=establishment_success_col,
        observed_window_success_col=observed_window_success_col,
        delay_avoided_failure_col=delay_avoided_failure_col,
        cohort_col=cohort_col,
        advisory_cohort_label=advisory_cohort_label,
        baseline_cohort_label=baseline_cohort_label,
    )
    if advisory_type_col in df.columns and not df.empty:
        metrics["by_advisory_type"] = {
            str(advisory_type): _compute_planting_decision_core_metrics(
                group,
                recommended_action_col=recommended_action_col,
                executed_planting_col=executed_planting_col,
                establishment_success_col=establishment_success_col,
                observed_window_success_col=observed_window_success_col,
                delay_avoided_failure_col=delay_avoided_failure_col,
                cohort_col=cohort_col,
                advisory_cohort_label=advisory_cohort_label,
                baseline_cohort_label=baseline_cohort_label,
            )
            for advisory_type, group in df.groupby(advisory_type_col, dropna=False)
        }
    return metrics


def compute_farmer_usefulness_metrics(
    df: pd.DataFrame,
    *,
    clear_enough_to_act_col: str = "clear_enough_to_act",
    followed_advice_col: str = "followed_advice",
    usefulness_score_col: str = "usefulness_score",
    clarity_score_col: str = "clarity_score",
    trust_score_col: str = "trust_score",
    concordance_col: str = "action_matches_advisory",
    recommended_action_col: str = "recommended_action",
    action_taken_col: str = "action_taken",
    outcome_col: str = "outcome_success",
    cohort_col: str = "cohort",
    advisory_cohort_label: str = "advisory",
    baseline_cohort_label: str = "baseline",
) -> dict[str, float | None]:
    """Compute adoption and usefulness metrics from structured farmer feedback."""

    actionability = _rate_from_column(df, clear_enough_to_act_col)
    adoption_rate = _adoption_rate(df[followed_advice_col]) if followed_advice_col in df.columns else None
    perceived_usefulness = _mean_from_columns(
        df,
        [usefulness_score_col, clarity_score_col, trust_score_col],
    )
    if concordance_col in df.columns:
        decision_concordance = _rate(_to_bool_series(df[concordance_col]))
    elif recommended_action_col in df.columns and action_taken_col in df.columns:
        decision_concordance = _rate(
            _normalize_text(df[recommended_action_col]) == _normalize_text(df[action_taken_col])
        )
    else:
        decision_concordance = None
    outcome_lift = _cohort_difference(
        df,
        value_col=outcome_col,
        cohort_col=cohort_col,
        treatment_label=advisory_cohort_label,
        baseline_label=baseline_cohort_label,
    )
    return {
        "advisory_actionability_rate": actionability,
        "adoption_rate": adoption_rate,
        "perceived_usefulness_score": perceived_usefulness,
        "decision_concordance": decision_concordance,
        "outcome_lift": outcome_lift,
    }


def _compute_planting_decision_core_metrics(
    df: pd.DataFrame,
    *,
    recommended_action_col: str,
    executed_planting_col: str,
    establishment_success_col: str,
    observed_window_success_col: str,
    delay_avoided_failure_col: str,
    cohort_col: str,
    advisory_cohort_label: str,
    baseline_cohort_label: str,
) -> dict[str, float | None]:
    recommended_action = _normalize_text(df[recommended_action_col]) if recommended_action_col in df.columns else None
    planted_mask = _to_bool_series(df[executed_planting_col]) if executed_planting_col in df.columns else None
    establishment_success = _to_bool_series(df[establishment_success_col]) if establishment_success_col in df.columns else None
    observed_window_success = (
        _to_bool_series(df[observed_window_success_col]) if observed_window_success_col in df.columns else None
    )
    delay_avoided_failure = (
        _to_bool_series(df[delay_avoided_failure_col]) if delay_avoided_failure_col in df.columns else None
    )

    if recommended_action is not None and planted_mask is None:
        planted_mask = recommended_action == "plant_now"
    if recommended_action is not None and delay_avoided_failure is None and observed_window_success is not None:
        delay_avoided_failure = (recommended_action == "delay") & ~observed_window_success

    plant_now_mask = recommended_action == "plant_now" if recommended_action is not None else None
    delay_mask = recommended_action == "delay" if recommended_action is not None else None
    advised_and_executed = (
        plant_now_mask & planted_mask if plant_now_mask is not None and planted_mask is not None else None
    )

    recommended_success = _subset_rate(establishment_success, advised_and_executed)
    false_go = None if recommended_success is None else 1.0 - recommended_success
    missed_opportunity = _subset_rate(observed_window_success, delay_mask)

    return {
        "planting_success_rate": _subset_rate(establishment_success, planted_mask),
        "recommended_planting_success_rate": recommended_success,
        "delayed_planting_avoided_failure_rate": _subset_rate(delay_avoided_failure, delay_mask),
        "false_go_planting_rate": false_go,
        "missed_opportunity_rate": missed_opportunity,
        "net_planting_uplift": _cohort_difference(
            df,
            value_col=establishment_success_col,
            cohort_col=cohort_col,
            treatment_label=advisory_cohort_label,
            baseline_label=baseline_cohort_label,
        ),
    }


def _pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    if np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _longest_dry_run(values: np.ndarray, threshold: float) -> int:
    longest = 0
    current = 0
    for value in values:
        if value < threshold:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _band_mask(values: np.ndarray, lower: float, upper: float | None) -> np.ndarray:
    mask = values >= lower
    if upper is not None:
        mask &= values < upper
    return mask


def _to_bool_series(values: Sequence[Any] | pd.Series) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) > 0
    normalized = _normalize_text(series)
    return normalized.isin({"1", "true", "yes", "y", "full", "partial", "followed"})


def _normalize_text(values: Sequence[Any] | pd.Series) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    return series.astype("string").fillna("").str.strip().str.lower()


def _rate(mask: Iterable[bool] | np.ndarray | pd.Series) -> float | None:
    series = np.asarray(list(mask) if not isinstance(mask, (np.ndarray, pd.Series)) else mask, dtype=bool)
    if series.size == 0:
        return None
    return float(series.mean())


def _subset_rate(values: pd.Series | None, subset_mask: pd.Series | None) -> float | None:
    if values is None or subset_mask is None:
        return None
    subset_mask = subset_mask.fillna(False).astype(bool)
    if not subset_mask.any():
        return None
    subset = values[subset_mask]
    if subset.empty:
        return None
    return float(subset.mean())


def _rate_from_column(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None
    return _rate(_to_bool_series(df[column]))


def _adoption_rate(values: pd.Series) -> float | None:
    normalized = _normalize_text(values)
    adopted = normalized.isin({"1", "true", "yes", "y", "full", "partial", "followed"})
    return _rate(adopted)


def _mean_from_columns(df: pd.DataFrame, columns: Sequence[str]) -> float | None:
    available = [column for column in columns if column in df.columns]
    if not available:
        return None
    values = df[available].apply(pd.to_numeric, errors="coerce")
    flattened = values.to_numpy(dtype=float).ravel()
    flattened = flattened[~np.isnan(flattened)]
    if flattened.size == 0:
        return None
    return float(flattened.mean())


def _cohort_difference(
    df: pd.DataFrame,
    *,
    value_col: str,
    cohort_col: str,
    treatment_label: str,
    baseline_label: str,
) -> float | None:
    if value_col not in df.columns or cohort_col not in df.columns:
        return None
    cohorts = _normalize_text(df[cohort_col])
    treatment_mask = cohorts == treatment_label.lower()
    baseline_mask = cohorts == baseline_label.lower()
    if not treatment_mask.any() or not baseline_mask.any():
        return None
    values = _to_bool_series(df[value_col]).astype(float)
    return float(values[treatment_mask].mean() - values[baseline_mask].mean())
