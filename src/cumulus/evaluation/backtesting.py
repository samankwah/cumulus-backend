"""Backtesting and scorecard utilities."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from cumulus.evaluation.framework import DEFAULT_REPORTING_CUTS, build_evaluation_framework
from cumulus.evaluation.metrics import compute_regression_metrics


def evaluate_predictions(
    df: pd.DataFrame,
    *,
    observed_col: str = "rainfall_mm",
    predicted_col: str = "rainfall_corrected_mm",
) -> dict[str, Any]:
    return compute_regression_metrics(df[observed_col].to_numpy(), df[predicted_col].to_numpy())


def build_evaluation_scorecard(
    forecast_df: pd.DataFrame,
    *,
    advisory_df: pd.DataFrame | None = None,
    feedback_df: pd.DataFrame | None = None,
    reference_system: str = "Ghana maize advisory product",
    reference_crop: str = "maize",
    reference_country: str = "Ghana",
    segment_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Build the multi-layer evaluation scorecard defined by the framework."""

    from cumulus.evaluation.metrics import compute_advisory_decision_metrics, compute_farmer_usefulness_metrics

    framework = build_evaluation_framework(
        reference_system=reference_system,
        reference_crop=reference_crop,
        reference_country=reference_country,
    )
    cuts = segment_columns or list(DEFAULT_REPORTING_CUTS)
    scorecard = {
        "framework": framework,
        "reporting_cuts": cuts,
        "layers": {
            "forecast_performance": {
                "status": "available",
                "overall": evaluate_predictions(forecast_df),
                "cuts": _build_cut_metrics(forecast_df, cuts, evaluate_predictions),
            },
            "advisory_decision_performance": _empty_layer(
                "No advisory outcome table provided for decision evaluation."
            ),
            "farmer_usefulness_and_adoption": _empty_layer(
                "No farmer feedback table provided for usefulness and adoption evaluation."
            ),
        },
    }
    if advisory_df is not None and not advisory_df.empty:
        scorecard["layers"]["advisory_decision_performance"] = {
            "status": "available",
            "overall": compute_advisory_decision_metrics(advisory_df),
            "cuts": _build_cut_metrics(advisory_df, cuts, compute_advisory_decision_metrics),
        }
    if feedback_df is not None and not feedback_df.empty:
        scorecard["layers"]["farmer_usefulness_and_adoption"] = {
            "status": "available",
            "overall": compute_farmer_usefulness_metrics(feedback_df),
            "cuts": _build_cut_metrics(feedback_df, cuts, compute_farmer_usefulness_metrics),
        }
    return scorecard


def _build_cut_metrics(
    df: pd.DataFrame,
    cuts: list[str],
    metric_fn: Callable[[pd.DataFrame], dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    cut_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for column in cuts:
        if column not in df.columns:
            continue
        groups: dict[str, dict[str, Any]] = {}
        for value, group in df.groupby(column, dropna=False):
            label = "unknown" if pd.isna(value) else str(value)
            groups[label] = metric_fn(group)
        if groups:
            cut_metrics[column] = groups
    return cut_metrics


def _empty_layer(reason: str) -> dict[str, str]:
    return {
        "status": "not_available",
        "reason": reason,
    }
