"""Evaluation layer."""

from cumulus.evaluation.backtesting import build_evaluation_scorecard, evaluate_predictions
from cumulus.evaluation.framework import build_evaluation_framework, render_evaluation_framework_markdown
from cumulus.evaluation.metrics import (
    compute_advisory_decision_metrics,
    compute_farmer_usefulness_metrics,
    compute_regression_metrics,
)
from cumulus.evaluation.reports import write_evaluation_report, write_scorecard_report

__all__ = [
    "build_evaluation_framework",
    "build_evaluation_scorecard",
    "compute_advisory_decision_metrics",
    "compute_farmer_usefulness_metrics",
    "compute_regression_metrics",
    "evaluate_predictions",
    "render_evaluation_framework_markdown",
    "write_evaluation_report",
    "write_scorecard_report",
]
