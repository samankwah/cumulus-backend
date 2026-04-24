"""Reporting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from cumulus.evaluation.framework import build_evaluation_framework, render_evaluation_framework_markdown
from cumulus.utils.io import ensure_directory, write_json


def write_evaluation_report(output_dir: Path, metrics: dict[str, float], predictions: pd.DataFrame) -> None:
    ensure_directory(output_dir)
    write_json(output_dir / "metrics.json", metrics)
    predictions.to_csv(output_dir / "predictions.csv", index=False)


def write_scorecard_report(output_dir: Path, scorecard: dict[str, Any]) -> None:
    """Persist the evaluation scorecard alongside a Markdown framework brief."""

    ensure_directory(output_dir)
    write_json(output_dir / "scorecard.json", scorecard)
    framework = scorecard.get("framework") or build_evaluation_framework()
    (output_dir / "framework.md").write_text(
        render_evaluation_framework_markdown(framework),
        encoding="utf-8",
    )
