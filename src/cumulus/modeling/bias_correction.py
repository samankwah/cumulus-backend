"""Bias correction implementations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MeanBiasCorrector:
    scale_factor: float = 1.0

    def fit(self, y_pred: np.ndarray, y_true: np.ndarray) -> "MeanBiasCorrector":
        pred_mean = float(np.mean(y_pred)) if len(y_pred) else 0.0
        true_mean = float(np.mean(y_true)) if len(y_true) else 0.0
        if pred_mean <= 0:
            self.scale_factor = 1.0
        else:
            self.scale_factor = true_mean / pred_mean
        return self

    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(y_pred, dtype=float) * self.scale_factor, a_min=0.0, a_max=None)

    def to_dict(self) -> dict[str, float | str]:
        return {"method": "mean_bias", "scale_factor": self.scale_factor}


@dataclass
class QuantileMappingCorrector:
    predicted_quantiles: np.ndarray | None = None
    observed_quantiles: np.ndarray | None = None
    quantile_count: int = 99

    def fit(self, y_pred: np.ndarray, y_true: np.ndarray) -> "QuantileMappingCorrector":
        quantiles = np.linspace(0.01, 0.99, self.quantile_count)
        predicted_quantiles = np.quantile(y_pred, quantiles)
        observed_quantiles = np.quantile(y_true, quantiles)
        self.predicted_quantiles, self.observed_quantiles = _collapse_duplicate_quantiles(
            predicted_quantiles,
            observed_quantiles,
        )
        return self

    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        if self.predicted_quantiles is None or self.observed_quantiles is None:
            raise ValueError("Corrector must be fitted before transform.")
        if len(self.predicted_quantiles) == 1:
            corrected = np.full_like(np.asarray(y_pred, dtype=float), self.observed_quantiles[0], dtype=float)
        else:
            corrected = np.interp(y_pred, self.predicted_quantiles, self.observed_quantiles)
        return np.clip(corrected, a_min=0.0, a_max=None)

    def to_dict(self) -> dict[str, object]:
        return {
            "method": "quantile_mapping",
            "predicted_quantiles": self.predicted_quantiles.tolist() if self.predicted_quantiles is not None else [],
            "observed_quantiles": self.observed_quantiles.tolist() if self.observed_quantiles is not None else [],
            "quantile_count": self.quantile_count,
        }


def build_bias_corrector(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    min_samples: int,
    quantile_count: int,
) -> MeanBiasCorrector | QuantileMappingCorrector:
    if len(y_pred) < min_samples or len(np.unique(y_pred)) < 5:
        return MeanBiasCorrector().fit(y_pred, y_true)
    return QuantileMappingCorrector(quantile_count=quantile_count).fit(y_pred, y_true)


def _collapse_duplicate_quantiles(
    predicted_quantiles: np.ndarray,
    observed_quantiles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(predicted_quantiles, dtype=float)
    observed = np.asarray(observed_quantiles, dtype=float)
    if len(predicted) == 0:
        return predicted, observed

    unique_predicted, inverse = np.unique(predicted, return_inverse=True)
    if len(unique_predicted) == len(predicted):
        return predicted, observed

    collapsed_observed = np.zeros(len(unique_predicted), dtype=float)
    counts = np.zeros(len(unique_predicted), dtype=float)
    for source_index, target_index in enumerate(inverse):
        collapsed_observed[target_index] += observed[source_index]
        counts[target_index] += 1.0
    collapsed_observed = collapsed_observed / counts
    return unique_predicted, collapsed_observed
