from __future__ import annotations

import numpy as np


def forecast_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or not np.isfinite(predicted).all():
        raise ValueError("metrics require finite, equally shaped forecasts")
    error = actual - predicted
    denominator = np.abs(actual) + np.abs(predicted)
    smape_terms = np.divide(200 * np.abs(error), denominator, out=np.zeros_like(error), where=denominator > 0)
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2))), "smape": float(np.mean(smape_terms))}


def trainable_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

