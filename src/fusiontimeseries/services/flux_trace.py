from typing import Any, Generator

import torch

from fusiontimeseries.services.evaluator import Evaluator


class FluxTrace:
    """A class to iterate over time series data in sliding windows for forecasting.

    This class allows generating context and target pairs from a time series trace
    for training or evaluating forecasting models.
    """

    def __init__(
        self,
        trace: torch.Tensor,
        prediction_length: int,
        context_length: int | None = None,
        window: int | None = None,
    ) -> None:
        """Initialize the FluxTrace iterator.

        Args:
            trace (torch.Tensor): The full time series data tensor.
            prediction_length (int): Length of the prediction horizon.
            context_length (int | None): Length of the context window. If None,
                defaults to trace length minus prediction_length.
            window (int | None): Step size for sliding the window. If None,
                defaults to context_length (non-overlapping windows).
        """
        self.trace = trace
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.window = window
        self.metrics: list[dict[str, Any]] = []

    def __iter__(self) -> Generator[tuple[torch.Tensor, torch.Tensor, int], None, None]:
        """Iterate over the trace yielding context and target pairs.

        Yields:
            tuple[torch.Tensor, torch.Tensor]: A tuple of (context, target) tensors.
                - context: Tensor of shape [1, context_length] for model input.
                - target: Tensor of shape [prediction_length] for ground truth.
        """
        trace_length = self.trace.shape[-1]
        context_length = self.context_length or trace_length - self.prediction_length
        start = 0
        end = context_length
        while end + self.prediction_length <= trace_length:
            yield (
                self.trace[..., start:end].unsqueeze(0),
                self.trace[..., end : end + self.prediction_length],
                start,
            )
            start += self.window or context_length
            end += self.window or context_length

    def record(
        self,
        forecast: torch.Tensor,
        target: torch.Tensor,
        context: torch.Tensor | None = None,
        metadata: dict[str, Any] = {},
    ) -> None:
        """Record evaluation metrics for a given forecast and ground truth.

        Args:
            forecast (torch.Tensor): Forecasted values tensor.
            target (torch.Tensor): Ground truth values tensor.
            context (torch.Tensor | None): Context values tensor (history).
            metadata (dict[str, Any]): Additional metadata for the evaluation. Defaults to empty dict.
        """
        mae = Evaluator.mae(forecast, target)
        rmse = Evaluator.rmse(forecast, target)
        nrmse = Evaluator.nrmse(forecast, target)
        nd = Evaluator.nd(forecast, target)
        mape = Evaluator.mape(forecast, target)
        smape = Evaluator.smape(forecast, target)
        mase = Evaluator.mase(forecast, target, context)
        directional_acc = Evaluator.directional_accuracy(forecast, target)

        self.metrics.append(
            {
                "MAE": mae,
                "RMSE": rmse,
                "NRMSE": nrmse,
                "ND": nd,
                "MAPE": mape,
                "sMAPE": smape,
                "MASE": mase,
                "Directional Accuracy": directional_acc,
                "metadata": metadata,
            }
        )

    def forecast_summary(self) -> dict[str, float]:
        """Compute average metrics over all recorded forecasts.

        Returns:
            dict[str, float]: A dictionary with average metrics.
        """
        summary = {
            "MAE": 0.0,
            "RMSE": 0.0,
            "NRMSE": 0.0,
            "ND": 0.0,
            "MAPE": 0.0,
            "sMAPE": 0.0,
            "MASE": 0.0,
            "Directional Accuracy": 0.0,
        }
        n = len(self.metrics)
        if n == 0:
            return summary

        for metric in self.metrics:
            summary["MAE"] += metric["MAE"]
            summary["RMSE"] += metric["RMSE"]
            summary["NRMSE"] += metric["NRMSE"]
            summary["ND"] += metric["ND"]
            summary["MAPE"] += metric["MAPE"]
            summary["sMAPE"] += metric["sMAPE"]
            summary["MASE"] += metric["MASE"]
            summary["Directional Accuracy"] += metric["Directional Accuracy"]

        for key in summary:
            summary[key] /= n

        return summary
