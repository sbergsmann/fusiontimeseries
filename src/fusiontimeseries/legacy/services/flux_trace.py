from typing import Any, Generator

import torch


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
