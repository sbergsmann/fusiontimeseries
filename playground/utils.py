"""Utility functions for the notebook playgrounds."""

import torch
from matplotlib import pyplot as plt

__all__ = ["plot_forecast"]


def plot_forecast(
    ctx: torch.Tensor,
    quantile_fc: torch.Tensor,
    real_future_values: torch.Tensor | None = None,
    batch_idx: int = 0,
):
    """Plot a TiRex timeseries forecast.

    Args:
        ctx (torch.Tensor): Timeseries context for zero-shot forecasting.
        quantile_fc (torch.Tensor): Forecast quantiles in shape (N, forecast_len, 9)
        real_future_values (torch.Tensor | None, optional): Actual values to be forecasted. Defaults to None.
        batch_idx (int, optional): Index of the timeseries in the batch to plot. Defaults to 0.
    """
    median_forecast = quantile_fc[batch_idx, :, 4].numpy()
    lower_bound = quantile_fc[batch_idx, :, 0].numpy()
    upper_bound = quantile_fc[batch_idx, :, 8].numpy()
    context = ctx[batch_idx, :].numpy()

    original_x = range(len(context))
    forecast_x = range(len(context), len(context) + len(median_forecast))

    plt.figure(figsize=(12, 6))
    plt.plot(original_x, context, label="Ground Truth Context", color="#4a90d9")
    if real_future_values is not None:
        original_fut_x = range(len(context), len(context) + len(real_future_values))
        plt.plot(
            original_fut_x,
            real_future_values,
            label="Ground Truth Future",
            color="#4a90d9",
            linestyle=":",
        )
    plt.plot(
        forecast_x,
        median_forecast,
        label="Forecast (Median)",
        color="#d94e4e",
        linestyle="--",
    )
    plt.fill_between(
        forecast_x,
        lower_bound,
        upper_bound,
        color="#d94e4e",
        alpha=0.1,
        label="Forecast 10% - 90% Quantiles",
    )
    plt.xlim(left=0)
    plt.legend()
    plt.grid(True)
    plt.show()
