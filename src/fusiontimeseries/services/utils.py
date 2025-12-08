import torch

__all__ = ["Utils"]


class Utils:
    @staticmethod
    def median_forecast(forecast: torch.Tensor) -> torch.Tensor:
        """Compute the median forecast from the forecast tensor.

        Args:
            forecast (torch.Tensor): Forecast tensor of shape [N, prediction_length, n_quantiles].

        Returns:
            torch.Tensor: Median forecast tensor of shape [N, prediction_length].
        """
        n_quantiles = forecast.shape[-1]
        median_index = n_quantiles // 2
        return forecast[:, :, median_index]
