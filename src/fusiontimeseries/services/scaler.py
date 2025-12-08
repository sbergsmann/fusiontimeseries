import torch

__all__ = ["Scaler"]


class Scaler:
    """Utility class for normalizing and denormalizing time series data."""

    @staticmethod
    def setnorm(ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize the context tensor using z-score normalization.

        Args:
            ctx (torch.Tensor): Input context tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - normed: Normalized tensor.
                - mean: Mean tensor used for normalization.
                - std: Standard deviation tensor used for normalization.
        """
        mean = ctx.mean(dim=-1, keepdim=True)
        std = ctx.std(dim=-1, keepdim=True) + 1e-8
        normed = (ctx - mean) / std
        return normed, mean, std

    @staticmethod
    def denorm(
        pred: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """Denormalize the prediction tensor back to original scale.

        Args:
            pred (torch.Tensor): Normalized prediction tensor.
            mean (torch.Tensor): Mean tensor from normalization.
            std (torch.Tensor): Standard deviation tensor from normalization.


        Returns:
            torch.Tensor: Denormalized prediction tensor.
        """
        return pred * std + mean
