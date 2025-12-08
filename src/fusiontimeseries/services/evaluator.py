import torch


__all__ = ["Evaluator"]


class Evaluator:
    """Evaluator class for computing forecasting metrics."""

    @staticmethod
    def mae(pred: torch.Tensor, tgt: torch.Tensor) -> float:
        """Compute Mean Absolute Error (MAE).

        Args:
            pred (torch.Tensor): Predicted values tensor.
            tgt (torch.Tensor): Target values tensor.

        Returns:
            float: MAE value.
        """
        return torch.mean(torch.abs(pred - tgt)).item()

    @staticmethod
    def rmse(pred: torch.Tensor, tgt: torch.Tensor) -> float:
        """Compute Root Mean Squared Error (RMSE).

        Args:
            pred (torch.Tensor): Predicted values tensor.
            tgt (torch.Tensor): Target values tensor.

        Returns:
            float: RMSE value.
        """
        return torch.sqrt(torch.mean((pred - tgt) ** 2)).item()

    @staticmethod
    def nrmse(pred: torch.Tensor, tgt: torch.Tensor) -> float:
        """Compute Normalized Root Mean Squared Error (NRMSE).

        Normalized by the range of the target (max - min).

        Args:
            pred (torch.Tensor): Predicted values tensor.
            tgt (torch.Tensor): Target values tensor.

        Returns:
            float: NRMSE value.
        """
        rmse_val = torch.sqrt(torch.mean((pred - tgt) ** 2))
        tgt_range = torch.max(tgt) - torch.min(tgt)
        if tgt_range == 0:
            return float("inf")
        return (rmse_val / tgt_range).item()

    @staticmethod
    def nd(pred: torch.Tensor, tgt: torch.Tensor) -> float:
        """Compute Normalized Deviation (ND).

        Sum of absolute errors divided by sum of absolute target values.
        Also known as WAPE (Weighted Absolute Percentage Error).

        Args:
            pred (torch.Tensor): Predicted values tensor.
            tgt (torch.Tensor): Target values tensor.

        Returns:
            float: ND value.
        """
        sum_abs_error = torch.sum(torch.abs(pred - tgt))
        sum_abs_tgt = torch.sum(torch.abs(tgt))
        if sum_abs_tgt == 0:
            return float("inf")
        return (sum_abs_error / sum_abs_tgt).item()

    @staticmethod
    def mape(pred: torch.Tensor, tgt: torch.Tensor) -> float:
        """Compute Mean Absolute Percentage Error (MAPE).

        Args:
            pred (torch.Tensor): Predicted values tensor.
            tgt (torch.Tensor): Target values tensor.

        Returns:
            float: MAPE value as percentage.
        """
        # Avoid division by zero by adding small epsilon where tgt is zero
        epsilon = 1e-8
        safe_tgt = torch.where(tgt == 0, epsilon, tgt)
        return (100.0 * torch.abs(pred - tgt) / torch.abs(safe_tgt)).mean().item()

    @staticmethod
    def smape(pred: torch.Tensor, tgt: torch.Tensor) -> float:
        """Compute Symmetric Mean Absolute Percentage Error (sMAPE).

        Args:
            pred (torch.Tensor): Predicted values tensor.
            tgt (torch.Tensor): Target values tensor.

        Returns:
            float: sMAPE value.
        """
        return (
            (100.0 * torch.abs(pred - tgt) / ((torch.abs(tgt) + torch.abs(pred)) / 2))
            .mean()
            .item()
        )

    @staticmethod
    def mase(
        pred: torch.Tensor, tgt: torch.Tensor, context: torch.Tensor | None = None
    ) -> float:
        """Compute Mean Absolute Scaled Error (MASE).

        Uses naive forecast on context (history) as benchmark if available,
        otherwise uses naive forecast on target (less robust).

        Args:
            pred (torch.Tensor): Predicted values tensor.
            tgt (torch.Tensor): Target values tensor.
            context (torch.Tensor | None): Context/History tensor.

        Returns:
            float: MASE value.
        """
        mae_pred = torch.mean(torch.abs(pred - tgt))

        if context is not None and context.numel() > 1:
            # Use context (history) for naive error scale
            # Calculate mean absolute difference of the context (in-sample naive error)
            ctx = context.squeeze()
            if ctx.ndim > 1:
                ctx = ctx.view(-1)
            scale = torch.mean(torch.abs(ctx[1:] - ctx[:-1]))
        else:
            # Fallback: use target (less robust as it's out-of-sample)
            naive_pred = tgt[:-1]
            naive_tgt = tgt[1:]
            scale = torch.mean(torch.abs(naive_pred - naive_tgt))

        if scale == 0:
            return float("inf")

        return (mae_pred / scale).item()

    @staticmethod
    def directional_accuracy(pred: torch.Tensor, tgt: torch.Tensor) -> float:
        """Compute Directional Accuracy (percentage of correct direction predictions).

        Measures if the predicted direction of change matches the actual direction.

        Args:
            pred (torch.Tensor): Predicted values tensor.
            tgt (torch.Tensor): Target values tensor.

        Returns:
            float: Directional accuracy as percentage (0-100).
        """
        if len(pred) < 2 or len(tgt) < 2:
            return 0.0
        pred_changes = torch.sign(pred[1:] - pred[:-1])
        tgt_changes = torch.sign(tgt[1:] - tgt[:-1])
        correct = torch.sum(pred_changes == tgt_changes).item()
        total = len(pred_changes)
        return (correct / total) * 100.0
