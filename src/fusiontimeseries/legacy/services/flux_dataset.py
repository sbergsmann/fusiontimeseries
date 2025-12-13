import torch
from torch.utils.data import Dataset

from fusiontimeseries.legacy.services.flux_trace import FluxTrace
from fusiontimeseries.legacy.services.flux_trace_provider import FluxTraceProvider


class FluxDataset(Dataset):
    """PyTorch Dataset for flux traces using FluxTraceProvider."""

    def __init__(
        self,
        provider: FluxTraceProvider,
        prediction_length: int,
        context_length: int | None = None,
        window: int | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            provider (FluxTraceProvider): The data provider.
            prediction_length (int): Length of prediction horizon.
            context_length (int | None): Length of context window.
            window (int | None): Step size for sliding window.
        """
        self.provider: FluxTraceProvider = provider
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.window = window

        self.samples: list[torch.Tensor]
        self.targets: list[torch.Tensor]
        self.samples, self.targets = self._load_all_traces()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx], self.targets[idx]

    def _load_all_traces(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Load all traces with correct context windows into memory.

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: All samples and targets.
        """
        samples = []
        targets = []
        for idx in range(len(self.provider)):
            trace = self.provider.load_flux_energy_data(idx)
            flux_trace: FluxTrace = FluxTrace(
                trace=trace,
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                window=self.window,
            )
            for ctx, tgt, _ in flux_trace:
                samples.append(ctx)
                targets.append(tgt)
        return samples, targets


# Example usage:
# dataset = FluxDataset(
#     provider=fluxtrace_provider,
#     iterations=[0, 1, 2],
#     prediction_length=PREDICTION_LEN,
#     context_length=CONTEXT_LEN,
#     window=10,
# )
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
