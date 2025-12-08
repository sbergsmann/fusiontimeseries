from torch.utils.data import Dataset

from fusiontimeseries.services.flux_trace import FluxTrace
from fusiontimeseries.services.flux_trace_provider import FluxTraceProvider


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

    def __len__(self) -> int:
        return len(self.provider)

    def __getitem__(self, idx: int) -> FluxTrace:
        trace = self.provider.load_flux_energy_data(idx)
        flux_trace = FluxTrace(
            trace=trace,
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            window=self.window,
        )
        return flux_trace


# Example usage:
# dataset = FluxDataset(
#     provider=fluxtrace_provider,
#     iterations=[0, 1, 2],
#     prediction_length=PREDICTION_LEN,
#     context_length=CONTEXT_LEN,
#     window=10,
# )
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
