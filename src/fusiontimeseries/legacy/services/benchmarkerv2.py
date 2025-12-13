from datetime import datetime
import os
from pathlib import Path
from pydantic import BaseModel, Field
import torch
from rich.progress import Progress
import json

from torch.utils.data import DataLoader

from fusiontimeseries.legacy.services.flux_dataset import FluxDataset
from fusiontimeseries.legacy.services.scaler import Scaler

__all__ = ["FluxForecastingBenchmarkerV2", "BenchmarkV2"]


class SamplePrediction(BaseModel):
    """A single sample prediction with raw data for post-processing."""

    sample_id: int
    context: list[float] = Field(description="Raw context timeseries (not normalized)")
    target: list[float] = Field(description="Raw target timeseries (not normalized)")
    prediction_quantiles: list[list[float]] = Field(
        description="Normalized prediction quantiles [n_quantiles, prediction_length]"
    )
    normalization_mean: float = Field(description="Mean used for normalization")
    normalization_std: float = Field(description="Std used for normalization")


class BenchmarkV2(BaseModel):
    model: str
    prediction_length: int
    context_length: int | None = None
    window: int | None = None
    benchmark_start_timestamp: float | None = Field(
        default=None, description="Posix Timestamp."
    )
    benchmark_end_timestamp: float | None = Field(
        default=None, description="Posix Timestamp."
    )
    samples: list[SamplePrediction] = []


class FluxForecastingBenchmarkerV2:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(
        self,
        dataset: FluxDataset,
        model: str,
        benchmark_file_convention: str = "{timestamp}_benchmarkv2_{model}.json",
    ) -> None:
        self.dataset = dataset
        self.model = model
        self.benchmark_file_convention = benchmark_file_convention

        save_dir_env_var: str | None = os.getenv("BENCHMARK_SAVE_DIR")
        if save_dir_env_var is None:
            raise ValueError(
                "Environment variable BENCHMARK_SAVE_DIR is not set. Please set it to the desired benchmark save directory."
            )
        self.save_dir: Path = Path(save_dir_env_var)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"BenchmarkerV2 will save results to: {self.save_dir}")

        self.samples: list[SamplePrediction] = []
        self.benchmark_start_time: datetime | None = None
        self.benchmark_end_time: datetime | None = None

    def benchmark(self, batch_size: int = 8, stop_after: int | None = None) -> None:
        # reset
        self.samples = []
        self.benchmark_start_time = None
        self.benchmark_end_time = None

        dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=False
        )  # Don't shuffle to maintain sample order

        total = stop_after if stop_after is not None else len(dataloader)
        sample_id = 0

        self.benchmark_start_time = datetime.now()
        with Progress() as progress:
            task = progress.add_task("[cyan]Benchmarking V2...", total=total)

            for idx, batch in enumerate(dataloader):
                if stop_after is not None and idx >= stop_after:
                    break

                ctx_batch, tgt_batch = batch
                ctx_batch: torch.Tensor  # shape: [batch_size, 1, context_length]
                tgt_batch: torch.Tensor  # shape: [batch_size, prediction_length]

                # Normalize context (batch-wise)
                normed_ctx, mean, std = Scaler.setnorm(ctx_batch)

                # Generate forecast (returns normalized predictions)
                with torch.no_grad():
                    forecast: torch.Tensor = self.run_pipeline(
                        normed_ctx
                    )  # shape: [batch_size, prediction_length, n_quantiles]

                # Ensure forecast has correct batch dimension
                if forecast.shape[0] != ctx_batch.shape[0]:
                    print(
                        f"Warning: Forecast batch size {forecast.shape[0]} != input batch size {ctx_batch.shape[0]}"
                    )

                # Store raw data for each sample in the batch
                for i in range(forecast.shape[0]):  # Use forecast batch size
                    # Extract individual samples
                    ctx_raw = ctx_batch[i].squeeze(0).cpu()  # [context_length]
                    tgt_raw = tgt_batch[i].cpu()  # [prediction_length]
                    pred_quantiles = forecast[
                        i
                    ].cpu()  # [prediction_length, n_quantiles]

                    # Store as [n_quantiles, prediction_length] for easier processing
                    pred_quantiles_transposed = pred_quantiles.permute(
                        1, 0
                    ).tolist()  # [n_quantiles, prediction_length]

                    sample = SamplePrediction(
                        sample_id=sample_id,
                        context=ctx_raw.tolist(),
                        target=tgt_raw.tolist(),
                        prediction_quantiles=pred_quantiles_transposed,
                        normalization_mean=mean[i].item(),
                        normalization_std=std[i].item(),
                    )

                    self.samples.append(sample)
                    sample_id += 1

                progress.update(task, advance=1)

        self.benchmark_end_time = datetime.now()

    def save_benchmark(self) -> None:
        """Save the benchmark to a file."""
        benchmark: BenchmarkV2 = BenchmarkV2(
            model=self.model,
            prediction_length=self.dataset.prediction_length,
            context_length=self.dataset.context_length,
            window=self.dataset.window,
            benchmark_start_timestamp=self.benchmark_start_time.timestamp()
            if self.benchmark_start_time
            else None,
            benchmark_end_timestamp=self.benchmark_end_time.timestamp()
            if self.benchmark_end_time
            else None,
            samples=self.samples,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = self.model.replace("/", "_")
        filename = self.benchmark_file_convention.format(
            model=safe_model, timestamp=timestamp
        )

        with open(self.save_dir / filename, "w") as f:
            json.dump(
                benchmark.model_dump(exclude_none=True, exclude_defaults=True),
                f,
                indent=4,
            )

    def run_pipeline(self, ctx: torch.Tensor) -> torch.Tensor:
        """Run the forecasting pipeline on the input tensor.

        Args:
            ctx (torch.Tensor): Context tensor of shape [N, 1, context_length] (normalized).

        Returns:
            torch.Tensor: Forecast tensor of shape [N, prediction_length, n_quantiles] (normalized).
        """
        raise NotImplementedError("Subclasses should implement this method.")
