from datetime import datetime
import os
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field
import torch
from rich.progress import Progress
import json

from torch.utils.data import DataLoader

from fusiontimeseries.services.flux_dataset import FluxDataset
from fusiontimeseries.services.flux_trace import FluxTrace
from fusiontimeseries.services.scaler import Scaler
from fusiontimeseries.services.utils import Utils

__all__ = ["FluxForecastingBenchmarker", "Benchmark"]


class Benchmark(BaseModel):
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
    metrics: list[dict[str, Any]] = []


class FluxForecastingBenchmarker:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(
        self,
        dataset: FluxDataset,
        model: str,
        benchmark_file_convention: str = "{timestamp}_benchmark_{model}.json",
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
        print(f"Benchmarker will save results to: {self.save_dir}")

        self.metrics: list[dict[str, Any]] = []
        self.benchmark_start_time: datetime | None = None
        self.benchmark_end_time: datetime | None = None

    def benchmark(self, batch_size: int = 1, stop_after: int | None = None) -> None:
        # reset
        self.metrics = []
        self.benchmark_start_time = None
        self.benchmark_end_time = None

        dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x[0]
        )

        total = stop_after if stop_after is not None else len(dataloader)

        self.benchmark_start_time = datetime.now()
        with Progress() as progress:
            task = progress.add_task("[cyan]Benchmarking...", total=total)

            for idx, fluxtrace in enumerate(dataloader):
                fluxtrace: FluxTrace
                if stop_after is not None and idx >= stop_after:
                    break

                for ctx, tgt, _ in fluxtrace:
                    ctx: torch.Tensor = ctx.to(self.DEVICE)
                    tgt: torch.Tensor = tgt.to(self.DEVICE)
                    _: int

                    # Normalize context
                    normed_ctx, mean, std = Scaler.setnorm(ctx)

                    # Generate forecast
                    with torch.no_grad():
                        forecast: torch.Tensor = self.run_pipeline(
                            normed_ctx
                        )  # shape: [N, prediction_length, n_quantiles]

                    # Denormalize forecast
                    denormed_forecast = Scaler.denorm(forecast, mean, std)

                    fluxtrace.record(
                        forecast=Utils.median_forecast(denormed_forecast).squeeze(0),
                        target=tgt.squeeze(0),
                        context=ctx.squeeze(0),
                    )

                self.metrics.append(fluxtrace.forecast_summary())
                progress.update(task, advance=1)

        self.benchmark_end_time = datetime.now()

    def save_benchmark(self) -> None:
        """Save the benchmark to a file."""
        benchmark: Benchmark = Benchmark(
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
            metrics=self.metrics,
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
            ctx (torch.Tensor): Context tensor of shape [N, context_length].

        Returns:
            torch.Tensor: Forecast tensor of shape [N, prediction_length].
        """
        raise NotImplementedError("Subclasses should implement this method.")
