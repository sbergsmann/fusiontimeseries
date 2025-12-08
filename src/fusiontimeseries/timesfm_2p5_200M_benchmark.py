from rich.console import Console

# Initialize the Rich console object once
console = Console()

console.rule("[bold blue]Starting TimesFM 2.5 200M Benchmarking Script...[/bold blue]")
console.log("[bold blue]Importing Benchmarking Services...[/bold blue]")
from fusiontimeseries.services.benchmarker import FluxForecastingBenchmarker  # noqa: E402
from fusiontimeseries.services.flux_dataset import FluxDataset  # noqa: E402
from fusiontimeseries.services.flux_trace_provider import FluxTraceProvider  # noqa: E402

console.log("[bold blue]Importing TimesFM Library...[/bold blue]")
import timesfm  # noqa: E402
import torch  # noqa: E402


class TimesFmBenchmarker(FluxForecastingBenchmarker):
    def __init__(self, dataset: FluxDataset, model: str) -> None:
        super().__init__(dataset, model)

        # 1. Visualize Model Loading/Initialization
        with console.status(
            f"[bold cyan]Loading TimesFM Model: {model}...[/bold cyan]", spinner="dots"
        ) as _:
            self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                model, torch_compile=True
            )
            self.model.compile(
                timesfm.ForecastConfig(
                    max_context=1024,
                    max_horizon=256,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                )
            )
        console.log(
            f"[bold green]✅ Model '{model}' Loaded Successfully on {self.DEVICE}.[/bold green]"
        )

    def run_pipeline(self, input: torch.Tensor) -> torch.Tensor:
        """Run the TimesFM pipeline on the input tensor."""
        # TimesFM expects list of 1D tensors
        point_forecast, quantile_forecast = self.model.forecast(
            horizon=self.dataset.prediction_length,
            inputs=[input.squeeze(0)],  # list of [context_length]  # type: ignore
        )
        # quantile_forecast is [batch, prediction_length, n_quantiles]
        # Exclude the mean (first quantile), keep 10th to 90th
        return quantile_forecast[:, :, 1:]  # [1, prediction_length, 9]  # type: ignore


def main() -> None:
    MODEL = "google/timesfm-2.5-200m-pytorch"
    PREDICTION_LEN = 64
    CONTEXT_LEN = 128
    WINDOW = 32

    # --- Setup and Seeding ---
    console.rule("[bold yellow]TimesFM Benchmarking Setup[/bold yellow]")
    console.log(
        f"Model: [green]{MODEL}[/green] | Device: [blue]{TimesFmBenchmarker.DEVICE}[/blue]"
    )
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    console.log("Random seeds set to 42.")

    # --- Data Provider Initialization ---
    with console.status(
        "[bold cyan]Initializing FluxTraceProvider...[/bold cyan]", spinner="point"
    ):
        fluxtrace_provider = FluxTraceProvider()
    console.log("[bold green]✅ FluxTraceProvider Initialized.[/bold green]")

    # --- Dataset Initialization (The long-running part) ---
    with console.status(
        "[bold magenta]⏳ Loading and Preprocessing Flux Dataset...[/bold magenta]",
        spinner="line",
    ):
        # Assuming FluxDataset does the heavy data loading/fetching in its __init__
        dataset = FluxDataset(
            provider=fluxtrace_provider,
            prediction_length=PREDICTION_LEN,
            context_length=CONTEXT_LEN,
            window=WINDOW,
        )
    console.log(
        f"[bold green]✅ Dataset Loaded:[/bold green] [yellow]{len(dataset)}[/yellow] samples available."
    )

    # --- Benchmarker Initialization (Triggers Model Loading) ---
    benchmarker = TimesFmBenchmarker(dataset, model=MODEL)
    console.rule("[bold yellow]Benchmarking Process[/bold yellow]")

    console.log(
        "[bold blue]Running on Device:[/bold blue] "
        f"[green]{TimesFmBenchmarker.DEVICE}[/green]"
    )

    # --- Running Benchmark ---
    console.log("[bold red]🔥 Running Core Benchmark...[/bold red]")
    benchmarker.benchmark()
    console.log("[bold green]✅ Benchmark Run Complete.[/bold green]")

    # --- Saving Results ---
    with console.status(
        "[bold white]💾 Saving Benchmark Results...[/bold white]", spinner="star"
    ):
        benchmarker.save_benchmark()
    console.log("[bold green]✅ Results Saved. Benchmarking Finished![/bold green]")
    console.rule("[bold yellow]DONE[/bold yellow]")


if __name__ == "__main__":
    main()
