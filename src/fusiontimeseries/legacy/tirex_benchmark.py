from rich.console import Console

# Initialize the Rich console object once
console = Console()

console.rule("[bold blue]Starting TiRex Benchmarking Script...[/bold blue]")
console.log("[bold blue]Importing Benchmarking Services...[/bold blue]")
from fusiontimeseries.legacy.services.benchmarker import FluxForecastingBenchmarker  # noqa: E402
from fusiontimeseries.legacy.services.flux_dataset import FluxDataset  # noqa: E402
from fusiontimeseries.legacy.services.flux_trace_provider import FluxTraceProvider  # noqa: E402

console.log("[bold blue]Importing TiRex Library...[/bold blue]")
from tirex import load_model, ForecastModel  # noqa: E402
import torch  # noqa: E402


class TirexBenchmarker(FluxForecastingBenchmarker):
    def __init__(self, dataset: FluxDataset, model: str) -> None:
        super().__init__(dataset, model)

        # 1. Visualize Model Loading/Initialization
        with console.status(
            f"[bold cyan]Loading TiRex Model: {model}...[/bold cyan]", spinner="dots"
        ) as _:
            self.pipeline: ForecastModel = load_model(
                path=model, device=str(self.DEVICE)
            )  # type: ignore
        console.log(
            f"[bold green]✅ Model '{model}' Loaded Successfully on {self.DEVICE}.[/bold green]"
        )

    def run_pipeline(self, input: torch.Tensor) -> torch.Tensor:
        """Run the TiRex pipeline on the input tensor."""
        quantiles, _ = self.pipeline.forecast(
            # context=[ts.squeeze(0).to(self.DEVICE) for ts in input],
            context=input.squeeze(1).to(
                self.DEVICE
            ),  # [B, 1, context_length] -> [B, context_length]
            prediction_length=self.dataset.prediction_length,
        )
        # quantiles is [batch, prediction_length, n_quantiles]
        return quantiles  # type: ignore


def main() -> None:
    MODEL = "NX-AI/TiRex"
    PREDICTION_LEN = 64
    CONTEXT_LEN = 128
    WINDOW = 32

    # --- Setup and Seeding ---
    console.rule("[bold yellow]TiRex Benchmarking Setup[/bold yellow]")
    console.log(
        f"Model: [green]{MODEL}[/green] | Device: [blue]{TirexBenchmarker.DEVICE}[/blue]"
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
    benchmarker = TirexBenchmarker(dataset, model=MODEL)
    console.rule("[bold yellow]Benchmarking Process[/bold yellow]")

    console.log(
        "[bold blue]Running on Device:[/bold blue] "
        f"[green]{TirexBenchmarker.DEVICE}[/green]"
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
