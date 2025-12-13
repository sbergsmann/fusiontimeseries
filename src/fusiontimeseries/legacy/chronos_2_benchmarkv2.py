from rich.console import Console

# Initialize the Rich console object once
console = Console()

console.rule("[bold blue]Starting Chronos 2 Benchmarking V2 Script...[/bold blue]")
console.log("[bold blue]Importing Benchmarking Services...[/bold blue]")
from fusiontimeseries.legacy.services.benchmarkerv2 import FluxForecastingBenchmarkerV2  # noqa: E402
from fusiontimeseries.legacy.services.flux_dataset import FluxDataset  # noqa: E402
from fusiontimeseries.legacy.services.flux_trace_provider import FluxTraceProvider  # noqa: E402

console.log("[bold blue]Importing Chronos Library...[/bold blue]")
from chronos import Chronos2Pipeline  # noqa: E402
import torch  # noqa: E402


class Chronos2BenchmarkerV2(FluxForecastingBenchmarkerV2):
    def __init__(self, dataset: FluxDataset, model: str) -> None:
        super().__init__(dataset, model)

        # 1. Visualize Model Loading/Initialization
        with console.status(
            f"[bold cyan]Loading Chronos Model: {model}...[/bold cyan]", spinner="dots"
        ) as _:
            self.pipeline: Chronos2Pipeline = Chronos2Pipeline.from_pretrained(
                pretrained_model_name_or_path=model,
                device_map=self.DEVICE,
                dtype=torch.bfloat16,
            )
        console.log(
            f"[bold green]✅ Model '{model}' Loaded Successfully on {self.DEVICE}.[/bold green]"
        )

    def run_pipeline(self, input: torch.Tensor) -> torch.Tensor:
        """Run the Chronos pipeline on the input tensor.

        Returns normalized predictions.
        """
        # Chronos expects [batch, n_variates, context_length], n_variates=1
        # IMPORTANT: Keep input on CPU - the pipeline will handle device placement
        input_cpu = input.cpu()

        forecast: list[torch.Tensor] = self.pipeline.predict(
            inputs=input_cpu,  # [B, 1, context_length]
            prediction_length=self.dataset.prediction_length,
        )
        # forecast is a list of [1, n_quantiles, prediction_length] tensors
        # Concatenate into [batch, n_quantiles, prediction_length]
        forecast_batch = torch.cat(
            forecast, dim=0
        )  # [batch, n_quantiles, prediction_length]
        # Return as [batch, prediction_length, n_quantiles] (still normalized)
        return forecast_batch.permute(0, 2, 1).to(self.DEVICE)


def main() -> None:
    MODEL = "amazon/chronos-2"
    PREDICTION_LEN = 64
    CONTEXT_LEN = 128
    WINDOW = 32

    # --- Setup and Seeding ---
    console.rule("[bold yellow]Chronos 2 Benchmarking V2 Setup[/bold yellow]")
    console.log(
        f"Model: [green]{MODEL}[/green] | Device: [blue]{Chronos2BenchmarkerV2.DEVICE}[/blue]"
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
    benchmarker = Chronos2BenchmarkerV2(dataset, model=MODEL)
    console.rule("[bold yellow]Benchmarking V2 Process[/bold yellow]")

    console.log(
        "[bold blue]Running on Device:[/bold blue] "
        f"[green]{Chronos2BenchmarkerV2.DEVICE}[/green]"
    )

    # --- Running Benchmark ---
    console.log("[bold red]🔥 Running Core Benchmark V2...[/bold red]")
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
