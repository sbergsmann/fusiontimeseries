from functools import cache
import os
from pathlib import Path

import numpy as np
import torch

__all__ = ["FluxTraceProvider", "FluxType"]


class FluxType:
    """Enumeration of flux types available in the data files.

    Attributes:
        ELECTRON_FLUX (int): Index for electron flux data (0).
        ENERGY_FLUX (int): Index for energy flux data (1).
        ION_FLUX (int): Index for ion flux data (2).
    """

    ELECTRON_FLUX: int = 0
    ENERGY_FLUX: int = 1
    ION_FLUX: int = 2


class FluxTraceProvider:
    """Data Access Provider for flux traces.

    Provides methods to load and access flux trace data from files.
    Uses caching for efficient data loading and environment variables
    for configuration.

    Attributes:
        FLUX_TYPE (FluxType): Enumeration of available flux types.
        dir (Path): Directory containing the flux data files.
    """

    FLUX_TYPE: FluxType = FluxType()

    def __init__(
        self,
        filename_convention: str = "fluxes_{iteration}.dat",
    ) -> None:
        """Data Access Provider for flux traces

        Args:
            filename_convention (str, optional): The filename convention. Defaults to "fluxes_{iteration}.dat".
        """
        self.filename_convention = filename_convention

        flux_dir_str: str | None = os.getenv("FLUX_TRACE_DIR")
        if flux_dir_str is None:
            raise ValueError(
                "Environment variable FLUX_TRACE_DIR is not set. Please set it to the directory containing flux trace data files."
            )
        self.dir: Path = Path(flux_dir_str)
        os.makedirs(self.dir, exist_ok=True)

    @cache
    def load_flux_energy_data(self, iteration: int) -> torch.Tensor:
        """Load energy flux data for a specific iteration.

        Loads flux data from a file and extracts the energy flux column.
        Results are cached for performance.

        Args:
            iteration (int): The iteration number to load data for.

        Returns:
            torch.Tensor: Energy flux data as a PyTorch tensor.
        """
        file_path: Path = self.dir / self.filename_convention.format(
            iteration=iteration
        )
        data: np.ndarray = np.loadtxt(file_path)
        return torch.from_numpy(data[:, self.FLUX_TYPE.ENERGY_FLUX])

    @cache
    def __len__(self) -> int:
        """Get the number of flux trace files available.

        Returns:
            int: Number of flux trace files.
        """
        return len(os.listdir(self.dir))
