# TimeseriesDataset

from dataclasses import dataclass
import json
from typing import Any, Callable, Iterator, Literal, Self

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import IterableDataset
import pandas as pd
from numpy.typing import NDArray
import numpy as np

from fusiontimeseries.lib.config import OP_NAMES, T_OP, FTSConfig


ID_BENCHMARK_IDXS: list[int] = [8, 115, 131, 148, 235, 262]


@dataclass
class FluxData:
    idx: int
    distribution: Literal["id", "ood"]

    # operating parameters
    shat: float  # magnetic sheering (current in plasma)
    q: float  # safety factor, how often a particle takes a polodial turn before a torodial turn
    rlt: float  # ion temperature gradient, main driver of turbulence
    rln: float  # density gradient

    # simulation timeseries data
    energy_flux: list[float]

    @property
    def is_benchmark(self) -> bool:
        return self.distribution == "ood" or self.idx in ID_BENCHMARK_IDXS

    @property
    def operating_parameters(self) -> NDArray:
        return np.array([self.shat, self.q, self.rlt, self.rln])

    def copy(self, **kwargs) -> "FluxData":
        new_data: dict[str, Any] = self.__dict__.copy()

        for k, v in kwargs.items():
            new_data[k] = v

        return FluxData(**new_data)


def get_train_flux_traces(
    flux_data: list[FluxData],
    subsample_factor: int,
    full_subsampling: bool,
    drop_ids: list[int] = [],
) -> dict[int, FluxData]:
    """Get all valid and subsampled flux traces.

    Args:
        full_subsampling (bool, optional): Whether to use subsampling for all windows. Defaults to False.

    Returns:
        dict[int, np.ndarray]: The dictionary of flux traces.
    """
    HORIZON: int = 240  # head and tail length to consider for mean flux

    valid_flux_traces: dict[int, FluxData] = {}
    incremental_idx: int = 0
    for _flux_data in flux_data:
        _flux_data: FluxData

        # filter out benchmark traces
        if _flux_data.is_benchmark or _flux_data.idx in drop_ids:
            continue

        flux: NDArray[np.float32] = np.array(_flux_data.energy_flux, dtype=np.float32)

        # Step 1: Check mean flux at head and tail
        mean_head: float = float(np.mean(flux[:HORIZON]))
        mean_tail: float = float(np.mean(flux[-HORIZON:]))
        if not (1.0 <= mean_head <= np.inf) or not (1.0 <= mean_tail <= np.inf):
            continue

        # Step 2: Subsample
        subsampled_flux: FluxData = _flux_data.copy(
            energy_flux=flux[::subsample_factor].tolist(), idx=incremental_idx
        )
        if full_subsampling:
            for partial in range(1, subsample_factor + 1):
                idx: int = (incremental_idx + 1) * subsample_factor - partial
                valid_flux_traces[idx] = subsampled_flux.copy(
                    idx=idx,
                    energy_flux=flux[
                        partial % subsample_factor :: subsample_factor
                    ].tolist(),
                )
        else:
            valid_flux_traces[incremental_idx] = subsampled_flux
        incremental_idx += 1

    return valid_flux_traces


class FTSDataProcessingMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

    @staticmethod
    def stratify_by_mean(
        flux_data: dict[int, FluxData],
        config: FTSConfig,
    ):
        # stratify all flux data time series by mean energy flux
        energy_flux_means: list[np.floating] = [
            np.mean(flux_data.energy_flux[-config.pred_tail_timestamps :])
            for flux_data in flux_data.values()
        ]

        # Use quantile-based binning to create equally-sized bins
        stratify_labels = pd.qcut(
            energy_flux_means,  # type: ignore
            q=config.stratification_bins,
            labels=False,
            duplicates="drop",
        )

        train_flux_data, val_flux_data = train_test_split(
            list(flux_data.values()),
            test_size=config.val_size,
            random_state=config.random_seed,
            stratify=stratify_labels,
        )
        return train_flux_data, val_flux_data

    @staticmethod
    def stratify_by_opc_pca_kmeans(
        flux_data: dict[int, FluxData],
        config: FTSConfig,
    ):
        opcs: np.ndarray = np.array(
            [flux_data.operating_parameters for flux_data in flux_data.values()]
        )

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(opcs)  # shape (nr_time_series, 2)
        kmeans = KMeans(
            n_clusters=config.stratification_bins,
            random_state=config.random_seed,
        )
        stratify_labels = kmeans.fit_predict(pca_result)  # shape (nr_time_series,)

        train_flux_data, val_flux_data = train_test_split(
            list(flux_data.values()),
            test_size=config.val_size,
            random_state=config.random_seed,
            stratify=stratify_labels,
        )
        return train_flux_data, val_flux_data

    @classmethod
    def load_flux_data(cls, config: FTSConfig) -> list[FluxData]:
        _flux_data: list[FluxData] = [
            FluxData(**sample) for sample in json.load(open(config.data_path, "r"))
        ]
        return _flux_data

    @classmethod
    def train_val_split(
        cls,
        config: FTSConfig,
    ) -> tuple[Self, Self]:
        _flux_data: list[FluxData] = cls.load_flux_data(config)

        flux_data: dict[int, FluxData] = get_train_flux_traces(
            flux_data=_flux_data,
            full_subsampling=config.full_subsampling,
            subsample_factor=config.subsample_factor,
            drop_ids=config.drop_ids,
        )

        strat_fn: Callable[
            [dict[int, FluxData], FTSConfig],
            tuple[list[FluxData], list[FluxData]],
        ]
        match config.stratification:
            case "tail_mean":
                strat_fn = cls.stratify_by_mean
            case "opc_pca":
                strat_fn = cls.stratify_by_opc_pca_kmeans
            case _:
                raise ValueError(
                    f"Unknown stratification method: {config.stratification}"
                )
        train_flux_data, val_flux_data = strat_fn(flux_data, config)
        print(
            f"Train/Val split: {len(train_flux_data)} / {len(val_flux_data)} time series."
        )

        train_time_series = [np.array(fd.energy_flux) for fd in train_flux_data]
        train_ops = np.array(
            [fd.operating_parameters for fd in train_flux_data],
        )
        train_dataset = cls(
            time_series=train_time_series,
            operating_parameters=train_ops,
            config=config,
            mode="train",
        )

        val_time_series = [np.array(fd.energy_flux) for fd in val_flux_data]
        val_ops = np.array(
            [fd.operating_parameters for fd in val_flux_data],
        )
        val_dataset = cls(
            time_series=val_time_series,
            operating_parameters=val_ops,
            config=config,
            mode="val",
        )

        return train_dataset, val_dataset

    @classmethod
    def get_benchmark_flux_traces(
        cls,
        config: FTSConfig,
    ) -> dict[Literal["ood", "id"], dict[int, FluxData]]:
        """Get benchmark flux traces for in-distribution and out-of-distribution.

        Returns:
            dict[str, dict[int, FluxData]]: The dictionary of benchmark flux traces with shape (266,).
        """

        benchmark_flux_traces: dict[Literal["ood", "id"], dict[int, FluxData]] = {}
        _flux_data = cls.load_flux_data(config)

        id_flux_traces: list[FluxData] = [
            entry
            for entry in _flux_data
            if entry.distribution == "id" and entry.is_benchmark
        ]
        benchmark_flux_traces["id"] = {
            flux_data.idx: flux_data.copy(
                energy_flux=flux_data.energy_flux[:: config.subsample_factor]
            )
            for flux_data in id_flux_traces
        }

        ood_flux_traces: list[FluxData] = [
            entry for entry in _flux_data if entry.distribution == "ood"
        ]
        benchmark_flux_traces["ood"] = {
            flux_data.idx: flux_data.copy(
                energy_flux=flux_data.energy_flux[:: config.subsample_factor]
            )
            for flux_data in ood_flux_traces
        }

        return benchmark_flux_traces


class TimeseriesDataset(IterableDataset, FTSDataProcessingMixin):
    def __init__(
        self,
        time_series: list[np.ndarray],
        operating_parameters: np.ndarray,
        config: FTSConfig,
        mode: Literal["train", "val"],
    ) -> None:
        self.time_series = time_series
        self.config = config
        self.mode = mode
        self.ops = operating_parameters

        np.random.seed(self.config.random_seed)
        match self.config.sampling_strategy:
            case "linear":
                self.prepare_opc_stratified_sampling_linear()
            case "pca_cluster":
                self.prepare_opc_stratified_sampling_pca_cluster()

        self.ops_cov = np.cov(self.ops, rowvar=False)
        self.ops_std = np.sqrt(np.diag(self.ops_cov))
        self.ops_corr = self.ops_cov / np.outer(self.ops_std, self.ops_std)
        self.ops_beta = 0.05
        self.ts_alpha = 0.01

    def prepare_opc_stratified_sampling_linear(self):
        # assign each time series to a bin based on each operating parameter
        self.bin_assignments: list[np.ndarray] = []
        for index, param in enumerate(OP_NAMES):
            conditioning_parameter = self.ops[:, index]
            bin_assignment = self._assign_bins(conditioning_parameter, param)
            self.bin_assignments.append(bin_assignment)

    def _assign_bins(
        self, conditioning_parameter: np.ndarray, param: T_OP
    ) -> np.ndarray:
        # assign each time series index into bins of equal size based on the operating parameter value range
        nr_bins: int = self.config.sampling_bins
        bins = np.linspace(
            self.config.op_ranges[param][0],
            self.config.op_ranges[param][1],
            nr_bins + 1,
        )
        bin_assignment = np.digitize(conditioning_parameter, bins)
        return bin_assignment

    def prepare_opc_stratified_sampling_pca_cluster(self):
        # assign each time series to a bin based on kmeans clustering of all operating parameters
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.ops)  # shape (nr_time_series, 2)
        kmeans = KMeans(
            n_clusters=self.config.sampling_bins,
            random_state=self.config.random_seed,
        )
        bin_assignment = kmeans.fit_predict(pca_result)  # shape (nr_time_series,)
        self.bin_assignments: list[np.ndarray] = [bin_assignment]

    def apply_data_augmentation(
        self, time_series: np.ndarray, operating_parameters: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        augmented_series = time_series.copy()
        augmented_ops = operating_parameters.copy()
        match self.config.data_augmentation:
            case "white_noise":
                # stationary noise
                noise_std = self.ts_alpha * np.std(time_series)
                ts_noise = np.random.normal(
                    loc=0.0, scale=noise_std, size=time_series.shape
                )
                augmented_series += ts_noise

                z = np.random.multivariate_normal(
                    mean=np.zeros(self.config.num_ops), cov=self.ops_corr
                )
                delta_theta = self.ops_beta * self.ops_std * z
                augmented_ops += delta_theta
            case "random_walk":
                # temporal noise which increases variance linearly with time
                # simulates drift in the time series
                # physics do not stay intact under this augmentation!
                raise NotImplementedError(
                    "random_walk augmentation not implemented yet."
                )
            case None:
                pass
            case _:
                raise ValueError(
                    f"Unknown data augmentation: {self.config.data_augmentation}"
                )
        return augmented_series, augmented_ops

    def prepare_sample(self, idx: int) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            "prepare_sample method must be implemented in subclass."
        )

    def __iter__(self) -> Iterator:
        if self.mode == "train":
            while True:
                # indefinitely iterate and yield random samples, stratified by bins
                for bin_assignment in self.bin_assignments:
                    unique_bins = np.unique(bin_assignment)
                    for selected_bin in unique_bins:
                        # a list of time series indices that belong to the selected bin
                        candidates = np.where(bin_assignment == selected_bin)[0]
                        if len(candidates) == 0:
                            continue

                        time_series_idx: int = np.random.choice(candidates)
                        yield self.prepare_sample(time_series_idx)
        else:  # mode == "val"
            for time_series_idx in range(len(self.time_series)):
                # yield each sample exactly once
                yield self.prepare_sample(time_series_idx)
