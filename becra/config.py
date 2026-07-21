from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = WORKSPACE_ROOT / "data"
TSL_ROOT = WORKSPACE_ROOT / "third_party" / "Time-Series-Library"
OUTPUT_ROOT = WORKSPACE_ROOT / "outputs"
LONG_TERM_HORIZONS = [96, 192, 336, 720]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    csv_path: Path
    tsl_data: str
    freq: str
    target: str = "OT"
    features: str = "M"
    domain: str = ""

    @property
    def root_path(self) -> Path:
        return self.csv_path.parent

    @property
    def data_path(self) -> str:
        return self.csv_path.name

    @property
    def n_channels(self) -> int:
        header = pd.read_csv(self.csv_path, nrows=0)
        return len([c for c in header.columns if c != "date"])

    def with_csv(self, csv_path: Path) -> "DatasetSpec":
        return DatasetSpec(
            name=self.name,
            csv_path=Path(csv_path),
            tsl_data=self.tsl_data,
            freq=self.freq,
            target=self.target,
            features=self.features,
            domain=self.domain,
        )


@dataclass(frozen=True)
class ToolSpec:
    name: str
    category: str
    executable: bool
    description: str = ""


@dataclass(frozen=True)
class ToolchainSpec:
    name: str
    forecasting: str
    imputation: str = "none"
    anomaly: str = "none"
    transform: str = "none"
    decomposition: str = "none"
    normalization: str = "standard"
    description: str = ""
    tags: tuple[str, ...] = ()
    model_args: dict[str, Any] = field(default_factory=dict)
    quick_args: dict[str, Any] = field(default_factory=dict)

    @property
    def model(self) -> str:
        """Compatibility alias used by the Time-Series-Library runner."""
        return self.forecasting

    @property
    def stages(self) -> dict[str, str]:
        return {
            "imputation": self.imputation,
            "anomaly_handling": self.anomaly,
            "transformation": self.transform,
            "decomposition": self.decomposition,
            "normalization": self.normalization,
            "forecasting": self.forecasting,
        }

    def args_for(self, quick: bool = False) -> dict[str, Any]:
        args = dict(self.model_args)
        if quick:
            args.update(self.quick_args)
        return args

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stages": self.stages,
            "description": self.description,
            "tags": list(self.tags),
            "model_args": self.model_args,
        }


DATASETS: dict[str, DatasetSpec] = {
    "ETTh1": DatasetSpec("ETTh1", DATA_ROOT / "ETT-small" / "ETTh1.csv", "ETTh1", "h", domain="electricity transformer"),
    "ETTh2": DatasetSpec("ETTh2", DATA_ROOT / "ETT-small" / "ETTh2.csv", "ETTh2", "h", domain="electricity transformer"),
    "ETTm1": DatasetSpec("ETTm1", DATA_ROOT / "ETT-small" / "ETTm1.csv", "ETTm1", "15min", domain="electricity transformer"),
    "ETTm2": DatasetSpec("ETTm2", DATA_ROOT / "ETT-small" / "ETTm2.csv", "ETTm2", "15min", domain="electricity transformer"),
    "Electricity": DatasetSpec("Electricity", DATA_ROOT / "electricity" / "electricity.csv", "custom", "h", domain="electricity consumption"),
    "Weather": DatasetSpec("Weather", DATA_ROOT / "weather" / "weather.csv", "custom", "10min", domain="meteorology"),
}


def get_dataset(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError as exc:
        known = ", ".join(DATASETS)
        raise KeyError(f"Unknown dataset {name!r}. Known datasets: {known}") from exc


def all_dataset_names() -> list[str]:
    return list(DATASETS)


def _common_transformer_args() -> dict[str, Any]:
    return {
        "label_len": 48,
        "e_layers": 2,
        "d_layers": 1,
        "factor": 3,
        "d_model": 512,
        "d_ff": 2048,
        "dropout": 0.1,
    }


PAPER_TOOL_LIBRARY: dict[str, list[ToolSpec]] = {
    "imputation": [
        ToolSpec("none", "imputation", True, "Skip imputation."),
        ToolSpec("mean", "imputation", True, "Mean imputation."),
        ToolSpec("median", "imputation", True, "Median imputation."),
        ToolSpec("mode", "imputation", True, "Mode imputation."),
        ToolSpec("forward_fill", "imputation", True, "Forward fill."),
        ToolSpec("backward_fill", "imputation", True, "Backward fill."),
        ToolSpec("linear_interpolation", "imputation", True, "Linear interpolation."),
        ToolSpec("knn", "imputation", True, "K-nearest-neighbor imputation."),
        ToolSpec("mice", "imputation", False, "MICE imputation; listed for prompt planning, not used in default long-term runs."),
        ToolSpec("saits", "imputation", False, "SAITS neural imputation; not implemented in this reconstruction."),
        ToolSpec("timesnet_imputation", "imputation", False, "TimesNet imputation mode; separate from forecasting runs."),
    ],
    "anomaly_handling": [
        ToolSpec("none", "anomaly_handling", True, "Skip anomaly handling."),
        ToolSpec("z_score", "anomaly_handling", True, "Z-score clipping."),
        ToolSpec("iqr", "anomaly_handling", True, "Interquartile-range clipping."),
        ToolSpec("isolation_forest", "anomaly_handling", False, "Isolation Forest detector; not in default long-term runs."),
        ToolSpec("lof", "anomaly_handling", False, "Local Outlier Factor detector; not in default long-term runs."),
        ToolSpec("one_class_svm", "anomaly_handling", False, "One-Class SVM detector; not in default long-term runs."),
        ToolSpec("autoencoder", "anomaly_handling", False, "Autoencoder detector; not implemented in this reconstruction."),
    ],
    "transformation": [
        ToolSpec("none", "transformation", True, "Skip transformation."),
        ToolSpec("log", "transformation", False, "Log transform; disabled by default because metrics need inverse transform."),
        ToolSpec("box_cox", "transformation", False, "Box-Cox transform; disabled by default because metrics need inverse transform."),
        ToolSpec("yeo_johnson", "transformation", False, "Yeo-Johnson transform; disabled by default because metrics need inverse transform."),
        ToolSpec("differencing", "transformation", False, "Differencing; disabled by default because forecasts need reconstruction."),
    ],
    "decomposition": [
        ToolSpec("none", "decomposition", True, "Skip external decomposition."),
        ToolSpec("classical_decomposition", "decomposition", True, "Moving-average decomposition when supported by the forecasting model."),
        ToolSpec("stl", "decomposition", False, "STL decomposition; prompt-level tool, not externally recomposed here."),
        ToolSpec("fft", "decomposition", True, "FFT low-pass smoothing applied in preprocessing before forecasting."),
        ToolSpec("wavelet", "decomposition", False, "Wavelet transform; not implemented in this reconstruction."),
        ToolSpec("emd", "decomposition", False, "EMD trend residual; excluded from exploration (no observed gain)."),
        ToolSpec("ceemdan", "decomposition", False, "CEEMDAN; not implemented in this reconstruction."),
    ],
    "normalization": [
        ToolSpec("standard_scaler", "normalization", True, "StandardScaler; also used internally by Time-Series-Library data loaders."),
        ToolSpec("minmax_scaler", "normalization", False, "MinMaxScaler; disabled by default because metrics need inverse transform."),
        ToolSpec("maxabs_scaler", "normalization", False, "MaxAbsScaler; disabled by default because metrics need inverse transform."),
        ToolSpec("robust_scaler", "normalization", False, "RobustScaler; disabled by default because metrics need inverse transform."),
    ],
    "forecasting": [
        ToolSpec("ARIMA", "forecasting", False, "Classical ARIMA; not used in neural long-term benchmark runs."),
        ToolSpec("SARIMA", "forecasting", False, "Seasonal ARIMA; not used in neural long-term benchmark runs."),
        ToolSpec("ETS", "forecasting", False, "Exponential smoothing; not used in neural long-term benchmark runs."),
        ToolSpec("LightGBM", "forecasting", False, "Gradient boosting with lag features; not implemented in this reconstruction."),
        ToolSpec("XGBoost", "forecasting", False, "Gradient boosting with lag features; not implemented in this reconstruction."),
        ToolSpec("LSTM", "forecasting", False, "Recurrent neural network; not part of default reproduced baselines."),
        ToolSpec("GRU", "forecasting", False, "Recurrent neural network; not part of default reproduced baselines."),
        ToolSpec("TCN", "forecasting", False, "Temporal convolutional network; not part of default reproduced baselines."),
        ToolSpec("Informer", "forecasting", True, "Available in Time-Series-Library."),
        ToolSpec("Autoformer", "forecasting", True, "Available in Time-Series-Library."),
        ToolSpec("PatchTST", "forecasting", True, "Available in Time-Series-Library."),
        ToolSpec("iTransformer", "forecasting", True, "Available in Time-Series-Library."),
        ToolSpec("N-BEATS", "forecasting", False, "Listed in paper tool library; not available through this runner."),
        ToolSpec("N-HiTS", "forecasting", False, "Listed in paper tool library; not available through this runner."),
        ToolSpec("DLinear", "forecasting", True, "Available in Time-Series-Library."),
        ToolSpec("TiDE", "forecasting", True, "Available in Time-Series-Library."),
        ToolSpec("TimesNet", "forecasting", True, "Available in Time-Series-Library."),
        ToolSpec("TimeMixer", "forecasting", True, "Available in Time-Series-Library."),
        ToolSpec("TimeXer", "forecasting", True, "Strong multivariate forecaster; often best with no preprocessing."),
        ToolSpec("MultiPatchFormer", "forecasting", True, "Strong multiscale patch forecaster; often best with no preprocessing."),
    ],
}


def tool_library_as_dict() -> dict[str, list[dict[str, Any]]]:
    return {category: [asdict(tool) for tool in tools] for category, tools in PAPER_TOOL_LIBRARY.items()}


from .toolchain_registry import build_exploration_toolchains

TOOLCHAINS: dict[str, ToolchainSpec] = build_exploration_toolchains()


def get_toolchain(name: str) -> ToolchainSpec:
    try:
        return TOOLCHAINS[name]
    except KeyError as exc:
        from .toolchain_registry import register_toolchain_by_name

        spec = register_toolchain_by_name(name)
        if spec is not None:
            TOOLCHAINS[name] = spec
            return spec
        known = ", ".join(list(TOOLCHAINS)[:20])
        raise KeyError(f"Unknown toolchain {name!r}. Known toolchains (sample): {known} ...") from exc


def all_toolchain_names() -> list[str]:
    return list(TOOLCHAINS)
