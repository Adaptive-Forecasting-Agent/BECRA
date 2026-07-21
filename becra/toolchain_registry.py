"""Generate the full long-term benchmark toolchain pool (preprocess × forecast models)."""

from __future__ import annotations

import itertools
from typing import Any

from .config import ToolchainSpec, _common_transformer_args

# Exploration uses active tools only (no stage "none" skips).
EXPLORATION_IMPUTATIONS = ("linear_interpolation", "knn", "mean")
EXPLORATION_ANOMALIES = ("iqr", "z_score")
EXPLORATION_DECOMPOSITIONS = ("fft", "classical_decomposition")
CLASSICAL_DECOMP_MODELS = frozenset({"DLinear", "TimeMixer"})
EXTERNAL_DECOMP = frozenset({"fft"})

EXPLORATION_FORECAST_MODELS = (
    "Autoformer",
    "PatchTST",
    "iTransformer",
    "TimesNet",
    "TimeMixer",
    "DLinear",
    "TimeXer",
    "MultiPatchFormer",
)

# Strong forecasters explored both as plain baselines and with FFT-only decomposition
# (imputation/anomaly stay none). This closes the explore gap for patterns like
# none+fft+TimeXer that stage-wise planning may compose but the active preprocess
# grid never samples.
BASELINE_ONLY_FORECAST_MODELS = frozenset({"TimeXer", "MultiPatchFormer"})
BASELINE_FFT_DECOMPOSITIONS = ("none", "fft")

_SLUG = {
    "none": "none",
    "linear_interpolation": "linear",
    "knn": "knn",
    "mean": "mean",
    "iqr": "iqr",
    "z_score": "zscore",
    "classical_decomposition": "classical",
    "fft": "fft",
    "emd": "emd",
    "standard_scaler": "standard",
}


def _slug(*parts: str) -> str:
    return "_".join(parts)


def _model_args(model: str, decomposition: str) -> tuple[dict[str, Any], dict[str, Any]]:
    common = _common_transformer_args()
    if model == "TimeMixer":
        args: dict[str, Any] = {
            "label_len": 0,
            "e_layers": 2,
            "d_layers": 1,
            "factor": 3,
            "d_model": 16,
            "d_ff": 32,
            "learning_rate": 0.01,
            "train_epochs": 10,
            "patience": 5,
            "batch_size": 128,
            "down_sampling_layers": 3,
            "down_sampling_method": "avg",
            "down_sampling_window": 2,
        }
        quick = {"train_epochs": 1, "patience": 1, "batch_size": 8, "down_sampling_layers": 1}
        if decomposition == "classical_decomposition":
            args["decomp_method"] = "moving_avg"
            args["moving_avg"] = 25
        return args, quick

    if model == "DLinear":
        args = {
            "label_len": 48,
            "moving_avg": 25,
            "learning_rate": 0.0005,
            "train_epochs": 10,
            "patience": 3,
            "batch_size": 32,
        }
        quick = {"train_epochs": 1, "patience": 1, "batch_size": 8}
        return args, quick

    if model == "PatchTST":
        args = {**common, "e_layers": 1, "n_heads": 8}
        quick = {"d_model": 32, "d_ff": 64, "n_heads": 2, "train_epochs": 1, "patience": 1, "batch_size": 8}
        return args, quick

    if model == "iTransformer":
        args = {
            **common,
            "e_layers": 3,
            "d_ff": 512,
            "factor": 3,
            "learning_rate": 0.0005,
            "batch_size": 16,
            "train_epochs": 10,
            "patience": 3,
        }
        quick = {"d_model": 32, "d_ff": 64, "n_heads": 2, "e_layers": 1, "train_epochs": 1, "patience": 1, "batch_size": 4}
        return args, quick

    if model == "TimesNet":
        args = {**common, "d_model": 16, "d_ff": 32, "top_k": 5}
        quick = {"d_model": 8, "d_ff": 16, "top_k": 2, "train_epochs": 1, "patience": 1, "batch_size": 8}
        return args, quick

    if model == "TiDE":
        args = {
            **common,
            "d_model": 128,
            "d_ff": 256,
            "learning_rate": 0.001,
            "batch_size": 32,
            "train_epochs": 10,
            "patience": 3,
        }
        quick = {"d_model": 32, "d_ff": 64, "train_epochs": 1, "patience": 1, "batch_size": 8}
        return args, quick

    if model in ("Informer", "Autoformer"):
        args = {**common, "factor": 3, "train_epochs": 10, "patience": 3}
        quick = {"d_model": 32, "d_ff": 64, "n_heads": 2, "train_epochs": 1, "patience": 1, "batch_size": 8}
        return args, quick

    if model == "TimeXer":
        args = {
            **common,
            "d_model": 256,
            "d_ff": 512,
            "e_layers": 1,
            "factor": 3,
            "learning_rate": 0.0001,
            "batch_size": 4,
            "train_epochs": 10,
            "patience": 3,
        }
        quick = {"d_model": 64, "d_ff": 128, "e_layers": 1, "train_epochs": 1, "patience": 1, "batch_size": 4}
        return args, quick

    if model == "MultiPatchFormer":
        args = {
            **common,
            "d_model": 256,
            "d_ff": 512,
            "e_layers": 1,
            "n_heads": 8,
            "batch_size": 32,
            "train_epochs": 10,
            "patience": 3,
        }
        quick = {"d_model": 64, "d_ff": 128, "n_heads": 4, "train_epochs": 1, "patience": 1, "batch_size": 8}
        return args, quick

    raise ValueError(f"No model args template for {model!r}")


def _tags(imputation: str, anomaly: str, decomposition: str, model: str) -> tuple[str, ...]:
    tags: list[str] = [model.lower()]
    if imputation != "none":
        tags.append("imputation")
    if anomaly != "none":
        tags.append("robust")
    if decomposition == "classical_decomposition":
        tags.extend(("multiscale", "seasonal"))
    elif decomposition == "fft":
        tags.append("frequency")
    elif decomposition == "emd":
        tags.append("nonstationary")
    return tuple(tags)


def _make_toolchain(
    imputation: str,
    anomaly: str,
    decomposition: str,
    forecasting: str,
) -> ToolchainSpec:
    name = _slug(
        _SLUG[imputation],
        _SLUG[anomaly],
        "none",
        _SLUG[decomposition],
        _SLUG["standard_scaler"],
        forecasting.lower(),
    )
    model_args, quick_args = _model_args(forecasting, decomposition)
    stage_labels = {
        "linear_interpolation": "Linear Interpolation",
        "z_score": "Z-Score",
        "classical_decomposition": "Classical Decomposition",
        "fft": "FFT (low-pass)",
        "emd": "EMD (trend residual)",
        "knn": "KNN",
        "mean": "Mean",
    }
    parts = [
        stage_labels.get(imputation, imputation.title() if imputation != "none" else "None"),
        stage_labels.get(anomaly, anomaly.title() if anomaly != "none" else "None"),
        "None",
        stage_labels.get(decomposition, decomposition.title() if decomposition != "none" else "None"),
        "StandardScaler",
        forecasting,
    ]
    description = " -> ".join(parts) + "."
    return ToolchainSpec(
        name=name,
        imputation=imputation,
        anomaly=anomaly,
        transform="none",
        decomposition=decomposition,
        normalization="standard_scaler",
        forecasting=forecasting,
        description=description,
        tags=_tags(imputation, anomaly, decomposition, forecasting),
        model_args=model_args,
        quick_args=quick_args,
    )


def build_exploration_toolchains() -> dict[str, ToolchainSpec]:
    """Active preprocess grid plus plain/FFT baselines for strong forecasters."""
    toolchains: dict[str, ToolchainSpec] = {}
    for imp, anom, decomp, model in itertools.product(
        EXPLORATION_IMPUTATIONS,
        EXPLORATION_ANOMALIES,
        EXPLORATION_DECOMPOSITIONS,
        EXPLORATION_FORECAST_MODELS,
    ):
        if decomp == "classical_decomposition" and model not in CLASSICAL_DECOMP_MODELS:
            continue
        spec = _make_toolchain(imp, anom, decomp, model)
        toolchains[spec.name] = spec
    for model in sorted(BASELINE_ONLY_FORECAST_MODELS):
        for decomp in BASELINE_FFT_DECOMPOSITIONS:
            spec = _make_toolchain("none", "none", decomp, model)
            toolchains[spec.name] = spec
    return toolchains


def _toolchain_name_from_stage_map(stage_map: dict[str, str]) -> str:
    return _slug(
        _SLUG.get(stage_map["imputation"], stage_map["imputation"]),
        _SLUG.get(stage_map["anomaly_handling"], stage_map["anomaly_handling"]),
        "none",
        _SLUG.get(stage_map["decomposition"], stage_map["decomposition"]),
        _SLUG["standard_scaler"],
        stage_map["forecasting"].lower(),
    )


def normalize_stage_map(stages: dict[str, Any]) -> dict[str, str] | None:
    """Parse planner/LLM JSON stages into canonical tool names."""
    from .tools import normalize_tool_name

    stage_map = {
        "imputation": normalize_tool_name(str(stages.get("imputation", ""))),
        "anomaly_handling": normalize_tool_name(
            str(stages.get("anomaly_handling", stages.get("anomaly", "")))
        ),
        "transformation": "none",
        "decomposition": normalize_tool_name(str(stages.get("decomposition", ""))),
        "normalization": "standard_scaler",
        "forecasting": normalize_tool_name(str(stages.get("forecasting", ""))),
    }
    for stage in ("imputation", "anomaly_handling", "decomposition", "forecasting"):
        if not stage_map[stage]:
            return None
    return stage_map


def is_valid_stage_combination(
    imputation: str,
    anomaly: str,
    decomposition: str,
    forecasting: str,
) -> bool:
    """Any stage-wise product is allowed unless explicitly incompatible."""
    allowed_imp = {"none", *EXPLORATION_IMPUTATIONS}
    allowed_anom = {"none", *EXPLORATION_ANOMALIES}
    allowed_decomp = {"none", *EXPLORATION_DECOMPOSITIONS}
    if imputation not in allowed_imp or anomaly not in allowed_anom or decomposition not in allowed_decomp:
        return False
    if forecasting not in EXPLORATION_FORECAST_MODELS:
        return False
    if decomposition == "classical_decomposition" and forecasting not in CLASSICAL_DECOMP_MODELS:
        return False
    return True


def register_toolchain_from_stages(stages: dict[str, Any]) -> ToolchainSpec | None:
    """Materialize any valid stage-wise combo into TOOLCHAINS (not limited to pre-enumerated grid)."""
    from .config import TOOLCHAINS

    stage_map = normalize_stage_map(stages)
    if stage_map is None:
        return None
    imp = stage_map["imputation"]
    anom = stage_map["anomaly_handling"]
    decomp = stage_map["decomposition"]
    model = stage_map["forecasting"]
    if not is_valid_stage_combination(imp, anom, decomp, model):
        return None
    name = _toolchain_name_from_stage_map(stage_map)
    existing = TOOLCHAINS.get(name)
    if existing is not None:
        return existing
    spec = _make_toolchain(imp, anom, decomp, model)
    TOOLCHAINS[name] = spec
    return spec


def resolve_toolchain_from_stages(stages: dict[str, Any]) -> str | None:
    """Map per-stage tool choices to a toolchain name, registering the spec on demand."""
    spec = register_toolchain_from_stages(stages)
    return spec.name if spec is not None else None


def _parse_toolchain_name(name: str) -> tuple[str, str, str, str] | None:
    """Decode slug imp_anom_none_decomp_standard_model into stage tools."""
    parts = name.split("_")
    if len(parts) != 6 or parts[2] != "none" or parts[4] != "standard":
        return None
    imp_slug, anom_slug, _, decomp_slug, _, model_slug = parts
    rev = {v: k for k, v in _SLUG.items()}
    imp = rev.get(imp_slug)
    anom = rev.get(anom_slug)
    decomp = rev.get(decomp_slug)
    if not imp or not anom or not decomp:
        return None
    model_map = {
        "patchtst": "PatchTST",
        "itransformer": "iTransformer",
        "timesnet": "TimesNet",
        "timemixer": "TimeMixer",
        "dlinear": "DLinear",
        "tide": "TiDE",
        "autoformer": "Autoformer",
        "timexer": "TimeXer",
        "multipatchformer": "MultiPatchFormer",
    }
    model = model_map.get(model_slug)
    if model is None:
        return None
    return imp, anom, decomp, model


def register_toolchain_by_name(name: str) -> ToolchainSpec | None:
    """Register a toolchain from its slug if stages are valid."""
    from .config import TOOLCHAINS

    if name in TOOLCHAINS:
        return TOOLCHAINS[name]
    parsed = _parse_toolchain_name(name)
    if parsed is None:
        return None
    imp, anom, decomp, model = parsed
    if not is_valid_stage_combination(imp, anom, decomp, model):
        return None
    spec = _make_toolchain(imp, anom, decomp, model)
    if spec.name != name:
        return None
    TOOLCHAINS[name] = spec
    return spec


def planning_stage_tool_library() -> dict[str, list[str]]:
    """Executable tools exposed to stage-wise LLM planners."""
    return {
        "imputation": ["none", *EXPLORATION_IMPUTATIONS],
        "anomaly_handling": ["none", *EXPLORATION_ANOMALIES],
        "transformation": ["none"],
        "decomposition": ["none", *EXPLORATION_DECOMPOSITIONS],
        "normalization": ["standard_scaler"],
        "forecasting": list(EXPLORATION_FORECAST_MODELS),
    }
