from __future__ import annotations

from .config import TOOLCHAINS, ToolchainSpec
from .toolchain_registry import EXPLORATION_FORECAST_MODELS

PREPROCESSING_TASK_CATEGORIES = frozenset(
    {"Imputation", "Anomaly Handling", "Transformation", "Decomposition", "Normalization"}
)
FORECASTING_MODEL_TOOLS = frozenset(EXPLORATION_FORECAST_MODELS)
VALID_LESSON_TASK_CATEGORIES = PREPROCESSING_TASK_CATEGORIES | frozenset({"Forecasting", "Strategy"})

TASK_CATEGORY_TO_STAGE: dict[str, str] = {
    "Forecasting": "forecasting",
    "Imputation": "imputation",
    "Anomaly Handling": "anomaly_handling",
    "Transformation": "transformation",
    "Decomposition": "decomposition",
    "Normalization": "normalization",
}

_TOOL_ALIASES: dict[str, str] = {
    "linear": "linear_interpolation",
    "linear interpolation": "linear_interpolation",
    "standard": "standard_scaler",
    "standardscaler": "standard_scaler",
    "standard scaler": "standard_scaler",
    "classical": "classical_decomposition",
    "classical decomposition": "classical_decomposition",
    "zscore": "z_score",
    "z-score": "z_score",
    "z score": "z_score",
    "iqr": "iqr",
    "patchtst": "PatchTST",
    "itransformer": "iTransformer",
    "timesnet": "TimesNet",
    "timemixer": "TimeMixer",
    "dlinear": "DLinear",
}


def normalize_tool_name(name: str) -> str:
    raw = (name or "").strip()
    if not raw:
        return ""
    key = raw.lower().replace("-", "_").replace(" ", "_")
    if key in _TOOL_ALIASES:
        return _TOOL_ALIASES[key]
    for alias, canonical in _TOOL_ALIASES.items():
        if key == alias.replace(" ", "_"):
            return canonical
    if raw in {
        "PatchTST",
        "iTransformer",
        "TimesNet",
        "TimeMixer",
        "DLinear",
        "TiDE",
        "Informer",
        "Autoformer",
        "TimeXer",
        "MultiPatchFormer",
    }:
        return raw
    return raw


def stage_for_task_category(task_category: str) -> str | None:
    return TASK_CATEGORY_TO_STAGE.get(task_category)


def tool_at_stage(spec: ToolchainSpec, task_category: str) -> str:
    stage = stage_for_task_category(task_category)
    if not stage:
        return ""
    return normalize_tool_name(spec.stages.get(stage, ""))


def toolchain_contains_tool(toolchain_name: str, tool: str, task_category: str) -> bool:
    if not tool:
        return False
    spec = TOOLCHAINS[toolchain_name]
    want = normalize_tool_name(tool)
    have = tool_at_stage(spec, task_category)
    if not want or not have:
        return False
    return want.lower() == have.lower() or want.lower() in have.lower() or have.lower() in want.lower()


def toolchains_matching_tool(tool: str, task_category: str) -> list[str]:
    if not tool:
        return []
    return [name for name in TOOLCHAINS if toolchain_contains_tool(name, tool, task_category)]


def stages_dict(toolchain_name: str) -> dict[str, str]:
    return dict(TOOLCHAINS[toolchain_name].stages)


def evidence_toolchain_preprocessing_depth(toolchain_name: str) -> int:
    """Count non-`none` imputation / anomaly / decomposition stages (Alg. 4 parsimony)."""
    spec = TOOLCHAINS.get(toolchain_name)
    if spec is None:
        return 999
    stages = spec.stages
    depth = 0
    for key in ("imputation", "anomaly_handling", "decomposition"):
        if stages.get(key) not in {None, "none"}:
            depth += 1
    return depth


def is_preprocessing_lesson(tool: str, task_category: str) -> bool:
    """Legacy induce: imputation / anomaly / decomposition only."""
    if task_category not in PREPROCESSING_TASK_CATEGORIES:
        return False
    canon = normalize_tool_name(tool)
    return bool(canon) and canon not in FORECASTING_MODEL_TOOLS


def is_valid_lesson_tool(tool: str, task_category: str, stages: dict[str, str]) -> bool:
    """Paper-style induce: tool must match a stage on the evidence toolchain (Strategy exempt)."""
    if task_category == "Strategy":
        return True
    if task_category not in VALID_LESSON_TASK_CATEGORIES:
        return False
    stage = stage_for_task_category(task_category)
    if not stage:
        return False
    canon = normalize_tool_name(tool)
    if not canon:
        return False
    if task_category == "Forecasting":
        if canon not in FORECASTING_MODEL_TOOLS:
            return False
        have = normalize_tool_name(stages.get("forecasting", ""))
        return bool(have) and canon.lower() == have.lower()
    have = normalize_tool_name(stages.get(stage, ""))
    if not have or have == "none":
        return canon == "none" and task_category == "Transformation"
    return canon.lower() == have.lower()


def active_stage_tools(stages: dict[str, str]) -> list[tuple[str, str, str]]:
    """(stage_key, task_category, tool) for each non-trivial stage on a toolchain."""
    mapping = [
        ("imputation", "Imputation"),
        ("anomaly_handling", "Anomaly Handling"),
        ("transformation", "Transformation"),
        ("decomposition", "Decomposition"),
        ("normalization", "Normalization"),
        ("forecasting", "Forecasting"),
    ]
    out: list[tuple[str, str, str]] = []
    for stage_key, category in mapping:
        tool = normalize_tool_name(stages.get(stage_key, ""))
        if not tool:
            continue
        if stage_key in ("transformation",) and tool == "none":
            continue
        out.append((stage_key, category, tool))
    return out
