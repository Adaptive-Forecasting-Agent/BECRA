from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import OUTPUT_ROOT, TSL_ROOT, DatasetSpec, ToolchainSpec
from .preprocessing import materialize_dataset


METRIC_RE = re.compile(r"mse:\s*([0-9.eE+-]+),\s*mae:\s*([0-9.eE+-]+)")
SETTING_RE = re.compile(r">>>>>>>start training\s*:\s*(.+?)\s*>")
_active_model_ids: set[str] = set()
_run_registry_lock = threading.Lock()


def tsl_freq(freq: str) -> str:
    """Map dataset sampling labels to Time-Series-Library --freq keys.

    TimeMixer (and TimeFeatureEmbedding) only accept single-letter codes in
    ``freq_map`` (h/t/s/...). ``15min`` / ``10min`` work in the data loader but
    break TimeMixer with KeyError unless aliased to minutely ``t``.
    """
    return {"15min": "t", "10min": "t"}.get(freq, freq)


@dataclass
class ForecastRun:
    dataset: str
    pred_len: int
    toolchain: str
    model: str
    command: list[str]
    mse: float | None = None
    mae: float | None = None
    stdout_tail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "pred_len": self.pred_len,
            "toolchain": self.toolchain,
            "model": self.model,
            "mse": self.mse,
            "mae": self.mae,
            "command": " ".join(shlex.quote(part) for part in self.command),
        }


def build_tsl_command(
    spec: DatasetSpec,
    toolchain: ToolchainSpec,
    pred_len: int,
    seq_len: int = 96,
    python_bin: str | None = None,
    use_gpu: bool = True,
    gpu: int = 0,
    quick: bool = False,
    overrides: dict[str, Any] | None = None,
) -> list[str]:
    args = toolchain.args_for(quick=quick)
    if overrides:
        args.update({k: v for k, v in overrides.items() if v is not None})

    label_len = int(args.pop("label_len", 48))
    batch_size = int(args.pop("batch_size", 32))
    train_epochs = int(args.pop("train_epochs", 10))
    patience = int(args.pop("patience", 3))
    learning_rate = float(args.pop("learning_rate", 0.0001))
    num_workers = int(args.pop("num_workers", 0 if quick else 4))
    model_id = f"BECRA_{spec.name}_{toolchain.name}_sl{seq_len}_pl{pred_len}"

    command = [
        python_bin or sys.executable,
        "-u",
        "run.py",
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
        "--root_path",
        str(spec.root_path),
        "--data_path",
        spec.data_path,
        "--model_id",
        model_id,
        "--model",
        toolchain.model,
        "--data",
        spec.tsl_data,
        "--features",
        spec.features,
        "--seq_len",
        str(seq_len),
        "--label_len",
        str(label_len),
        "--pred_len",
        str(pred_len),
        "--enc_in",
        str(spec.n_channels),
        "--dec_in",
        str(spec.n_channels),
        "--c_out",
        str(spec.n_channels),
        "--freq",
        tsl_freq(spec.freq),
        "--target",
        spec.target,
        "--des",
        "BECRA",
        "--itr",
        "1",
        "--train_epochs",
        str(train_epochs),
        "--patience",
        str(patience),
        "--batch_size",
        str(batch_size),
        "--learning_rate",
        str(learning_rate),
        "--num_workers",
        str(num_workers),
        "--checkpoints",
        str(OUTPUT_ROOT / "checkpoints"),
    ]
    if not use_gpu:
        command.append("--no_use_gpu")
    else:
        # Physical GPU is selected via CUDA_VISIBLE_DEVICES in run_forecast; TSL always uses logical cuda:0.
        command.extend(["--gpu", "0", "--gpu_type", "cuda"])
    for key, value in args.items():
        if isinstance(value, bool):
            if value:
                command.append(f"--{key}")
        elif value is not None:
            command.extend([f"--{key}", str(value)])
    return command


def _parse_tsl_setting(stdout: str) -> str | None:
    for line in stdout.splitlines():
        match = SETTING_RE.search(line)
        if match:
            return match.group(1).strip()
    return None


def _register_active_run(model_id: str) -> None:
    with _run_registry_lock:
        _active_model_ids.add(model_id)


def _unregister_active_run(model_id: str) -> None:
    with _run_registry_lock:
        _active_model_ids.discard(model_id)


def prune_tsl_run_artifacts(
    model_id: str | None = None,
    *,
    setting: str | None = None,
    checkpoints_root: Path | None = None,
    also_results: bool = True,
) -> list[str]:
    """Remove artifact dirs for one **finished** run only (never prefix-guess under parallel load).

    Prefer the exact TSL ``setting`` folder name parsed from training stdout. Prefix-based
    deletion can collide across concurrent jobs (e.g. manual ``rm`` or ``_pl96_`` vs
    ``_pl720_`` on the same toolchain).
    """
    if os.getenv("BECRA_KEEP_CHECKPOINTS", "").strip().lower() in ("1", "true", "yes"):
        return []

    ckpt_root = checkpoints_root or (OUTPUT_ROOT / "checkpoints")
    roots = [ckpt_root]
    if also_results:
        roots.extend([TSL_ROOT / "results", TSL_ROOT / "test_results"])

    removed: list[str] = []
    names: list[str] = []
    if setting:
        names = [setting]
    elif model_id:
        with _run_registry_lock:
            if model_id in _active_model_ids:
                return []
        prefix = f"long_term_forecast_{model_id}_"
        for root in roots:
            if not root.exists():
                continue
            for path in root.iterdir():
                if path.is_dir() and path.name.startswith(prefix):
                    with _run_registry_lock:
                        if any(aid in path.name for aid in _active_model_ids if aid != model_id):
                            continue
                    names.append(path.name)

    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
        for name in names:
            path = root / name
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                removed.append(str(path))
    return removed


def run_forecast(
    spec: DatasetSpec,
    toolchain: ToolchainSpec,
    pred_len: int,
    seq_len: int = 96,
    python_bin: str | None = None,
    use_gpu: bool = True,
    gpu: int = 0,
    quick: bool = False,
    dry_run: bool = False,
    force_preprocess: bool = False,
    overrides: dict[str, Any] | None = None,
    timeout: int | None = None,
    prune_artifacts: bool = True,
) -> ForecastRun:
    prepared = spec if dry_run else materialize_dataset(spec, toolchain, force=force_preprocess)
    command = build_tsl_command(
        prepared,
        toolchain,
        pred_len=pred_len,
        seq_len=seq_len,
        python_bin=python_bin,
        use_gpu=use_gpu,
        gpu=gpu,
        quick=quick,
        overrides=overrides,
    )
    model_id = f"BECRA_{spec.name}_{toolchain.name}_sl{seq_len}_pl{pred_len}"
    run = ForecastRun(spec.name, pred_len, toolchain.name, toolchain.model, command)
    if dry_run:
        return run

    (OUTPUT_ROOT / "checkpoints").mkdir(parents=True, exist_ok=True)
    _register_active_run(model_id)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{TSL_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    if use_gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        completed = subprocess.run(
            command,
            cwd=TSL_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    finally:
        _unregister_active_run(model_id)

    run.stdout_tail = "\n".join(completed.stdout.splitlines()[-80:])
    if completed.returncode != 0:
        raise RuntimeError(f"Time-Series-Library run failed with exit code {completed.returncode}\n{run.stdout_tail}")
    metrics = METRIC_RE.findall(completed.stdout)
    if metrics:
        mse, mae = metrics[-1]
        run.mse = float(mse)
        run.mae = float(mae)
        if prune_artifacts:
            setting = _parse_tsl_setting(completed.stdout)
            removed = prune_tsl_run_artifacts(model_id, setting=setting)
            if removed and os.getenv("BECRA_PRUNE_VERBOSE", "").strip().lower() in ("1", "true", "yes"):
                print(f"[prune] removed {len(removed)} artifact dir(s) for {setting or model_id}")
    return run


