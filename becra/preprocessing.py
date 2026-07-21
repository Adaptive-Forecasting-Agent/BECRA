from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DatasetSpec, OUTPUT_ROOT, ToolchainSpec


def materialize_dataset(
    spec: DatasetSpec,
    toolchain: ToolchainSpec,
    output_root: Path = OUTPUT_ROOT / "prepared_data",
    force: bool = False,
) -> DatasetSpec:
    """Create a toolchain-specific CSV when preprocessing changes the raw data."""
    from .toolchain_registry import EXTERNAL_DECOMP

    needs_materialized = (
        toolchain.imputation != "none"
        or toolchain.anomaly != "none"
        or toolchain.decomposition in EXTERNAL_DECOMP
    )
    if not needs_materialized:
        return spec

    out_dir = Path(output_root) / spec.name / toolchain.name
    out_csv = out_dir / spec.data_path
    if out_csv.exists() and not force:
        return spec.with_csv(out_csv)

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(spec.csv_path)
    date_col = df["date"] if "date" in df.columns else None
    numeric = df.drop(columns=["date"], errors="ignore").apply(pd.to_numeric, errors="coerce")

    numeric = _impute(numeric, toolchain.imputation)
    numeric = _handle_anomalies(numeric, toolchain.anomaly)
    numeric = _external_decomposition(numeric, toolchain.decomposition)

    if date_col is not None:
        result = pd.concat([date_col, numeric], axis=1)
    else:
        result = numeric
    result.to_csv(out_csv, index=False)
    return spec.with_csv(out_csv)


def _impute(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "none":
        return df
    if method == "linear_interpolation":
        return df.interpolate(method="linear", limit_direction="both").ffill().bfill()
    if method == "forward_fill":
        return df.ffill().bfill()
    if method == "backward_fill":
        return df.bfill().ffill()
    if method == "mean":
        return df.fillna(df.mean(numeric_only=True))
    if method == "median":
        return df.fillna(df.median(numeric_only=True))
    if method == "mode":
        modes = df.mode(dropna=True)
        fill = modes.iloc[0] if not modes.empty else df.mean(numeric_only=True)
        return df.fillna(fill)
    if method == "knn":
        from sklearn.impute import KNNImputer

        imputed = KNNImputer(n_neighbors=5).fit_transform(df)
        return pd.DataFrame(imputed, columns=df.columns, index=df.index)
    raise ValueError(f"Unsupported imputation method: {method}")


def _handle_anomalies(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "none":
        return df
    if method == "iqr":
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        iqr = q3 - q1
        low = q1 - 3.0 * iqr
        high = q3 + 3.0 * iqr
        return df.clip(lower=low, upper=high, axis=1)
    if method == "z_score":
        mean = df.mean()
        std = df.std().replace(0, np.nan).fillna(1.0)
        low = mean - 4.0 * std
        high = mean + 4.0 * std
        return df.clip(lower=low, upper=high, axis=1)
    raise ValueError(f"Unsupported anomaly handling method: {method}")


def _external_decomposition(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Apply FFT/EMD as external preprocessing (classical decomp stays in the forecast model)."""
    if method in ("none", "classical_decomposition"):
        return df
    if method == "fft":
        return _fft_lowpass(df)
    if method == "emd":
        return _emd_trend(df)
    raise ValueError(f"Unsupported external decomposition method: {method}")


def _fft_lowpass(df: pd.DataFrame, keep_ratio: float = 0.12) -> pd.DataFrame:
    """Keep low-frequency FFT components per channel (denoising / smooth trend)."""
    out = df.copy()
    n = len(out)
    if n < 8:
        return out
    keep = max(2, int(n * keep_ratio))
    for col in out.columns:
        series = out[col].to_numpy(dtype=float)
        if np.isnan(series).any():
            series = pd.Series(series).interpolate(limit_direction="both").ffill().bfill().to_numpy()
        spec = np.fft.rfft(series)
        spec[keep:] = 0.0
        smoothed = np.fft.irfft(spec, n=n)
        out[col] = smoothed
    return out


def _emd_trend(df: pd.DataFrame, max_imfs: int = 4) -> pd.DataFrame:
    """Replace each channel with the residual (trend) from EMD — last IMF sum excluded."""
    try:
        from PyEMD import EMD
    except ImportError as exc:
        raise ImportError(
            "EMD decomposition requires PyEMD. Install with: pip install EMD-signal"
        ) from exc

    emd = EMD()
    out = df.copy()
    for col in out.columns:
        series = out[col].to_numpy(dtype=float)
        if np.isnan(series).any():
            series = pd.Series(series).interpolate(limit_direction="both").ffill().bfill().to_numpy()
        imfs = emd.emd(series, max_imf=max_imfs)
        if imfs is None or len(imfs) == 0:
            continue
        if len(imfs) == 1:
            out[col] = imfs[0]
        else:
            out[col] = np.sum(imfs[:-1], axis=0)
    return out
