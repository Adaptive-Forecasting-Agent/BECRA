from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from scipy import stats

try:
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tools.sm_exceptions import InterpolationWarning
except Exception:  # pragma: no cover - statsmodels is optional on remote machines
    adfuller = None
    kpss = None
    InterpolationWarning = Warning

from .config import DatasetSpec


@dataclass
class MetaFeatures:
    dataset: str
    series_length: int
    num_variates: int
    sampling_frequency: str
    missing_rate: float
    longest_missing_run: int
    missing_pattern: str
    mean: float
    std: float
    variance: float
    coefficient_of_variation: float
    volatility_level: str
    anomaly_ratio: float
    linear_trend_r2: float
    nonlinear_trend_r2: float
    trend_level: str
    seasonality_strength: float
    dominant_period: float | None
    fft_period: float | None
    fft_strength: float
    seasonality_level: str
    adf_pvalue: float | None
    kpss_pvalue: float | None
    stationarity_label: str
    skewness: float
    kurtosis: float
    entropy: float
    snr_db: float
    mean_abs_correlation: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_meta_features(spec: DatasetSpec, max_points: int = 5000) -> MetaFeatures:
    df = pd.read_csv(spec.csv_path)
    if "date" in df.columns:
        values = df.drop(columns=["date"])
    else:
        values = df.copy()
    values = values.apply(pd.to_numeric, errors="coerce")

    target = values[spec.target] if spec.target in values.columns else values.iloc[:, -1]
    target_clean = target.interpolate(limit_direction="both").ffill().bfill()
    x = _sample_series(target_clean.to_numpy(dtype=float), max_points=max_points)
    x = x[np.isfinite(x)]
    if x.size < 8:
        raise ValueError(f"Not enough finite target values in {spec.csv_path}")

    missing_rate = float(values.isna().to_numpy().mean())
    longest_missing_run = _longest_missing_run(target.isna().to_numpy())
    missing_pattern = _missing_pattern(target.isna().to_numpy())

    mean = float(np.mean(x))
    std = float(np.std(x))
    variance = float(np.var(x))
    cv = float(std / (abs(mean) + 1e-8))
    volatility_level = _bucket(cv, low=0.15, high=0.6, labels=("low", "medium", "high"))

    anomaly_ratio = _robust_anomaly_ratio(x)
    linear_r2, nonlinear_r2 = _trend_scores(x)
    trend_level = _bucket(max(linear_r2, nonlinear_r2), low=0.05, high=0.25, labels=("weak", "moderate", "strong"))

    seasonality_strength, dominant_period = _seasonality_from_acf(x, spec.freq)
    fft_period, fft_strength = _fft_period(x)
    seasonality_level = _bucket(max(seasonality_strength, fft_strength), low=0.18, high=0.35, labels=("weak", "moderate", "strong"))

    adf_pvalue = _adf_pvalue(x)
    kpss_pvalue = _kpss_pvalue(x)
    stationarity_label = _stationarity_label(adf_pvalue, kpss_pvalue)

    skewness = float(stats.skew(x, nan_policy="omit"))
    kurtosis = float(stats.kurtosis(x, fisher=True, nan_policy="omit"))
    entropy = _entropy(x)
    snr_db = _snr_db(x)
    mean_abs_correlation = _mean_abs_corr(values, max_points=max_points)

    return MetaFeatures(
        dataset=spec.name,
        series_length=int(len(df)),
        num_variates=int(values.shape[1]),
        sampling_frequency=spec.freq,
        missing_rate=missing_rate,
        longest_missing_run=longest_missing_run,
        missing_pattern=missing_pattern,
        mean=mean,
        std=std,
        variance=variance,
        coefficient_of_variation=cv,
        volatility_level=volatility_level,
        anomaly_ratio=anomaly_ratio,
        linear_trend_r2=linear_r2,
        nonlinear_trend_r2=nonlinear_r2,
        trend_level=trend_level,
        seasonality_strength=seasonality_strength,
        dominant_period=dominant_period,
        fft_period=fft_period,
        fft_strength=fft_strength,
        seasonality_level=seasonality_level,
        adf_pvalue=adf_pvalue,
        kpss_pvalue=kpss_pvalue,
        stationarity_label=stationarity_label,
        skewness=skewness,
        kurtosis=kurtosis,
        entropy=entropy,
        snr_db=snr_db,
        mean_abs_correlation=mean_abs_correlation,
    )


def summarize_meta(meta: MetaFeatures | dict[str, Any]) -> str:
    data = meta.to_dict() if isinstance(meta, MetaFeatures) else dict(meta)
    keys = [
        "dataset",
        "series_length",
        "num_variates",
        "sampling_frequency",
        "missing_rate",
        "missing_pattern",
        "volatility_level",
        "trend_level",
        "seasonality_level",
        "dominant_period",
        "stationarity_label",
        "anomaly_ratio",
        "mean_abs_correlation",
    ]
    return ", ".join(f"{k}={data.get(k)}" for k in keys)


def _sample_series(x: np.ndarray, max_points: int) -> np.ndarray:
    if x.size <= max_points:
        return x
    idx = np.linspace(0, x.size - 1, max_points).astype(int)
    return x[idx]


def _longest_missing_run(mask: np.ndarray) -> int:
    longest = current = 0
    for item in mask:
        current = current + 1 if item else 0
        longest = max(longest, current)
    return int(longest)


def _missing_pattern(mask: np.ndarray) -> str:
    rate = float(mask.mean()) if mask.size else 0.0
    if rate == 0:
        return "none"
    longest = _longest_missing_run(mask)
    if longest >= max(8, int(0.02 * mask.size)):
        return "block-wise"
    transitions = int(np.count_nonzero(np.diff(mask.astype(int)))) if mask.size > 1 else 0
    if transitions > max(10, int(rate * mask.size * 0.5)):
        return "point-wise dispersed"
    return "mixed"


def _bucket(value: float, low: float, high: float, labels: tuple[str, str, str]) -> str:
    if value < low:
        return labels[0]
    if value < high:
        return labels[1]
    return labels[2]


def _robust_anomaly_ratio(x: np.ndarray) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-8
    robust_z = 0.6745 * (x - med) / mad
    return float(np.mean(np.abs(robust_z) > 3.5))


def _trend_scores(x: np.ndarray) -> tuple[float, float]:
    t = np.linspace(0.0, 1.0, len(x))
    total = np.sum((x - np.mean(x)) ** 2) + 1e-8
    lin = np.polyfit(t, x, deg=1)
    lin_pred = np.polyval(lin, t)
    poly = np.polyfit(t, x, deg=2)
    poly_pred = np.polyval(poly, t)
    return float(1.0 - np.sum((x - lin_pred) ** 2) / total), float(1.0 - np.sum((x - poly_pred) ** 2) / total)


def _candidate_periods(freq: str, n: int) -> list[int]:
    table = {
        "h": [24, 168],
        "15min": [96, 672],
        "t": [96, 672],
        "10min": [144, 1008],
        "d": [7, 30, 365],
    }
    periods = table.get(freq, [24, 96, 168])
    return [p for p in periods if 2 <= p < n // 2]


def _seasonality_from_acf(x: np.ndarray, freq: str) -> tuple[float, float | None]:
    x = x - np.mean(x)
    denom = np.dot(x, x) + 1e-8
    best_strength = 0.0
    best_period: float | None = None
    for lag in _candidate_periods(freq, len(x)):
        corr = float(np.dot(x[lag:], x[:-lag]) / denom)
        if corr > best_strength:
            best_strength = corr
            best_period = float(lag)
    return float(max(0.0, best_strength)), best_period


def _fft_period(x: np.ndarray) -> tuple[float | None, float]:
    centered = x - np.mean(x)
    spectrum = np.abs(np.fft.rfft(centered))
    if spectrum.size <= 2:
        return None, 0.0
    spectrum[0] = 0.0
    idx = int(np.argmax(spectrum))
    if idx <= 0:
        return None, 0.0
    strength = float(spectrum[idx] / (np.sum(spectrum) + 1e-8))
    period = float(len(centered) / idx)
    return period, strength


def _adf_pvalue(x: np.ndarray) -> float | None:
    if adfuller is None or len(x) < 32:
        return None
    try:
        return float(adfuller(x, autolag="AIC")[1])
    except Exception:
        return None


def _kpss_pvalue(x: np.ndarray) -> float | None:
    if kpss is None or len(x) < 32:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            return float(kpss(x, regression="c", nlags="auto")[1])
    except Exception:
        return None


def _stationarity_label(adf_p: float | None, kpss_p: float | None) -> str:
    adf_stationary = adf_p is not None and adf_p < 0.05
    kpss_stationary = kpss_p is not None and kpss_p >= 0.05
    if adf_stationary and kpss_stationary:
        return "stationary"
    if (adf_p is not None and adf_p >= 0.05) or (kpss_p is not None and kpss_p < 0.05):
        return "non-stationary"
    return "uncertain"


def _entropy(x: np.ndarray, bins: int = 32) -> float:
    hist, _ = np.histogram(x, bins=bins, density=False)
    probs = hist / (hist.sum() + 1e-8)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _snr_db(x: np.ndarray) -> float:
    if len(x) < 9:
        return 0.0
    window = min(25, max(5, len(x) // 50))
    kernel = np.ones(window) / window
    smooth = np.convolve(x, kernel, mode="same")
    noise = x - smooth
    return float(10.0 * np.log10((np.var(smooth) + 1e-8) / (np.var(noise) + 1e-8)))


def _mean_abs_corr(values: pd.DataFrame, max_points: int) -> float | None:
    if values.shape[1] < 2:
        return None
    sampled = values.interpolate(limit_direction="both").ffill().bfill()
    if len(sampled) > max_points:
        sampled = sampled.iloc[np.linspace(0, len(sampled) - 1, max_points).astype(int)]
    if sampled.shape[1] > 64:
        sampled = sampled.iloc[:, np.linspace(0, sampled.shape[1] - 1, 64).astype(int)]
    corr = sampled.corr().to_numpy(dtype=float)
    upper = corr[np.triu_indices_from(corr, k=1)]
    upper = upper[np.isfinite(upper)]
    if upper.size == 0:
        return None
    return float(np.mean(np.abs(upper)))
