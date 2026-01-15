# src/features.py
import numpy as np
from typing import Tuple, List

def polygon_area_shoelace(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    x2 = np.concatenate([x, x[:1]])
    y2 = np.concatenate([y, y[:1]])
    return 0.5 * float(np.abs(np.sum(x2[:-1] * y2[1:] - x2[1:] * y2[:-1])))

def symmetry_feature(y: np.ndarray) -> float:
    n = len(y)
    if n < 4:
        return 0.0
    h = n // 2
    a = y[:h]
    b = y[-h:][::-1]
    return float(np.mean(np.abs(a - b)))

def fill_factor_feature(x: np.ndarray, y: np.ndarray, area: float) -> float:
    xr = float(np.max(x) - np.min(x))
    yr = float(np.max(y) - np.min(y))
    denom = xr * yr
    if denom <= 1e-9:
        return 0.0
    return float(abs(area) / denom)

def skewness(y: np.ndarray) -> float:
    y = y.astype(np.float64)
    m = y.mean()
    s = y.std()
    if s < 1e-12:
        return 0.0
    return float(np.mean(((y - m) / s) ** 3))

def kurtosis_excess(y: np.ndarray) -> float:
    y = y.astype(np.float64)
    m = y.mean()
    s = y.std()
    if s < 1e-12:
        return 0.0
    return float(np.mean(((y - m) / s) ** 4) - 3.0)

def mean_slopes(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    n = len(x)
    if n < 4:
        return 0.0, 0.0
    h = n // 2
    dx1 = np.diff(x[:h]); dy1 = np.diff(y[:h])
    dx2 = np.diff(x[h:]); dy2 = np.diff(y[h:])
    s1 = np.mean(dy1 / (dx1 + 1e-9)) if len(dx1) else 0.0
    s2 = np.mean(dy2 / (dx2 + 1e-9)) if len(dx2) else 0.0
    return float(s1), float(s2)

def inflection_count(y: np.ndarray) -> float:
    if len(y) < 5:
        return 0.0
    dy = np.diff(y)
    ddy = np.diff(dy)
    s = np.sign(ddy)
    s[s == 0] = 1
    return float(np.sum(s[1:] != s[:-1]))

def fft_energy(y: np.ndarray, k: int = 5) -> np.ndarray:
    y = y.astype(np.float64)
    y = y - y.mean()
    spec = np.fft.rfft(y)
    mag = np.abs(spec)
    vals = mag[1:k+1]  # skip DC
    if len(vals) < k:
        vals = np.pad(vals, (0, k - len(vals)), mode="constant")
    return vals.astype(np.float32)

FEATURE_NAMES_7 = [
    "area", "y_max", "y_min", "y_mean", "y_std", "symmetry_l1", "fill_factor"
]

FEATURE_NAMES_17 = [
    "area", "y_max", "y_min", "y_mean", "y_std", "symmetry_l1", "fill_factor",
    "skewness", "kurtosis_excess", "slope_first_half", "slope_second_half", "inflection_count",
    "fft_mag_1", "fft_mag_2", "fft_mag_3", "fft_mag_4", "fft_mag_5"
]

def compute_features_7(x_res: np.ndarray, y_res: np.ndarray) -> np.ndarray:
    area = polygon_area_shoelace(x_res, y_res)
    y_max = float(np.max(y_res))
    y_min = float(np.min(y_res))
    y_mean = float(np.mean(y_res))
    y_std = float(np.std(y_res))
    sym = symmetry_feature(y_res)
    ff = fill_factor_feature(x_res, y_res, area)
    return np.array([area, y_max, y_min, y_mean, y_std, sym, ff], dtype=np.float32)

def compute_features_17(x_res: np.ndarray, y_res: np.ndarray) -> np.ndarray:
    area = polygon_area_shoelace(x_res, y_res)
    y_max = float(np.max(y_res))
    y_min = float(np.min(y_res))
    y_mean = float(np.mean(y_res))
    y_std = float(np.std(y_res))
    sym = symmetry_feature(y_res)
    ff = fill_factor_feature(x_res, y_res, area)

    sk = skewness(y_res)
    ku = kurtosis_excess(y_res)
    s1, s2 = mean_slopes(x_res, y_res)
    infl = inflection_count(y_res)
    fft5 = fft_energy(y_res, k=5)

    return np.array(
        [area, y_max, y_min, y_mean, y_std, sym, ff,
         sk, ku, s1, s2, infl,
         fft5[0], fft5[1], fft5[2], fft5[3], fft5[4]],
        dtype=np.float32
    )

def standardize_features(train_F: np.ndarray, val_F: np.ndarray, test_F: np.ndarray):
    mu = train_F.mean(axis=0, keepdims=True)
    sd = train_F.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (train_F - mu) / sd, (val_F - mu) / sd, (test_F - mu) / sd, mu, sd
