from __future__ import annotations

from dataclasses import dataclass, asdict
from io import BytesIO, StringIO
import json
import math
import zipfile
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import erf

from . import cafa_legacy


PEAK_TYPES = ("Gaussian", "Skew Normal", "Lorentzian", "Pseudo-Voigt")


WAVELET_TYPES = ("GD1", "MH / GD2", "GD3")


def wavelet_kernel(kind: str, scale: float) -> np.ndarray:
    """Return a zero-mean, L2-normalized discrete Gaussian-derivative wavelet.

    ``scale`` is expressed in sample points.  The supported preview wavelets are
    the first Gaussian derivative (GD1), the Mexican Hat / second derivative
    (MH / GD2), and the third Gaussian derivative (GD3).
    """
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Wavelet scale must be a finite number > 0.")
    if kind not in WAVELET_TYPES:
        raise ValueError(f"Unsupported wavelet: {kind}")

    radius = max(1, int(math.ceil(4.0 * scale)))
    t = np.arange(-radius, radius + 1, dtype=np.float64)
    u = t / scale
    g = np.exp(-0.5 * u * u)

    if kind == "GD1":
        k = -u * g
    elif kind == "MH / GD2":
        # Conventional Mexican Hat sign; the opposite sign is equivalent
        # for feature localization.
        k = (1.0 - u * u) * g
    else:  # GD3
        k = (3.0 * u - u ** 3) * g

    # Discrete truncation can leave a tiny DC component.
    k = k - np.mean(k)
    norm = float(np.linalg.norm(k))
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("Wavelet kernel collapsed at this scale.")
    return k / norm


def wavelet_transform_1d(signal: np.ndarray, kind: str, scale: float = 1.0) -> np.ndarray:
    """Convolve a 1-D signal with a preview wavelet using edge replication.

    The returned array has exactly the same length as ``signal``.
    """
    y = np.asarray(signal, dtype=np.float64)
    if y.ndim != 1 or y.size < 2:
        raise ValueError("Wavelet preview expects a one-dimensional signal.")
    if not np.all(np.isfinite(y)):
        raise ValueError("Wavelet preview signal contains non-finite values.")

    k = wavelet_kernel(kind, scale)
    radius = (len(k) - 1) // 2
    padded = np.pad(y, (radius, radius), mode="edge")
    # Reverse the kernel for mathematical convolution.  For display, the
    # sign convention is not scientifically material, but keeping a true
    # convolution makes the definition explicit.
    out = np.convolve(padded, k, mode="valid")
    return out.astype(np.float64, copy=False)


@dataclass(frozen=True)
class RangeSpec:
    min: float
    max: float

    def validate(self, name: str, *, strictly_positive: bool = False,
                 bounds: tuple[float, float] | None = None) -> None:
        if not np.isfinite(self.min) or not np.isfinite(self.max):
            raise ValueError(f"{name}: min and max must be finite numbers.")
        if self.min > self.max:
            raise ValueError(f"{name}: min must be <= max.")
        if strictly_positive and self.min <= 0:
            raise ValueError(f"{name}: values must be > 0.")
        if bounds is not None:
            lo, hi = bounds
            if self.min < lo or self.max > hi:
                raise ValueError(f"{name}: values must be within [{lo}, {hi}].")

    def draw(self, rng: np.random.Generator, size: int) -> np.ndarray:
        if self.min == self.max:
            return np.full(size, self.min, dtype=float)
        return rng.uniform(self.min, self.max, size=size)


@dataclass(frozen=True)
class GeneratorConfig:
    mode: str = "custom"  # "custom" or "cafa"
    x_min: float = 55.0
    x_max: float = 200.0
    n_points: int = 352

    mu1: RangeSpec = RangeSpec(100.0, 110.0)
    mu2: RangeSpec = RangeSpec(130.0, 140.0)
    sigma1: RangeSpec = RangeSpec(12.0, 18.0)
    sigma2: RangeSpec = RangeSpec(12.0, 18.0)
    A1: RangeSpec = RangeSpec(0.1, 1.0)
    A2: RangeSpec = RangeSpec(0.1, 1.0)

    alpha1: RangeSpec = RangeSpec(-5.0, 5.0)
    alpha2: RangeSpec = RangeSpec(-5.0, 5.0)
    eta1: RangeSpec = RangeSpec(0.0, 1.0)
    eta2: RangeSpec = RangeSpec(0.0, 1.0)

    selected_peak_types: tuple[str, ...] = ("Gaussian",)
    paired_peak_types: bool = True
    noise_enabled: bool = True
    noise_pct: RangeSpec = RangeSpec(0.0, 2.0)
    n_samples: int = 1000
    seed: int = 20260925
    generation_strategy: str = "independent"
    constrained_shape: str = "Gaussian"
    min_peak_distance_points: float = 10.0
    sigma_frac_min: float = 0.02
    sigma_frac_max: float = 0.06
    sigma_ratio_min: float = 0.5
    sigma_ratio_max: float = 2.0
    delta_w_min: float = 0.25
    delta_w_max: float = 3.0
    edge_sigma_margin: float = 4.0
    amplitude_max_min: float = 0.5
    amplitude_max_max: float = 1.0
    amplitude_ratio_min: float = 0.1
    amplitude_ratio_max: float = 1.0

    def validate(self) -> None:
        if self.mode not in {"custom", "cafa"}:
            raise ValueError("mode must be 'custom' or 'cafa'.")

        if not (1 <= int(self.n_samples) <= 100_000):
            raise ValueError("Number of samples must be between 1 and 100000.")

        if self.mode == "cafa":
            return

        if self.generation_strategy not in {"independent", "constrained", "constrained_mix"}:
            raise ValueError("Unknown generation strategy.")

        if self.generation_strategy in {"constrained", "constrained_mix"}:
            if int(self.n_points) <= 200:
                raise ValueError("Width-constrained mode requires more than 200 points.")
            allowed_shapes = {"Gaussian", "Lorentzian", "Pseudo-Voigt"}
            if self.constrained_shape not in allowed_shapes:
                raise ValueError("Constrained shape must be Gaussian, Lorentzian or Pseudo-Voigt.")
            if not self.selected_peak_types:
                raise ValueError("Select at least one peak type.")
            if any(k not in allowed_shapes for k in self.selected_peak_types):
                raise ValueError("Width-constrained modes support Gaussian, Lorentzian and Pseudo-Voigt only.")
            if self.min_peak_distance_points < 0:
                raise ValueError("Minimum peak distance must be >= 0.")
            self.A1.validate("A1", strictly_positive=True, bounds=(0.1, 1.0))
            self.A2.validate("A2", strictly_positive=True, bounds=(0.1, 1.0))
            if not np.isfinite(self.x_min) or not np.isfinite(self.x_max) or self.x_min >= self.x_max:
                raise ValueError("X min must be smaller than X max.")
            self.eta1.validate("eta1", bounds=(0.0, 1.0))
            self.eta2.validate("eta2", bounds=(0.0, 1.0))
            self.noise_pct.validate("noise", bounds=(0.0, 2.0))
            return

        if not np.isfinite(self.x_min) or not np.isfinite(self.x_max) or self.x_min >= self.x_max:
            raise ValueError("X min must be smaller than X max.")
        if int(self.n_points) < 2:
            raise ValueError("Number of points must be at least 2.")

        for name, spec in (
            ("mu1", self.mu1), ("mu2", self.mu2),
            ("sigma1", self.sigma1), ("sigma2", self.sigma2),
            ("A1", self.A1), ("A2", self.A2),
            ("alpha1", self.alpha1), ("alpha2", self.alpha2),
        ):
            spec.validate(name, strictly_positive=name.startswith("sigma") or name.startswith("A"))

        for name, spec in (("eta1", self.eta1), ("eta2", self.eta2)):
            spec.validate(name, bounds=(0.0, 1.0))

        self.noise_pct.validate("noise", bounds=(0.0, 2.0))
        if not self.selected_peak_types:
            raise ValueError("Select at least one peak type.")

        if self.mu1.min < self.x_min or self.mu1.max > self.x_max:
            raise ValueError("mu1 range must lie inside the X range.")
        if self.mu2.min < self.x_min or self.mu2.max > self.x_max:
            raise ValueError("mu2 range must lie inside the X range.")

        if not self.selected_peak_types:
            raise ValueError("Select at least one mathematical peak type.")
        unknown = set(self.selected_peak_types) - set(PEAK_TYPES)
        if unknown:
            raise ValueError(f"Unknown peak type(s): {sorted(unknown)}")


@dataclass
class GeneratedDataset:
    x: np.ndarray
    peak1: np.ndarray
    peak2: np.ndarray
    summed: np.ndarray
    labels: pd.DataFrame
    config: dict
    metadata: dict
    legacy_labels: np.ndarray | None = None

    def pipeline_payload(self) -> dict:
        """In-memory hand-off for the next module / Transformer adapter."""
        return {
            "spectra": self.summed,
            "labels": self.labels,
            "x_axis": self.x,
            "config": self.config,
            "metadata": self.metadata,
        }


def _make_axis(x_min: float, x_max: float, n_points: int) -> tuple[np.ndarray, dict]:
    # linspace includes both endpoints and avoids endpoint drift.
    x = np.linspace(float(x_min), float(x_max), int(n_points), dtype=np.float64)
    # Tiny endpoint guard for floating-point edge cases only.
    tol = max(1e-12, np.finfo(float).eps * max(abs(x_min), abs(x_max), 1.0) * 32)
    if x[0] < x_min - tol or x[-1] > x_max + tol:
        x[0] = x_min
        x[-1] = x_max
    meta = {
        "requested_range": [float(x_min), float(x_max)],
        "actual_range": [float(x[0]), float(x[-1])],
        "n_points": int(n_points),
        "dx": float((x[-1] - x[0]) / (len(x) - 1)),
        "endpoint_tolerance": tol,
    }
    return x, meta


def gaussian(x: np.ndarray, A: float, mu: float, sigma: float) -> np.ndarray:
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def lorentzian(x: np.ndarray, A: float, mu: float, sigma: float) -> np.ndarray:
    # sigma is the Lorentz scale parameter (HWHM), because a Lorentzian has no finite SD.
    return A / (1.0 + ((x - mu) / sigma) ** 2)


def skew_normal_peak(x: np.ndarray, A: float, mu: float, sigma: float, alpha: float) -> np.ndarray:
    z = (x - mu) / sigma
    phi = np.exp(-0.5 * z * z)
    Phi = 0.5 * (1.0 + erf(alpha * z / math.sqrt(2.0)))
    y = 2.0 * phi * Phi
    ymax = float(np.max(y))
    if ymax <= 0 or not np.isfinite(ymax):
        return np.zeros_like(x)
    return A * y / ymax


def pseudo_voigt(x: np.ndarray, A: float, mu: float, sigma: float, eta: float) -> np.ndarray:
    g = gaussian(x, 1.0, mu, sigma)
    l = lorentzian(x, 1.0, mu, sigma)
    y = (1.0 - eta) * g + eta * l
    ymax = float(np.max(y))
    return A * y / ymax if ymax > 0 else np.zeros_like(x)


def _render_peak(kind: str, x: np.ndarray, A: float, mu: float, sigma: float,
                 alpha: float, eta: float) -> np.ndarray:
    if kind == "Gaussian":
        return gaussian(x, A, mu, sigma)
    if kind == "Skew Normal":
        return skew_normal_peak(x, A, mu, sigma, alpha)
    if kind == "Lorentzian":
        return lorentzian(x, A, mu, sigma)
    if kind == "Pseudo-Voigt":
        return pseudo_voigt(x, A, mu, sigma, eta)
    raise ValueError(f"Unsupported peak type: {kind}")


def _add_noise_p2p(y: np.ndarray, pct: float, peak_height: float,
                   rng: np.random.Generator) -> np.ndarray:
    if pct <= 0:
        return y.copy()
    z = rng.normal(size=len(y))
    span = float(z.max() - z.min())
    if span <= 0:
        return y.copy()
    noise = (pct / 100.0) * peak_height * z / span
    return y + noise


def _config_to_jsonable(cfg: GeneratorConfig) -> dict:
    d = asdict(cfg)
    d["selected_peak_types"] = list(cfg.selected_peak_types)
    return d


def _generate_cafa(cfg: GeneratorConfig) -> GeneratedDataset:
    """
    Wrapper around the legacy CA-FA methodology. It intentionally mirrors the
    original RNG order so summed spectra and legacy labels are identical to
    cafa_legacy.generate_dataset(n, seed), while also retaining both components
    for the six-sample visual control panel.
    """
    rng = np.random.default_rng(cfg.seed)
    x = cafa_legacy.V.copy()

    peak1 = []
    peak2 = []
    summed = []
    legacy_rows = []
    rows = []

    for i in range(cfg.n_samples):
        c_ca = rng.uniform(1.0, 10.0)
        c_fa = rng.uniform(1.0, 10.0)

        p_ca = (cafa_legacy.A_CA(c_ca), cafa_legacy.m_CA(c_ca), cafa_legacy.sigma_CA(c_ca))
        p_fa = (cafa_legacy.A_FA(c_fa), cafa_legacy.m_FA(c_fa), cafa_legacy.sigma_FA(c_fa))

        ca_clean = cafa_legacy.gaussian(*p_ca)
        fa_clean = cafa_legacy.gaussian(*p_fa)

        ca = cafa_legacy.jw_noise_1pct(ca_clean, p_ca[0], rng)
        fa = cafa_legacy.jw_noise_1pct(fa_clean, p_fa[0], rng)
        s = ca + fa

        peak1.append(ca)
        peak2.append(fa)
        summed.append(s)

        legacy = [c_ca, c_fa, p_ca[0], p_ca[1], p_ca[2], p_fa[0], p_fa[1], p_fa[2]]
        legacy_rows.append(legacy)
        rows.append({
            "sample_index": i,
            "peak1_type": "CA Gaussian",
            "peak2_type": "FA Gaussian",
            "mu1": p_ca[1], "sigma1": p_ca[2], "A1": p_ca[0],
            "alpha1": np.nan, "eta1": np.nan,
            "mu2": p_fa[1], "sigma2": p_fa[2], "A2": p_fa[0],
            "alpha2": np.nan, "eta2": np.nan,
            "noise_pct1": 1.0, "noise_pct2": 1.0,
            "c_CA": c_ca, "c_FA": c_fa,
        })

    X = np.asarray(summed)
    Y = np.asarray(legacy_rows)
    p1 = np.asarray(peak1)
    p2 = np.asarray(peak2)

    axis_meta = {
        "requested_range": [float(x[0]), float(x[-1])],
        "actual_range": [float(x[0]), float(x[-1])],
        "n_points": int(len(x)),
        "dx": float(x[1] - x[0]),
        "endpoint_tolerance": 0.0,
    }
    metadata = {
        "mode": "CA-FA",
        "source": "legacy CA_FA_Gaussian_Generator_SHORT.py",
        "legacy_compatible": True,
        "axis": axis_meta,
        "n_samples": int(cfg.n_samples),
        "seed": int(cfg.seed),
        "noise_definition": "Independent normal noise per peak; peak-to-peak noise span = 1% of that peak height.",
        "pipeline_spectra_key": "summed",
    }
    return GeneratedDataset(
        x=x, peak1=p1, peak2=p2, summed=X,
        labels=pd.DataFrame(rows),
        config={"mode": "cafa", "n_samples": cfg.n_samples, "seed": cfg.seed,
                "locked_preset": True},
        metadata=metadata,
        legacy_labels=Y,
    )


def _uniform_or_const(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(lo if lo == hi else rng.uniform(lo, hi))


def _generate_constrained(cfg: GeneratorConfig) -> GeneratedDataset:
    cfg.validate()
    rng = np.random.default_rng(cfg.seed)
    x, axis_meta = _make_axis(cfg.x_min, cfg.x_max, cfg.n_points)
    n = int(cfg.n_points)
    dx = float((x[-1] - x[0]) / (n - 1))

    p1 = np.empty((cfg.n_samples, n), dtype=np.float32)
    p2 = np.empty((cfg.n_samples, n), dtype=np.float32)
    summed = np.empty((cfg.n_samples, n), dtype=np.float32)
    rows = []
    mix = np.asarray(["Gaussian", "Lorentzian", "Pseudo-Voigt"], dtype=object)

    for i in range(cfg.n_samples):
        ok = False
        for _ in range(1000):
            s1i = rng.uniform(cfg.sigma_frac_min*n, cfg.sigma_frac_max*n)
            s2i = rng.uniform(cfg.sigma_frac_min*n, cfg.sigma_frac_max*n)
            ratio = s1i/s2i
            if not (cfg.sigma_ratio_min <= ratio <= cfg.sigma_ratio_max):
                continue

            f1 = 2.355*s1i
            f2 = 2.355*s2i
            w = (f1+f2)/2.0
            dlo = max(cfg.delta_w_min*w, cfg.min_peak_distance_points)
            dhi = cfg.delta_w_max*w
            if dlo > dhi:
                raise ValueError("Minimum peak distance is incompatible with sampled widths.")

            d = _uniform_or_const(rng, dlo, dhi)
            cmin = cfg.edge_sigma_margin*s1i + d/2.0
            cmax = (n-1) - cfg.edge_sigma_margin*s2i - d/2.0
            if cmin >= cmax:
                continue

            c = rng.uniform(cmin, cmax)
            m1i = c-d/2.0
            m2i = c+d/2.0
            if m1i-cfg.edge_sigma_margin*s1i > 0 and m2i+cfg.edge_sigma_margin*s2i < n-1:
                ok = True
                break

        if not ok:
            raise ValueError("Could not generate a valid constrained pair after 1000 attempts.")

        # Independent amplitudes: broader coverage than the previous
        # Amax + Amin/Amax rule.
        A1 = float(cfg.A1.draw(rng, 1)[0])
        A2 = float(cfg.A2.draw(rng, 1)[0])

        if cfg.generation_strategy == "constrained_mix":
            if cfg.paired_peak_types:
                k = str(rng.choice(mix))
                k1 = k2 = k
            else:
                k1 = str(rng.choice(mix))
                k2 = str(rng.choice(mix))
        else:
            choices = np.asarray(cfg.selected_peak_types, dtype=object)
            if cfg.paired_peak_types:
                k = str(rng.choice(choices))
                k1 = k2 = k
            else:
                k1 = str(rng.choice(choices))
                k2 = str(rng.choice(choices))

        e1 = float(cfg.eta1.draw(rng, 1)[0])
        e2 = float(cfg.eta2.draw(rng, 1)[0])
        m1 = float(x[0] + m1i*dx)
        m2 = float(x[0] + m2i*dx)
        s1 = float(s1i*dx)
        s2 = float(s2i*dx)

        y1 = _render_peak(k1, x, A1, m1, s1, 0.0, e1)
        y2 = _render_peak(k2, x, A2, m2, s2, 0.0, e2)

        if cfg.noise_enabled:
            np1 = float(cfg.noise_pct.draw(rng, 1)[0])
            np2 = float(cfg.noise_pct.draw(rng, 1)[0])
        else:
            np1 = np2 = 0.0

        y1 = _add_noise_p2p(y1, np1, float(np.max(y1)), rng)
        y2 = _add_noise_p2p(y2, np2, float(np.max(y2)), rng)

        p1[i] = y1.astype(np.float32)
        p2[i] = y2.astype(np.float32)
        summed[i] = (y1+y2).astype(np.float32)

        rows.append({
            "sample_index": i,
            "peak1_type": k1, "peak2_type": k2,
            "mu1": m1, "sigma1": s1, "A1": A1, "alpha1": np.nan,
            "eta1": e1 if k1 == "Pseudo-Voigt" else np.nan,
            "mu2": m2, "sigma2": s2, "A2": A2, "alpha2": np.nan,
            "eta2": e2 if k2 == "Pseudo-Voigt" else np.nan,
            "noise_pct1": np1, "noise_pct2": np2,
            "c_CA": np.nan, "c_FA": np.nan,
            "mu1_index": m1i, "mu2_index": m2i,
            "sigma1_index": s1i, "sigma2_index": s2i,
            "FWHM1_index": f1, "FWHM2_index": f2,
            "mean_FWHM_index": w, "delta_mu_index": d,
            "center_index": c, "sigma_ratio": ratio,
            "amplitude_ratio": min(A1, A2) / max(A1, A2),
        })

    labels = pd.DataFrame(rows)
    metadata = {
        "mode": "custom",
        "generation_strategy": cfg.generation_strategy,
        "axis": axis_meta,
        "n_samples": int(cfg.n_samples),
        "seed": int(cfg.seed),
        "geometry_space": "point index",
        "sigma_fraction_of_n": [cfg.sigma_frac_min, cfg.sigma_frac_max],
        "sigma_ratio_range": [cfg.sigma_ratio_min, cfg.sigma_ratio_max],
        "delta_mu_over_mean_FWHM": [cfg.delta_w_min, cfg.delta_w_max],
        "minimum_peak_distance_points": cfg.min_peak_distance_points,
        "shape_sampling": (
            ("One of Gaussian/Lorentzian/Pseudo-Voigt is sampled per pair, p=1/3 each"
             if cfg.paired_peak_types
             else "Gaussian/Lorentzian/Pseudo-Voigt independently, p=1/3 each")
            if cfg.generation_strategy == "constrained_mix"
            else (
                "One selected type is sampled for the pair"
                if cfg.paired_peak_types
                else "Peak 1 and Peak 2 sample selected types independently"
            )
        ),
        "selected_peak_types": list(cfg.selected_peak_types),
        "paired_peak_types": bool(cfg.paired_peak_types),
        "amplitude_sampling": {
            "A1": [cfg.A1.min, cfg.A1.max],
            "A2": [cfg.A2.min, cfg.A2.max],
            "rule": "independent_uniform"
        },
        "pipeline_spectra_key": "summed",
    }
    return GeneratedDataset(
        x=x, peak1=p1, peak2=p2, summed=summed,
        labels=labels, config=_config_to_jsonable(cfg), metadata=metadata
    )

def _generate_custom(cfg: GeneratorConfig) -> GeneratedDataset:
    cfg.validate()
    rng = np.random.default_rng(cfg.seed)
    x, axis_meta = _make_axis(cfg.x_min, cfg.x_max, cfg.n_points)
    n = cfg.n_samples

    mu1 = cfg.mu1.draw(rng, n)
    mu2 = cfg.mu2.draw(rng, n)
    sigma1 = cfg.sigma1.draw(rng, n)
    sigma2 = cfg.sigma2.draw(rng, n)
    A1 = cfg.A1.draw(rng, n)
    A2 = cfg.A2.draw(rng, n)
    alpha1 = cfg.alpha1.draw(rng, n)
    alpha2 = cfg.alpha2.draw(rng, n)
    eta1 = cfg.eta1.draw(rng, n)
    eta2 = cfg.eta2.draw(rng, n)

    type_choices = np.asarray(cfg.selected_peak_types, dtype=object)
    if cfg.paired_peak_types:
        pair_kind = rng.choice(type_choices, size=n, replace=True)
        kind1 = pair_kind.copy()
        kind2 = pair_kind.copy()
    else:
        kind1 = rng.choice(type_choices, size=n, replace=True)
        kind2 = rng.choice(type_choices, size=n, replace=True)

    if cfg.noise_enabled:
        noise1 = cfg.noise_pct.draw(rng, n)
        noise2 = cfg.noise_pct.draw(rng, n)
    else:
        noise1 = np.zeros(n)
        noise2 = np.zeros(n)

    p1 = np.empty((n, len(x)), dtype=np.float32)
    p2 = np.empty((n, len(x)), dtype=np.float32)
    summed = np.empty((n, len(x)), dtype=np.float32)

    rows = []
    for i in range(n):
        y1 = _render_peak(str(kind1[i]), x, A1[i], mu1[i], sigma1[i], alpha1[i], eta1[i])
        y2 = _render_peak(str(kind2[i]), x, A2[i], mu2[i], sigma2[i], alpha2[i], eta2[i])

        y1n = _add_noise_p2p(y1, noise1[i], float(np.max(y1)), rng)
        y2n = _add_noise_p2p(y2, noise2[i], float(np.max(y2)), rng)

        p1[i] = y1n.astype(np.float32)
        p2[i] = y2n.astype(np.float32)
        summed[i] = (y1n + y2n).astype(np.float32)

        rows.append({
            "sample_index": i,
            "peak1_type": str(kind1[i]),
            "peak2_type": str(kind2[i]),
            "mu1": mu1[i], "sigma1": sigma1[i], "A1": A1[i],
            "alpha1": alpha1[i] if kind1[i] == "Skew Normal" else np.nan,
            "eta1": eta1[i] if kind1[i] == "Pseudo-Voigt" else np.nan,
            "mu2": mu2[i], "sigma2": sigma2[i], "A2": A2[i],
            "alpha2": alpha2[i] if kind2[i] == "Skew Normal" else np.nan,
            "eta2": eta2[i] if kind2[i] == "Pseudo-Voigt" else np.nan,
            "noise_pct1": noise1[i],
            "noise_pct2": noise2[i],
            "c_CA": np.nan,
            "c_FA": np.nan,
        })

    labels = pd.DataFrame(rows)
    metadata = {
        "mode": "custom",
        "axis": axis_meta,
        "n_samples": int(n),
        "seed": int(cfg.seed),
        "selected_peak_types": list(cfg.selected_peak_types),
        "type_sampling": (
            "One selected mathematical peak type is sampled per pair; both peaks share it."
            if cfg.paired_peak_types
            else "Peak 1 and Peak 2 are sampled independently with equal probability among selected mathematical peak types."
        ),
        "paired_peak_types": bool(cfg.paired_peak_types),
        "noise_definition": "Independent normal noise per peak; peak-to-peak noise span is the sampled percentage of that peak height.",
        "pipeline_spectra_key": "summed",
    }
    return GeneratedDataset(
        x=x, peak1=p1, peak2=p2, summed=summed,
        labels=labels,
        config=_config_to_jsonable(cfg),
        metadata=metadata,
    )


def generate_dataset(config: GeneratorConfig) -> GeneratedDataset:
    """Public programmatic entry point for the Transformer or any other module."""
    config.validate()
    if config.mode == "cafa":
        return _generate_cafa(config)
    if config.generation_strategy in {"constrained", "constrained_mix"}:
        return _generate_constrained(config)
    return _generate_custom(config)


def make_dataset_zip(dataset: GeneratedDataset, base_name: str) -> bytes:
    """Create a physical archive for the browser download / local copy."""
    safe = "".join(ch for ch in base_name.strip() if ch.isalnum() or ch in ("-", "_", ".")) or "double_peak_dataset"

    npz_buffer = BytesIO()
    np.savez_compressed(
        npz_buffer,
        x_axis=dataset.x,
        peak1=dataset.peak1,
        peak2=dataset.peak2,
        summed=dataset.summed,
    )

    labels_csv = dataset.labels.to_csv(index=False).encode("utf-8")

    with BytesIO() as bio:
        with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{safe}/spectra.npz", npz_buffer.getvalue())
            zf.writestr(f"{safe}/labels.csv", labels_csv)
            zf.writestr(f"{safe}/config.json", json.dumps(dataset.config, indent=2, ensure_ascii=False))
            zf.writestr(f"{safe}/metadata.json", json.dumps(dataset.metadata, indent=2, ensure_ascii=False))
            if dataset.legacy_labels is not None:
                legacy_df = pd.DataFrame(
                    dataset.legacy_labels,
                    columns=["c_CA", "c_FA", "A_CA", "mu_CA", "sigma_CA",
                             "A_FA", "mu_FA", "sigma_FA"]
                )
                zf.writestr(f"{safe}/legacy_labels.csv", legacy_df.to_csv(index=False).encode("utf-8"))
        return bio.getvalue()


def generate_dataset_chunk(config: GeneratorConfig, start: int, count: int, seed_offset: int = 0) -> GeneratedDataset:
    """
    Generate a deterministic chunk for interactive pause/continue workflows.

    For custom mode, each chunk uses seed + start so continuation is reproducible.
    For CA-FA mode, exact legacy identity across arbitrary chunk boundaries is not
    guaranteed by this helper; the GUI therefore generates CA-FA in one pass.
    """
    if count <= 0:
        raise ValueError("count must be > 0")
    if config.mode == "cafa":
        # Keep legacy CA-FA generation exact by generating from the beginning
        # and slicing the requested section.
        tmp = GeneratorConfig(mode="cafa", n_samples=start + count, seed=config.seed)
        full = _generate_cafa(tmp)
        sl = slice(start, start + count)
        return GeneratedDataset(
            x=full.x,
            peak1=full.peak1[sl],
            peak2=full.peak2[sl],
            summed=full.summed[sl],
            labels=full.labels.iloc[sl].reset_index(drop=True),
            config=full.config,
            metadata=full.metadata,
            legacy_labels=None if full.legacy_labels is None else full.legacy_labels[sl],
        )

    cfg = GeneratorConfig(
        mode=config.mode,
        x_min=config.x_min,
        x_max=config.x_max,
        n_points=config.n_points,
        mu1=config.mu1, mu2=config.mu2,
        sigma1=config.sigma1, sigma2=config.sigma2,
        A1=config.A1, A2=config.A2,
        alpha1=config.alpha1, alpha2=config.alpha2,
        eta1=config.eta1, eta2=config.eta2,
        selected_peak_types=config.selected_peak_types,
        paired_peak_types=config.paired_peak_types,
        noise_enabled=config.noise_enabled,
        noise_pct=config.noise_pct,
        n_samples=count,
        seed=int(config.seed + start + seed_offset),
        generation_strategy=config.generation_strategy,
        constrained_shape=config.constrained_shape,
        min_peak_distance_points=config.min_peak_distance_points,
        sigma_frac_min=config.sigma_frac_min,
        sigma_frac_max=config.sigma_frac_max,
        sigma_ratio_min=config.sigma_ratio_min,
        sigma_ratio_max=config.sigma_ratio_max,
        delta_w_min=config.delta_w_min,
        delta_w_max=config.delta_w_max,
        edge_sigma_margin=config.edge_sigma_margin,
        amplitude_max_min=config.amplitude_max_min,
        amplitude_max_max=config.amplitude_max_max,
        amplitude_ratio_min=config.amplitude_ratio_min,
        amplitude_ratio_max=config.amplitude_ratio_max,
    )
    ds = _generate_constrained(cfg) if cfg.generation_strategy in {"constrained", "constrained_mix"} else _generate_custom(cfg)
    ds.labels["sample_index"] = np.arange(start, start + count)
    return ds


def concatenate_datasets(parts: list[GeneratedDataset], full_config: dict | None = None) -> GeneratedDataset:
    if not parts:
        raise ValueError("No dataset parts to concatenate.")
    x = parts[0].x
    for p in parts[1:]:
        if not np.array_equal(x, p.x):
            raise ValueError("All dataset chunks must share the same x-axis.")
    peak1 = np.concatenate([p.peak1 for p in parts], axis=0)
    peak2 = np.concatenate([p.peak2 for p in parts], axis=0)
    summed = np.concatenate([p.summed for p in parts], axis=0)
    labels = pd.concat([p.labels for p in parts], ignore_index=True)
    legacy = None
    if all(p.legacy_labels is not None for p in parts):
        legacy = np.concatenate([p.legacy_labels for p in parts], axis=0)
    meta = dict(parts[0].metadata)
    meta["n_samples"] = int(len(summed))
    meta["partial_or_chunked"] = True
    return GeneratedDataset(
        x=x, peak1=peak1, peak2=peak2, summed=summed,
        labels=labels,
        config=full_config or parts[0].config,
        metadata=meta,
        legacy_labels=legacy,
    )
