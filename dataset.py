from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from sklearn.model_selection import train_test_split


class DatasetGenerationError(ValueError):
    """Raised when generated data is invalid or insufficient."""


@dataclass
class DatasetBundle:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    x_train_np: np.ndarray
    y_train_np: np.ndarray
    x_test_np: np.ndarray
    y_test_np: np.ndarray
    x_clean: np.ndarray
    y_clean: np.ndarray
    x_plot: np.ndarray
    y_plot: np.ndarray
    removed_points: int


@dataclass(frozen=True)
class NormalizationStats:
    """Mean/std statistics used to train in normalized coordinates."""

    x_mean: float
    x_std: float
    y_mean: float
    y_std: float

    @classmethod
    def from_training_arrays(
        cls,
        x_values: np.ndarray,
        y_values: np.ndarray,
    ) -> "NormalizationStats":
        return cls(
            x_mean=float(np.mean(x_values)),
            x_std=cls._safe_std(x_values),
            y_mean=float(np.mean(y_values)),
            y_std=cls._safe_std(y_values),
        )

    @staticmethod
    def _safe_std(values: np.ndarray) -> float:
        std = float(np.std(values))
        if not np.isfinite(std) or std < 1e-12:
            return 1.0
        return std

    def normalize_x_np(self, values: np.ndarray) -> np.ndarray:
        return (values - self.x_mean) / self.x_std

    def normalize_y_np(self, values: np.ndarray) -> np.ndarray:
        return (values - self.y_mean) / self.y_std

    def denormalize_y_np(self, values: np.ndarray) -> np.ndarray:
        return values * self.y_std + self.y_mean

    def normalize_x_value(self, value: float) -> float:
        return (value - self.x_mean) / self.x_std

    def denormalize_y_value(self, value: float) -> float:
        return value * self.y_std + self.y_mean

    def mse_to_original_scale(self, mse: float | None) -> float | None:
        if mse is None:
            return None
        return mse * (self.y_std ** 2)

    def to_dict(self) -> dict[str, float]:
        return {
            "x_mean": self.x_mean,
            "x_std": self.x_std,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
        }


def _evaluate_function(
    function: Callable[[np.ndarray], np.ndarray],
    x_values: np.ndarray,
) -> np.ndarray:
    with np.errstate(all="ignore"):
        y_values = np.asarray(function(x_values), dtype=np.float64)

    if y_values.ndim == 0:
        y_values = np.full_like(x_values, float(y_values), dtype=np.float64)

    y_values = np.reshape(y_values, -1)
    if y_values.size != x_values.size:
        raise DatasetGenerationError(
            "The function did not return one y value for each x value."
        )
    return y_values


def generate_dataset(
    function: Callable[[np.ndarray], np.ndarray],
    x_min: float,
    x_max: float,
    num_points: int,
    noise_level: float = 0.0,
    test_fraction: float = 0.2,
    seed: int | None = None,
    plot_points: int = 1000,
) -> DatasetBundle:
    if x_min >= x_max:
        raise DatasetGenerationError("x minimum must be smaller than x maximum.")
    if num_points < 10:
        raise DatasetGenerationError("Use at least 10 training points.")
    if noise_level < 0:
        raise DatasetGenerationError("Noise level cannot be negative.")

    x_values = np.linspace(x_min, x_max, num_points, dtype=np.float64)
    y_values = _evaluate_function(function, x_values)

    if noise_level > 0:
        rng = np.random.default_rng(seed)
        y_values = y_values + rng.normal(0.0, noise_level, size=y_values.shape)

    finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
    x_clean = x_values[finite_mask]
    y_clean = y_values[finite_mask]
    removed_points = int(num_points - x_clean.size)

    if x_clean.size < 10:
        raise DatasetGenerationError(
            "Too few finite data points remain. Adjust the x-range or function domain."
        )

    test_count = max(1, int(round(x_clean.size * test_fraction)))
    test_count = min(test_count, x_clean.size - 1)

    x_train, x_test, y_train, y_test = train_test_split(
        x_clean,
        y_clean,
        test_size=test_count,
        random_state=seed,
        shuffle=True,
    )

    x_plot = np.linspace(x_min, x_max, plot_points, dtype=np.float64)
    y_plot = _evaluate_function(function, x_plot)
    plot_mask = np.isfinite(x_plot) & np.isfinite(y_plot)
    x_plot = x_plot[plot_mask]
    y_plot = y_plot[plot_mask]
    if x_plot.size < 2:
        x_plot = x_clean.copy()
        y_plot = y_clean.copy()

    return DatasetBundle(
        x_train=torch.tensor(x_train.reshape(-1, 1), dtype=torch.float32),
        y_train=torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32),
        x_test=torch.tensor(x_test.reshape(-1, 1), dtype=torch.float32),
        y_test=torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32),
        x_train_np=x_train,
        y_train_np=y_train,
        x_test_np=x_test,
        y_test_np=y_test,
        x_clean=x_clean,
        y_clean=y_clean,
        x_plot=x_plot,
        y_plot=y_plot,
        removed_points=removed_points,
    )
