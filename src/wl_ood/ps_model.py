"""Cosmology-conditioned Gaussian model for log-power features."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import minimize


def fit_moments(
    features: np.ndarray,
    realization_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(features[:, realization_indices], dtype=np.float64)
    if selected.ndim != 3:
        raise ValueError("features must have shape (cosmology, realization, feature)")
    means = selected.mean(axis=1)
    n_realizations = selected.shape[1]
    n_features = selected.shape[2]
    denominator = n_realizations - n_features - 2
    if denominator <= 0:
        raise ValueError("not enough realizations for covariance correction")
    delta = selected - means[:, None, :]
    covariances = np.einsum("crf,crg->cfg", delta, delta) / denominator
    return means, covariances


@dataclass(frozen=True)
class FitResult:
    parameters: np.ndarray
    chi_squared: float
    objective: float
    evaluations: int


class PowerSpectrumModel:
    def __init__(
        self,
        cosmologies: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
    ) -> None:
        self.cosmologies = np.asarray(cosmologies, dtype=np.float64)
        self.means = np.asarray(means, dtype=np.float64)
        self.covariances = np.asarray(covariances, dtype=np.float64)
        self._validate_shapes()
        for index, covariance in enumerate(self.covariances):
            if not np.all(np.isfinite(covariance)):
                raise ValueError(f"covariance {index} is not finite")
            if not np.allclose(covariance, covariance.T):
                raise ValueError(f"covariance {index} is not symmetric")
            try:
                np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError as error:
                raise ValueError(
                    f"covariance {index} is not positive definite"
                ) from error

        self._mean_interp = LinearNDInterpolator(
            self.cosmologies,
            self.means,
            fill_value=np.nan,
        )
        self._cov_interp = LinearNDInterpolator(
            self.cosmologies,
            self.covariances,
            fill_value=np.nan,
        )

    def _validate_shapes(self) -> None:
        if self.cosmologies.ndim != 2 or self.cosmologies.shape[1] != 2:
            raise ValueError("cosmologies must have shape (count, 2)")
        if self.means.ndim != 2:
            raise ValueError("means must have shape (count, features)")
        expected_covariance_shape = (
            self.cosmologies.shape[0],
            self.means.shape[1],
            self.means.shape[1],
        )
        if self.means.shape[0] != self.cosmologies.shape[0]:
            raise ValueError("cosmology and mean counts must match")
        if self.covariances.shape != expected_covariance_shape:
            raise ValueError(
                f"covariances must have shape {expected_covariance_shape}"
            )

    def _moments(
        self,
        parameters: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        mean = np.asarray(
            self._mean_interp(parameters),
            dtype=np.float64,
        ).reshape(-1)
        feature_count = self.means.shape[1]
        covariance = np.asarray(
            self._cov_interp(parameters),
            dtype=np.float64,
        ).reshape(feature_count, feature_count)
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
            return None
        return mean, covariance

    def chi_squared(
        self,
        parameters: np.ndarray,
        features: np.ndarray,
    ) -> float:
        moments = self._moments(parameters)
        if moments is None:
            return float("inf")
        mean, covariance = moments
        delta = np.asarray(features, dtype=np.float64) - mean
        try:
            solved = np.linalg.solve(covariance, delta)
        except np.linalg.LinAlgError:
            return float("inf")
        return float(delta @ solved)

    def negative_log_likelihood(
        self,
        parameters: np.ndarray,
        features: np.ndarray,
    ) -> float:
        moments = self._moments(parameters)
        if moments is None:
            return float("inf")
        mean, covariance = moments
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0 or not np.isfinite(logdet):
            return float("inf")
        delta = np.asarray(features, dtype=np.float64) - mean
        try:
            chi_squared = float(delta @ np.linalg.solve(covariance, delta))
        except np.linalg.LinAlgError:
            return float("inf")
        return 0.5 * logdet + 0.5 * chi_squared

    def score_one(self, features: np.ndarray) -> FitResult:
        feature_vector = np.asarray(features, dtype=np.float64)
        if feature_vector.shape != (self.means.shape[1],):
            raise ValueError("feature vector has the wrong shape")
        if not np.all(np.isfinite(feature_vector)):
            raise ValueError("feature vector must be finite")

        result = minimize(
            lambda parameters: self.negative_log_likelihood(
                parameters,
                feature_vector,
            ),
            x0=np.array([0.3, 0.8]),
            method="Nelder-Mead",
        )
        if not result.success:
            raise RuntimeError(f"MAP optimization failed: {result.message}")
        objective = self.negative_log_likelihood(result.x, feature_vector)
        chi_squared = self.chi_squared(result.x, feature_vector)
        if (
            not np.isfinite(objective)
            or not np.isfinite(chi_squared)
            or chi_squared < 0.0
        ):
            raise RuntimeError("MAP optimization did not produce a valid score")
        return FitResult(
            parameters=np.asarray(result.x, dtype=np.float64),
            chi_squared=chi_squared,
            objective=objective,
            evaluations=int(result.nfev),
        )

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output,
            cosmologies=self.cosmologies,
            means=self.means,
            covariances=self.covariances,
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> "PowerSpectrumModel":
        with np.load(path, allow_pickle=False) as payload:
            return cls(
                payload["cosmologies"],
                payload["means"],
                payload["covariances"],
            )
