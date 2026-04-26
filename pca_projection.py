from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PCAProjection:
    components_: np.ndarray
    mean_: np.ndarray
    n_features_in_: int
    n_components_: int

    @classmethod
    def from_incremental_pca(cls, pca: object) -> "PCAProjection":
        components = np.asarray(getattr(pca, "components_"), dtype=np.float32)
        mean = np.asarray(
            getattr(pca, "mean_", np.zeros(components.shape[1], dtype=np.float32)),
            dtype=np.float32,
        )
        n_features_in = int(getattr(pca, "n_features_in_", components.shape[1]))
        n_components = int(getattr(pca, "n_components_", components.shape[0]))
        return cls(
            components_=components,
            mean_=mean,
            n_features_in_=n_features_in,
            n_components_=n_components,
        )

    def transform(self, x: np.ndarray) -> np.ndarray:
        matrix = np.asarray(x, dtype=np.float32)
        centered = matrix - self.mean_
        return centered @ self.components_.T
