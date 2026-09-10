"""Module TV3-B: Multi-Algorithm Clustering Engine.
Thành viên phụ trách: NH (Nguyễn Quang Hiếu).
Cung cấp các thuật toán nâng cao: DBSCAN, Hierarchical (Agglomerative), GMM (Gaussian Mixture Model).
Đảm bảo tính nhất quán về interface để so sánh và benchmark trực tiếp với K-Means.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from src.clustering import _matrix


@dataclass(frozen=True)
class ClusteringFitResult:
    """Kết quả fit chuẩn hoá cho mọi thuật toán clustering."""
    algorithm_name: str
    labels: NDArray[np.int_]
    n_clusters: int
    n_noise: int
    silhouette: float | None
    calinski_harabasz: float | None
    davies_bouldin: float | None
    model: Any
    metadata: dict[str, Any]


def calculate_metrics(matrix: NDArray[np.float64], labels: NDArray[np.int_]) -> tuple[float | None, float | None, float | None]:
    """Tính toán bộ 3 chỉ số đánh giá phân cụ: Silhouette, Calinski-Harabasz, Davies-Bouldin.
    
    Bỏ qua điểm nhiễu (label == -1) nếu có khi tính Silhouette / CH / DB.
    """
    valid_mask = labels >= 0
    valid_labels = labels[valid_mask]
    valid_matrix = matrix[valid_mask]

    n_unique = len(np.unique(valid_labels))
    if n_unique < 2 or len(valid_matrix) <= n_unique:
        return None, None, None

    try:
        sil = float(silhouette_score(valid_matrix, valid_labels))
    except Exception:
        sil = None

    try:
        ch = float(calinski_harabasz_score(valid_matrix, valid_labels))
    except Exception:
        ch = None

    try:
        db = float(davies_bouldin_score(valid_matrix, valid_labels))
    except Exception:
        db = None

    return sil, ch, db


def run_dbscan(
    X_scaled: ArrayLike,
    eps: float = 0.5,
    min_samples: int = 5,
) -> ClusteringFitResult:
    """Chạy thuật toán phân cụ dựa trên mật độ DBSCAN.
    
    Phù hợp phát hiện cụm hình dạng bất kỳ và định vị outlier/noise points (label = -1).
    """
    matrix = _matrix(X_scaled)
    if eps <= 0:
        raise ValueError("eps phải lớn hơn 0.")
    if min_samples < 1:
        raise ValueError("min_samples phải >= 1.")

    model = DBSCAN(eps=float(eps), min_samples=int(min_samples))
    labels = model.fit_predict(matrix)

    n_noise = int(np.sum(labels == -1))
    unique_clusters = set(labels) - {-1}
    n_clusters = len(unique_clusters)

    sil, ch, db = calculate_metrics(matrix, labels)

    return ClusteringFitResult(
        algorithm_name="DBSCAN",
        labels=labels,
        n_clusters=n_clusters,
        n_noise=n_noise,
        silhouette=sil,
        calinski_harabasz=ch,
        davies_bouldin=db,
        model=model,
        metadata={"eps": eps, "min_samples": min_samples, "algorithm": "density-based"}
    )


def run_hierarchical(
    X_scaled: ArrayLike,
    n_clusters: int = 4,
    linkage: str = "ward",
) -> ClusteringFitResult:
    """Chạy thuật toán phân cụ phân cấp Hierarchical (Agglomerative Clustering).
    
    linkage hỗ trợ: 'ward', 'complete', 'average', 'single'.
    """
    matrix = _matrix(X_scaled)
    if n_clusters < 2 or n_clusters > len(matrix):
        raise ValueError(f"n_clusters phải nằm trong khoảng [2, {len(matrix)}].")
    if linkage not in {"ward", "complete", "average", "single"}:
        raise ValueError("linkage phải là một trong: 'ward', 'complete', 'average', 'single'.")

    model = AgglomerativeClustering(n_clusters=int(n_clusters), linkage=linkage)
    labels = model.fit_predict(matrix)

    sil, ch, db = calculate_metrics(matrix, labels)

    return ClusteringFitResult(
        algorithm_name="Hierarchical (Agglomerative)",
        labels=labels,
        n_clusters=int(n_clusters),
        n_noise=0,
        silhouette=sil,
        calinski_harabasz=ch,
        davies_bouldin=db,
        model=model,
        metadata={"linkage": linkage, "algorithm": "hierarchical"}
    )


def run_gmm(
    X_scaled: ArrayLike,
    n_components: int = 4,
    covariance_type: str = "full",
    random_state: int = 42,
) -> ClusteringFitResult:
    """Chạy mô hình hỗn hợp Gauss (Gaussian Mixture Models - GMM).
    
    Mô hình xác suất cho phép soft clustering (phân cụm mềm với độ tin cậy xác suất).
    """
    matrix = _matrix(X_scaled)
    if n_components < 2 or n_components > len(matrix):
        raise ValueError(f"n_components phải nằm trong khoảng [2, {len(matrix)}].")
    if covariance_type not in {"full", "tied", "diag", "spherical"}:
        raise ValueError("covariance_type không hợp lệ.")

    model = GaussianMixture(
        n_components=int(n_components),
        covariance_type=covariance_type,
        random_state=int(random_state),
    )
    labels = model.fit_predict(matrix)

    sil, ch, db = calculate_metrics(matrix, labels)

    return ClusteringFitResult(
        algorithm_name="Gaussian Mixture Models (GMM)",
        labels=labels,
        n_clusters=int(n_components),
        n_noise=0,
        silhouette=sil,
        calinski_harabasz=ch,
        davies_bouldin=db,
        model=model,
        metadata={
            "covariance_type": covariance_type,
            "aic": float(model.aic(matrix)),
            "bic": float(model.bic(matrix)),
            "converged": bool(model.converged_),
            "algorithm": "probabilistic",
        }
    )
