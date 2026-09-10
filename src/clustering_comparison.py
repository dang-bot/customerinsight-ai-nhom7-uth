"""Module TV3-C: Multi-Metric Benchmark & Algorithm Comparison.
Thành viên phụ trách: TH (Trần Lê Thái Học).
Cung cấp phân tích toàn diện 3 chỉ số:
- Silhouette Score (Độ tách biệt cụm, càng cao càng tốt, [-1, 1])
- Calinski-Harabasz Index (Variance Ratio Criterion, càng cao cụm càng đặc và tách rời)
- Davies-Bouldin Index (Độ tương đồng cụm, càng thấp càng tốt, >= 0)
Cung cấp hàm so sánh tự động đa thuật toán: K-Means, DBSCAN, Hierarchical, GMM.
"""

from __future__ import annotations

from typing import Any
import pandas as pd
from numpy.typing import ArrayLike
import numpy as np

from src.clustering import _matrix, run_kmeans
from src.advanced_clustering import (
    calculate_metrics,
    run_dbscan,
    run_hierarchical,
    run_gmm,
)


def evaluate_kmeans_comprehensive(
    X_scaled: ArrayLike,
    k_min: int = 2,
    k_max: int = 10,
) -> pd.DataFrame:
    """Đánh giá K-Means trên dải K với đầy đủ 3 chỉ số Silhouette, Calinski-Harabasz, Davies-Bouldin."""
    matrix = _matrix(X_scaled)
    records = []
    for k in range(k_min, k_max + 1):
        km = run_kmeans(matrix, k)
        sil, ch, db = calculate_metrics(matrix, km.labels)
        records.append({
            "k": k,
            "inertia": km.inertia,
            "silhouette": sil,
            "calinski_harabasz": ch,
            "davies_bouldin": db,
        })
    return pd.DataFrame(records)


def _safe_float(val: float | None, digits: int = 4) -> float | None:
    if val is None or not np.isfinite(val):
        return None
    return round(float(val), digits)


def benchmark_all_algorithms(
    X_scaled: ArrayLike,
    target_k: int = 4,
    dbscan_eps: float = 0.6,
    dbscan_min_samples: int = 5,
) -> pd.DataFrame:
    """So sánh song song 4 thuật toán trên cùng ma trận dữ liệu chuẩn hóa.
    
    Thuật toán gồm:
    1. K-Means
    2. Hierarchical (Ward Linkage)
    3. GMM (Full Covariance)
    4. DBSCAN
    """
    matrix = _matrix(X_scaled)
    results = []

    # 1. K-Means
    km_fit = run_kmeans(matrix, target_k)
    km_sil, km_ch, km_db = calculate_metrics(matrix, km_fit.labels)
    results.append({
        "Algorithm": "K-Means",
        "Type": "Centroid-based",
        "Clusters": target_k,
        "NoisePoints": 0,
        "Silhouette": _safe_float(km_sil, 4),
        "Calinski_Harabasz": _safe_float(km_ch, 2),
        "Davies_Bouldin": _safe_float(km_db, 4),
        "Advantages": "Nhanh, dễ giải thích, tối ưu phân tách hình cầu",
        "Disadvantages": "Nhạy cảm với outlier, giả định cụm hình cầu lồi",
    })

    # 2. Hierarchical
    h_fit = run_hierarchical(matrix, n_clusters=target_k, linkage="ward")
    results.append({
        "Algorithm": "Hierarchical (Ward)",
        "Type": "Connectivity-based",
        "Clusters": h_fit.n_clusters,
        "NoisePoints": 0,
        "Silhouette": _safe_float(h_fit.silhouette, 4),
        "Calinski_Harabasz": _safe_float(h_fit.calinski_harabasz, 2),
        "Davies_Bouldin": _safe_float(h_fit.davies_bouldin, 4),
        "Advantages": "Tạo cấu trúc phả hệ (dendrogram), không cần random init",
        "Disadvantages": "Độ phức tạp O(N^2), chậm với dataset cực lớn",
    })

    # 3. GMM
    gmm_fit = run_gmm(matrix, n_components=target_k)
    results.append({
        "Algorithm": "Gaussian Mixture (GMM)",
        "Type": "Probabilistic (Distribution)",
        "Clusters": gmm_fit.n_clusters,
        "NoisePoints": 0,
        "Silhouette": _safe_float(gmm_fit.silhouette, 4),
        "Calinski_Harabasz": _safe_float(gmm_fit.calinski_harabasz, 2),
        "Davies_Bouldin": _safe_float(gmm_fit.davies_bouldin, 4),
        "Advantages": "Hỗ trợ cụm hình elip, xác suất thành viên mềm (soft clustering)",
        "Disadvantages": "Có thể hội tụ cực tiểu cục bộ (EM algorithm)",
    })

    # 4. DBSCAN
    db_fit = run_dbscan(matrix, eps=dbscan_eps, min_samples=dbscan_min_samples)
    results.append({
        "Algorithm": "DBSCAN",
        "Type": "Density-based",
        "Clusters": db_fit.n_clusters,
        "NoisePoints": db_fit.n_noise,
        "Silhouette": _safe_float(db_fit.silhouette, 4),
        "Calinski_Harabasz": _safe_float(db_fit.calinski_harabasz, 2),
        "Davies_Bouldin": _safe_float(db_fit.davies_bouldin, 4),
        "Advantages": "Tự tìm số cụm, phát hiện nhiễu và hình dạng bất kỳ",
        "Disadvantages": "Khó chọn tham số eps & min_samples khi mật độ chênh lệch",
    })

    df = pd.DataFrame(results)
    return df

