"""Kiểm thử chuyên sâu cho các tính năng nâng cao:
- NH: DBSCAN, Hierarchical, GMM (src/advanced_clustering.py)
- TH: Multi-Metric Benchmark (src/clustering_comparison.py)
- DK: Naïve Bayes Customer Prediction (src/bayes_classifier.py)
"""

import numpy as np
import pandas as pd
import pytest

from src.advanced_clustering import (
    calculate_metrics,
    run_dbscan,
    run_hierarchical,
    run_gmm,
)
from src.clustering_comparison import (
    benchmark_all_algorithms,
    evaluate_kmeans_comprehensive,
)
from src.bayes_classifier import (
    train_bayes_classifier,
    predict_single_customer,
)


@pytest.fixture
def synthetic_clusters() -> np.ndarray:
    rng = np.random.default_rng(42)
    return np.vstack([
        rng.normal(0.0, 0.2, (40, 3)),
        rng.normal(5.0, 0.2, (40, 3)),
        rng.normal(10.0, 0.2, (40, 3)),
    ])


def test_hierarchical_clustering(synthetic_clusters):
    result = run_hierarchical(synthetic_clusters, n_clusters=3, linkage="ward")
    assert result.algorithm_name == "Hierarchical (Agglomerative)"
    assert result.n_clusters == 3
    assert len(result.labels) == len(synthetic_clusters)
    assert result.silhouette is not None
    assert result.silhouette > 0.6  # Cụm phân tách rất rõ ràng


def test_gmm_clustering(synthetic_clusters):
    result = run_gmm(synthetic_clusters, n_components=3, random_state=42)
    assert "Gaussian Mixture" in result.algorithm_name
    assert result.n_clusters == 3
    assert len(result.labels) == len(synthetic_clusters)
    assert result.silhouette is not None
    assert result.metadata["converged"] is True


def test_dbscan_clustering(synthetic_clusters):
    result = run_dbscan(synthetic_clusters, eps=1.0, min_samples=5)
    assert result.algorithm_name == "DBSCAN"
    assert result.n_clusters >= 2
    assert len(result.labels) == len(synthetic_clusters)


def test_benchmark_all_algorithms(synthetic_clusters):
    df = benchmark_all_algorithms(synthetic_clusters, target_k=3)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    algorithms = df["Algorithm"].tolist()
    assert any("K-Means" in a for a in algorithms)
    assert any("Hierarchical" in a for a in algorithms)
    assert any("Gaussian" in a for a in algorithms)
    assert any("DBSCAN" in a for a in algorithms)
    assert "Silhouette" in df.columns
    assert "Calinski_Harabasz" in df.columns
    assert "Davies_Bouldin" in df.columns


def test_evaluate_kmeans_comprehensive(synthetic_clusters):
    df = evaluate_kmeans_comprehensive(synthetic_clusters, k_min=2, k_max=4)
    assert len(df) == 3
    assert set(df["k"]) == {2, 3, 4}
    assert "calinski_harabasz" in df.columns
    assert "davies_bouldin" in df.columns


def test_naive_bayes_classifier():
    # Tạo dữ liệu giả lập có nhãn
    rng = np.random.default_rng(42)
    c0 = rng.normal([10, 2, 50], 2, (30, 3))
    c1 = rng.normal([100, 15, 500], 10, (30, 3))
    c2 = rng.normal([200, 30, 1500], 20, (30, 3))
    
    data = np.vstack([c0, c1, c2])
    labels = np.array([0]*30 + [1]*30 + [2]*30)
    
    df = pd.DataFrame(data, columns=["Recency", "Frequency", "Monetary"])
    df["Cluster"] = labels
    
    model_res = train_bayes_classifier(df, test_size=0.2, random_state=42)
    assert model_res.accuracy >= 0.8
    assert len(model_res.classes) == 3
    
    # Test predict single customer
    pred = predict_single_customer(
        model_res.model,
        recency=12,
        frequency=2,
        monetary=48,
        segment_names={0: "Khách vãng lai", 1: "Khách tiềm năng", 2: "Khách VIP"}
    )
    assert pred["predicted_cluster"] == 0
    assert pred["segment_name"] == "Khách vãng lai"
    assert "confidence" in pred
    assert sum(pred["posterior_probabilities"].values()) == pytest.approx(1.0, rel=1e-2)
