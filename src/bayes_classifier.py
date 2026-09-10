"""Module TV3-D: Naïve Bayes Classifier for Customer Segment Prediction.
Thành viên phụ trách: DK (Đặng Đình Khang).
Mục đích:
- Ứng dụng lý thuyết xác suất Bayes (Chương 4 trong giáo trình AI UTH):
  P(Cluster | R, F, M) = P(R, F, M | Cluster) * P(Cluster) / P(R, F, M)
- Huấn luyện Gaussian Naïve Bayes trên tập dữ liệu khách hàng đã được gán nhãn cụm.
- Đánh giá mô hình phân loại: Accuracy, Confusion Matrix, Classification Report.
- Dự đoán tức thì (real-time inference) segment cho khách hàng mới dựa trên Recency, Frequency, Monetary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


@dataclass
class BayesModelResult:
    """Artifacts của mô hình Gaussian Naïve Bayes."""
    model: GaussianNB
    accuracy: float
    train_accuracy: float
    confusion_matrix: list[list[int]]
    classes: list[int]
    classification_report_dict: dict[str, Any]
    feature_names: list[str]


def train_bayes_classifier(
    df_customers: pd.DataFrame,
    feature_cols: list[str] = ["Recency", "Frequency", "Monetary"],
    target_col: str = "Cluster",
    test_size: float = 0.2,
    random_state: int = 42,
) -> BayesModelResult:
    """Huấn luyện mô hình GaussianNB phân loại phân khúc khách hàng."""
    if target_col not in df_customers.columns:
        raise ValueError(f"Thiếu cột nhãn '{target_col}' trong dữ liệu.")
    for col in feature_cols:
        if col not in df_customers.columns:
            raise ValueError(f"Thiếu cột đặc trưng '{col}' trong dữ liệu.")

    clean_df = df_customers.dropna(subset=feature_cols + [target_col])
    # Bỏ các nhãn nhiễu (-1 nếu có)
    clean_df = clean_df[clean_df[target_col] >= 0]

    X = clean_df[feature_cols].to_numpy(dtype=float)
    y = clean_df[target_col].to_numpy(dtype=int)

    if len(X) < 10:
        raise ValueError("Dữ liệu quá ít để huấn luyện mô hình phân loại.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    train_preds = gnb.predict(X_train)
    test_preds = gnb.predict(X_test)

    train_acc = float(accuracy_score(y_train, train_preds))
    test_acc = float(accuracy_score(y_test, test_preds))

    classes = [int(c) for c in gnb.classes_]
    cm = confusion_matrix(y_test, test_preds, labels=classes).tolist()
    report = classification_report(y_test, test_preds, output_dict=True, zero_division=0)

    return BayesModelResult(
        model=gnb,
        accuracy=test_acc,
        train_accuracy=train_acc,
        confusion_matrix=cm,
        classes=classes,
        classification_report_dict=report,
        feature_names=feature_cols,
    )


def predict_single_customer(
    model: GaussianNB,
    recency: float,
    frequency: float,
    monetary: float,
    segment_names: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Dự đoán phân khúc và xác suất hậu nghiệm (Posterior Probability) cho 1 khách hàng mới."""
    input_vector = np.array([[float(recency), float(frequency), float(monetary)]])
    predicted_cluster = int(model.predict(input_vector)[0])
    probabilities = model.predict_proba(input_vector)[0]

    prob_map = {
        int(cls): round(float(prob), 4)
        for cls, prob in zip(model.classes_, probabilities)
    }

    seg_name = segment_names.get(predicted_cluster, f"Cluster {predicted_cluster}") if segment_names else f"Cluster {predicted_cluster}"

    return {
        "predicted_cluster": predicted_cluster,
        "segment_name": seg_name,
        "confidence": round(float(np.max(probabilities)), 4),
        "posterior_probabilities": prob_map,
        "input": {
            "Recency": recency,
            "Frequency": frequency,
            "Monetary": monetary,
        }
    }
