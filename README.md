# CUSTOMERINSIGHT AI — HỆ THỐNG PHÂN KHÚC & DỰ ĐOÁN HÀNH VI KHÁCH HÀNG THÔNG MINH
> **BÁO CÁO ĐỒ ÁN MÔN HỌC: TRÍ TUỆ NHÂN TẠO (ARTIFICIAL INTELLIGENCE)**  
> **TRƯỜNG ĐẠI HỌC GIAO THÔNG VẬN TẢI TP.HCM (UTH) — VIỆN CÔNG NGHỆ THÔNG TIN**  
> **NHÓM THỰC HIỆN: NHÓM 7 (LỚP HỌC PHẦN: 2025-2026)**

---

## 🌟 Giới Thiệu Tổng Quan
Dự án **CustomerInsight AI** là một giải pháp học máy (Machine Learning) kết hợp chuyên sâu giữa **Học không giám sát (Unsupervised Learning)** và **Học có giám sát (Supervised Learning)** nhằm giải quyết bài toán khai phá dữ liệu khách hàng (Customer Segmentation) trong thương mại điện tử hiện đại.

Hệ thống được phát triển theo chuẩn kiến trúc khoa học dữ liệu hoàn chỉnh từ thu thập, chuẩn hóa, phân tích thăm dò (EDA), tối ưu hóa số cụm, đến đối sánh đa mô hình (Benchmark) và suy diễn xác suất hậu nghiệm theo thời gian thực (Real-time Inference).

---

## 👥 Phân Công Nhiệm Vụ Thành Viên (Nhóm 7)

| STT | Thành viên | Nhiệm vụ chuyên môn | Module phụ trách |
|:---:|:---|:---|:---|
| 1 | **Bùi Đăng Toàn Tỉnh** | Multi-algorithm Engine | Hiện thực hóa & tích hợp **Hierarchical (Ward)**, **GMM**, **DBSCAN** |
| 2 | **Đặng Hoàng Trung** | Mô hình dự đoán Naive Bayes | Ứng dụng lý thuyết xác suất Bayes (Chương 4) huấn luyện **Gaussian Naïve Bayes** |
| 3 | **Đặng Đình Khang** | Web UI & API | Xây dựng RESTful API FastAPI & thiết kế giao diện **Apple Glassmorphism** |
| 4 | **Nguyễn Quang Hiếu** | Báo cáo kỹ thuật | Phụ trách tài liệu kỹ thuật, slide thuyết trình & tổng hợp phân tích |
| 5 | **Trần Lê Thái Học** | Đánh giá & Benchmark (QA) | Đo lường hiệu năng 3 chỉ số Silhouette, CH, DB Index & bộ kiểm thử tự động 203 tests |

---

## 🚀 Điểm Khác Biệt & Đóng Góp Học Thuật Nổi Bật

Khác biệt hoàn toàn so với các bài toán gom cụm cơ bản chỉ dùng K-Means đơn lẻ:

1. **Đấu trường Đối sánh 4 Thuật toán (Multi-Algorithm Benchmark):**
   - **K-Means++:** Chuẩn mực phân hoạch tối ưu khoảng cách Euclidean.
   - **Hierarchical (Ward):** Phân cụm thứ bậc với ma trận liên kết phương sai nhỏ nhất.
   - **Gaussian Mixture Models (GMM):** Phân cụm mềm (Soft Clustering) dựa trên thuật toán Cực đại hóa Kỳ vọng (Expectation-Maximization).
   - **DBSCAN (Density-Based Spatial Clustering):** Nhận diện hình dạng cụm bất kỳ và cô lập các điểm ngoại lai bất thường (Whales/Outliers).

2. **Đánh Giá Khoa Học Trên 3 Thước Đo Toàn Diện:**
   - **Silhouette Score:** Đo lường mức độ gắn kết nội cụm so với khoảng cách cụm lân cận gần nhất ($s \in [-1, 1]$).
   - **Calinski-Harabasz Index (Variance Ratio Criterion):** Tỷ số phân tán giữa các cụm và trong từng cụm (càng cao càng tốt).
   - **Davies-Bouldin Index:** Đo lường độ tương đồng trung bình giữa mỗi cụm và cụm gần giống nhất (càng thấp càng tốt).

3. **Tích Hợp Lý Thuyết Xác Suất Bayes Thời Gian Thực (Chương 4):**
   - Ứng dụng công thức định lý Bayes để tính xác suất hậu nghiệm:
     $$P(\text{Cluster}_k \mid R, F, M) = \frac{P(\text{Cluster}_k) \prod_{i=1}^{3} P(X_i \mid \text{Cluster}_k)}{P(R, F, M)}$$
   - Cho phép hệ thống gán nhãn tức thì ($<2\text{ms}$) cho khách hàng mới mà không cần chạy lại thuật toán phân cụm toàn cục tốn kém.

4. **Giao Diện Apple Glassmorphism Luxury:**
   - Thiết kế kính mờ trong suốt (*frosted glass*), nền gradient cực quang (*Aurora Mesh*) và không gian 3D tương tác hạt (*Three.js Particle Cloud*).

---

## 📂 Cấu Trúc Mã Nguồn

```text
customerinsight-ai/
├── src/                               # Lõi thuật toán Machine Learning
│   ├── advanced_clustering.py         # Cụm mở rộng: DBSCAN, Hierarchical, GMM
│   ├── bayes_classifier.py            # Gaussian Naïve Bayes Classifier
│   ├── clustering_comparison.py       # Benchmark 3 chỉ số Silhouette, CH, DB
│   ├── clustering.py                  # Thuật toán K-Means++ cốt lõi
│   ├── preprocessing.py               # Xử lý khuyết thiếu, ngoại lai, chuẩn hóa Z-score
│   ├── profiling.py                   # Phân tích hồ sơ RFM đa chiều
│   └── validation.py                  # Kiểm định toàn vẹn schema dữ liệu
├── web/                               # Ứng dụng Web FastAPI
│   ├── app.py                         # RESTful API & Server Router
│   ├── templates/                     # 8 Giao diện Apple Glassmorphism
│   │   ├── overview.html              # Tổng quan dự án & 3D Orb
│   │   ├── data.html                  # Nạp & kiểm định chất lượng RFM
│   │   ├── eda.html                   # Phân tích tương quan & phân phối
│   │   ├── choose_k.html              # Phương pháp Elbow & Silhouette tìm K
│   │   ├── clustering.html            # Không gian 3D Scatter Plot
│   │   ├── benchmark.html             # Đấu trường đối sánh 4 mô hình
│   │   ├── predict.html               # Trình suy diễn Bayes thời gian thực
│   │   └── results.html               # Bảng tổng hợp hồ sơ khách hàng & Xuất CSV
│   └── static/                        # CSS Glassmorphism & JavaScript Controllers
├── tests/                             # Bộ kiểm thử tự động toàn diện
│   ├── test_nh_th_dk_advanced.py      # Kiểm thử thuật toán nâng cao & Bayes
│   ├── test_clustering.py
│   ├── test_preprocessing.py
│   └── ...                            # 203 automated test cases
├── docs/                              # Tài liệu học thuật đồ án
│   ├── BaoCao_Nhom7_CustomerInsightAI.md # Báo cáo chi tiết (35 trang)
│   └── Slide_ThuyetTrinh_Nhom7.md     # Kịch bản bảo vệ trước hội đồng
└── requirements.txt                   # Danh mục thư viện phụ thuộc
```

---

## ⚙️ Hướng Dẫn Cài Đặt & Vận Hành

### 1. Yêu cầu hệ thống
- Python 3.11 hoặc 3.12
- Hệ điều hành: Windows / macOS / Linux

### 2. Cài đặt môi trường
```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Cài đặt thư viện phụ thuộc
pip install -r requirements.txt
```

### 3. Chạy kiểm thử tự động (Verification)
```bash
pytest
# Kết quả mong đợi: 203 passed (100%)
```

### 4. Khởi chạy máy chủ
```bash
uvicorn web.app:app --host 127.0.0.1 --port 8000 --reload
```
Truy cập hệ thống tại: **http://127.0.0.1:8000**

---
*Bản quyền học thuật thuộc về Nhóm 7 — Khóa học Trí Tuệ Nhân Tạo, Trường ĐH Giao Thông Vận Tải TP.HCM (UTH).*
