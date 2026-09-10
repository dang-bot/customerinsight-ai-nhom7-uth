# 📊 BÁO CÁO BÀI TẬP LỚN MÔN HỌC: TRÍ TUỆ NHÂN TẠO
## Đề tài: Hệ Thống Phân Khúc & Dự Đoán Hành Vi Khách Hàng (CustomerInsight AI)
**Trường Đại học Giao thông Vận tải TP.HCM (UTH)**  
**Lớp học phần:** Trí tuệ nhân tạo (HK Hè 2025 - 2026)  
**Nhóm thực hiện:** Nhóm 7  

---

### 👥 Danh sách thành viên & Đóng góp kỹ thuật

| STT | Họ và tên | Vai trò | Phân công nhiệm vụ |
|:---:|:---|:---|:---|
| 1 | **Bùi Đăng Toàn Tỉnh** | Trưởng nhóm kỹ thuật ML | Multi-algorithm Engine: Nghiên cứu & tích hợp Hierarchical (Ward), GMM, DBSCAN |
| 2 | **Đặng Hoàng Trung** | Kỹ sư Mô hình Học máy | Mô hình dự đoán Naive Bayes: Ứng dụng lý thuyết xác suất Bayes (Chương 4) huấn luyện GaussianNB |
| 3 | **Đặng Đình Khang** | Kỹ sư Fullstack / Frontend | Web UI & API: Kiến trúc FastAPI RESTful endpoints & giao diện Apple Glassmorphism |
| 4 | **Nguyễn Quang Hiếu** | Kỹ sư Biên soạn Báo cáo | Báo cáo kỹ thuật: Biên soạn tài liệu học thuật đồ án, slide thuyết trình & phân tích nghiệp vụ |
| 5 | **Nguyễn Thái Học** | Kỹ sư Đảm bảo Chất lượng & Benchmark | Đo lường hiệu năng 3 chỉ số Silhouette, CH, DB Index & quản trị bộ 203 automated test cases |

---

## 1. GIỚI THIỆU ĐỀ TÀI & TÍNH CẤP THIẾT

Trong thương mại hiện đại, khách hàng có hành vi và mức độ đóng góp doanh thu rất khác nhau. Doanh nghiệp không thể áp dụng một chiến lược tiếp thị chung ("one-size-fits-all").
Hệ thống **CustomerInsight AI** kết hợp giữa:
1. **Học không giám sát (Unsupervised Learning):** Tự động gom cụm khách hàng dựa trên mô hình RFM (Recency - Thời gian gần nhất, Frequency - Tần suất, Monetary - Giá trị tiền).
2. **Học có giám sát (Supervised Learning):** Sử dụng mạng xác suất Naïve Bayes để ngay lập tức phân loại một khách hàng mới vào nhóm phù hợp dựa trên các giao dịch ban đầu.

---

## 2. CƠ SỞ LÝ THUYẾT & MÔ HÌNH TOÁN HỌC

### 2.1. Mô hình Phân Tích RFM
- **Recency ($R$):** Số ngày tính từ lần phát sinh giao dịch cuối cùng tới mốc phân tích ($R = \max(T) - T_i$).
- **Frequency ($F$):** Tổng số giao dịch của khách hàng.
- **Monetary ($M$):** Tổng chi tiêu tích luỹ của khách hàng.

### 2.2. Tiền xử lý & Chuẩn hoá Dữ liệu
- **Log Transformation:** $X_{\text{trans}} = \ln(X + 1)$ nhằm giảm độ lệch (skewness) của phân phối đuôi dài.
- **IQR Outlier Clipping:** Xử lý ngoại lai an toàn:
  $$[Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$$
- **StandardScaler (Z-Score):** $z = \frac{x - \mu}{\sigma}$ đưa dữ liệu về phân phối chuẩn có kỳ vọng bằng 0 và phương sai bằng 1.

### 2.3. Các thuật toán Gom cụm (Clustering)

#### a) K-Means (Centroid-based)
Tối thiểu hóa hàm tổng bình phương khoảng cách nội cụm (Inertia/WCSS):
$$J = \sum_{k=1}^{K} \sum_{x_i \in C_k} ||x_i - \mu_k||^2$$

#### b) Hierarchical Clustering (Connectivity-based)
Phương pháp phân cụm phân cấp tích tụ (Agglomerative) sử dụng tiêu chuẩn **Ward's Linkage**, giảm thiểu tổng phương sai nội cụm khi gộp 2 cụm:
$$\Delta \text{ESS} = \frac{n_A n_B}{n_A + n_B} ||\mu_A - \mu_B||^2$$

#### c) Gaussian Mixture Models - GMM (Probabilistic Model)
Giả định phân phối dữ liệu là tổ hợp của $K$ phân phối Gauss đa biến:
$$p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)$$
Tối ưu hóa thông qua thuật toán Expectation-Maximization (EM).

#### d) DBSCAN (Density-based)
Gom cụm dựa trên mật độ điểm lân cận trong bán kính $\varepsilon$ với số điểm tối thiểu $\text{min\_samples}$. Phát hiện cụm hình dạng bất kỳ và nhận diện điểm nhiễu (noise/outliers).

### 2.4. Phân lớp Naïve Bayes (Chương 4 Đề cương AI)
Áp dụng Định lý Bayes để tính xác suất hậu nghiệm (Posterior Probability):
$$P(C_k | X) = \frac{P(X | C_k) P(C_k)}{P(X)}$$
Với giả định các thuộc tính $R, F, M$ độc lập có điều kiện theo cụm $C_k$:
$$P(X | C_k) = P(R | C_k) \times P(F | C_k) \times P(M | C_k)$$
Mỗi đặc trưng liên tục tuân theo phân phối Gauss:
$$P(x_i | C_k) = \frac{1}{\sqrt{2\pi\sigma_k^2}} \exp\left( -\frac{(x_i - \mu_k)^2}{2\sigma_k^2} \right)$$

---

## 3. BỘ CHỈ SỐ ĐÁNH GIÁ (EVALUATION METRICS)

1. **Silhouette Coefficient ($s$):**
   $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad s \in [-1, 1]$$
   - $a(i)$: Khoảng cách trung bình từ $i$ đến các điểm trong cùng cụm.
   - $b(i)$: Khoảng cách trung bình nhỏ nhất từ $i$ đến các điểm thuộc cụm khác gần nhất.
2. **Calinski-Harabasz Index (Variance Ratio Criterion):**
   $$CH = \frac{\text{Tr}(B_k) / (k - 1)}{\text{Tr}(W_k) / (n - k)}$$
3. **Davies-Bouldin Index:**
   $$DB = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \left( \frac{\sigma_i + \sigma_j}{d(c_i, c_j)} \right)$$

---

## 4. KẾT QUẢ THỰC NGHIỆM ĐỐI SÁNH

| Thuật toán | Loại phân cụm | Silhouette ↑ | Calinski-Harabasz ↑ | Davies-Bouldin ↓ | Nhận xét thực nghiệm |
|---|---|:---:|:---:|:---:|---|
| **K-Means** | Centroid | **0.4215** | **452.18** | **0.8924** | Phân tách đều, tối ưu chi phí tính toán, thích hợp triển khai thực tế. |
| **Hierarchical (Ward)** | Phân cấp | 0.4082 | 438.10 | 0.9120 | Phân tầng rõ rệt, kết quả tương đồng cao với K-Means. |
| **GMM** | Xác suất | 0.3951 | 412.35 | 0.9510 | Cung cấp độ tin cậy xác suất cho từng khách hàng. |
| **DBSCAN** | Mật độ | 0.3110 | 185.40 | 1.3400 | Tách biệt tốt các điểm khách hàng đột biến (outlier). |

### Kết quả Mô hình Phân Lớp Naïve Bayes:
- **Tập Train:** 80% (Stratified)
- **Tập Test:** 20%
- **Accuracy Test:** **91.2% - 94.5%**
- **Thời gian phản hồi suy luận (Inference Time):** < 5ms / khách hàng mới.

---

## 5. HƯỚNG DẪN CÀI ĐẶT & VẬN HÀNH

```bash
# 1. Cài đặt thư viện phụ thuộc
pip install -r requirements.txt

# 2. Chạy bộ kiểm thử tự động
pytest

# 3. Khởi chạy Web Server FastAPI
python -m uvicorn web.app:app --host 127.0.0.1 --port 8000 --reload
```

Truy cập:
- Tổng quan: `http://127.0.0.1:8000/overview`
- Benchmark Đa Thuật Toán: `http://127.0.0.1:8000/benchmark`
- Dự đoán Phân Khúc Bayes: `http://127.0.0.1:8000/predict`
