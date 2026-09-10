# 🖥️ KỊCH BẢN THUYẾT TRÌNH & SLIDE DECK (10 - 15 PHÚT)
## Đề tài: CustomerInsight AI — Hệ Thống Phân Khúc & Dự Đoán Hành Vi Khách Hàng
**Nhóm 7 — Môn Trí Tuệ Nhân Tạo (UTH)**

---

### SLIDE 1: GIỚI THIỆU & THÀNH VIÊN
- **Tên đề tài:** CustomerInsight AI — Phân Khúc & Dự Đoán Khách Hàng Bằng Học Máy & Lý Thuyết Xác Suất Bayes
- **Giảng viên hướng dẫn:** Bộ môn Trí tuệ Nhân tạo - Trường ĐH Giao thông Vận tải TP.HCM
- **Thành viên Nhóm 7:**
| Thành viên | Vai trò | Phụ trách chính |
|---|---|---|
| **Bùi Đăng Toàn Tỉnh** | Multi-algorithm Engine | Nghiên cứu & tích hợp Hierarchical (Ward), GMM, DBSCAN |
| **Đặng Hoàng Trung** | Naive Bayes Predictor | Ứng dụng lý thuyết Bayes (Chương 4) dự đoán phân khúc tức thì |
| **Đặng Đình Khang** | Web UI & API | Kiến trúc FastAPI RESTful endpoints & Apple Glassmorphism UI |
| **Nguyễn Quang Hiếu** | Technical Report | Báo cáo kỹ thuật học thuật, kịch bản slide thuyết trình |
| **Nguyễn Thái Học** | Benchmark & QA | Đo lường hiệu năng 3 chỉ số & bộ 203 automated test cases |

---

### SLIDE 2: BÀI TOÁN KINH DOANH & MỤC TIÊU KỸ THUẬT
- **Bài toán:** Phân hóa hành vi khách hàng từ tập dữ liệu bán lẻ (RFM).
- **Mục tiêu:**
  - Không dừng lại ở 1 thuật toán cơ bản: Triển khai và so sánh **4 thuật toán học máy**.
  - Không chỉ dừng ở phân cụm mô tả (Descriptive): Mở rộng sang **mô hình dự đoán thời gian thực (Predictive AI)** với Naïve Bayes.
  - Xây dựng sản phẩm hoàn chỉnh: Kiến trúc FastAPI + Giao diện Web + 200+ Unit/Integration Tests.

---

### SLIDE 3: QUY TRÌNH TIỀN XỬ LÝ & PIPELINE DỮ LIỆU
- **Mô hình RFM:** Recency, Frequency, Monetary.
- **Quy trình chuẩn hóa:**
  1. Loại bỏ bản ghi thiếu / lỗi định dạng / CustomerID trùng.
  2. Xử lý lệch phân phối bằng $\ln(X + 1)$.
  3. IQR Outlier Clipping để bảo toàn cấu trúc phân phối.
  4. Z-Score Standardization về $\mu = 0, \sigma = 1$.

---

### SLIDE 4: ĐỐI SÁNH 4 THUẬT TOÁN PHÂN CỤM (NH & TH)
| Thuật toán | Cơ chế | Ưu điểm thực nghiệm | Nhược điểm |
|---|---|---|---|
| **K-Means** | Trọng tâm (Centroid) | Nhanh, Silhouette cao (0.42) | Cụm hình cầu |
| **Hierarchical** | Phân cấp tích tụ | Cấu trúc cây phả hệ rõ | $O(N^2)$ tính toán |
| **GMM** | Hỗn hợp Gauss | Soft-clustering, xác suất tin cậy | Nhạy khởi tạo |
| **DBSCAN** | Dựa trên mật độ | Tách biệt điểm dị biệt (noise) | Nhạy tham số $\varepsilon$ |

---

### SLIDE 5: ỨNG DỤNG ĐỊNH LÝ BAYES DỰ ĐOÁN THỜI GIAN THỰC (DK)
- **Cơ sở lý thuyết:** Chương 4 Giáo trình AI — Phân lớp Bayes:
  $$P(\text{Cluster} | R, F, M) = \frac{P(R, F, M | \text{Cluster}) \times P(\text{Cluster})}{P(R, F, M)}$$
- **Tính năng độc đáo:** Khi có khách hàng mới nhập vào hệ thống, mô hình đưa ra ngay phân khúc kèm độ tin cậy phần trăm và gợi ý hành động cụ thể cho marketing.
- **Độ chính xác:** Test Accuracy đạt trên **91%**.

---

### SLIDE 6: TRÌNH DIỄN HỆ THỐNG LIVE DEMO (BT & ĐT)
- Demo 7 màn hình trên Web Dashboard:
  1. Khám phá dữ liệu (EDA) tương tác
  2. Phân tích Elbow & Silhouette chọn $K$
  3. Mô hình phân cụm 2D/3D
  4. **Màn hình Benchmark 4 thuật toán (`/benchmark`)**
  5. **Màn hình Dự đoán Khách hàng mới bằng Bayes (`/predict`)**
  6. Xuất kết quả CSV & Báo cáo chất lượng

---

### SLIDE 7: KẾT LUẬN & HƯỚNG PHÁT TRIỂN
- Hoàn thành trọn vẹn yêu cầu môn học với chất lượng vượt trội.
- Kiến trúc phần mềm chuẩn mực, tài liệu rõ ràng, phân công minh bạch 5 thành viên.
- Sẵn sàng mở rộng: Kết nối cơ sở dữ liệu thời gian thực (PostgreSQL) và tích hợp thêm Deep Learning.
