# 📑 BÁO CÁO BÀI TẬP LỚN MÔN HỌC: TRÍ TUỆ NHÂN TẠO (AI)

---

**BỘ GIAO THÔNG VẬN TẢI**  
**TRƯỜNG ĐẠI HỌC GIAO THÔNG VẬN TẢI TP.HCM (UTH)**  
**VIỆN CÔNG NGHỆ THÔNG TIN — BỘ MÔN TRÍ TUỆ NHÂN TẠO**

---

### **ĐỀ TÀI:**  
# **HỆ THỐNG PHÂN KHÚC & DỰ ĐOÁN HÀNH VI KHÁCH HÀNG THÔNG MINH**  
### *(CUSTOMERINSIGHT AI: MULTI-ALGORITHM CLUSTERING & REAL-TIME GAUSSIAN NAÏVE BAYES INFERENCE)*

* **Học phần:** Trí Tuệ Nhân Tạo (Mã HP: 121033)
* **Học kỳ:** HK Hè — Năm học 2025 - 2026
* **Nhóm sinh viên thực hiện:** **Nhóm 7**

---

### 👥 BẢNG PHÂN CÔNG NHIỆM VỤ THÀNH VIÊN (NHÓM 7)

| STT | Họ và Tên | Vai trò chuyên môn | Nhiệm vụ chi tiết | Mức độ hoàn thành |
|:---:|:---|:---|:---|:---:|
| 1 | **Bùi Đăng Toàn Tỉnh** | Trưởng nhóm kỹ thuật ML | Nghiên cứu cơ sở toán học & lập trình động cơ đa thuật toán: **Hierarchical (Ward's Linkage)**, **Gaussian Mixture Models (GMM/EM)** và **DBSCAN** xử lý ngoại lai (`src/advanced_clustering.py`). | 100% |
| 2 | **Đặng Hoàng Trung** | Kỹ sư Mô hình Học máy | Xây dựng mô hình học có giám sát **Gaussian Naïve Bayes** dựa trên Chương 4 Giáo trình AI UTH, huấn luyện và tối ưu bộ suy diễn phân khúc tức thì (`src/bayes_classifier.py`). | 100% |
| 3 | **Đặng Đình Khang** | Kỹ sư Fullstack / Frontend | Thiết kế kiến trúc **FastAPI RESTful Endpoints**, đồng bộ hóa State luồng xử lý và phát triển giao diện người dùng cao cấp **Apple Glassmorphism / Frosting Luxury**. | 100% |
| 4 | **Nguyễn Quang Hiếu** | Kỹ sư Báo cáo Học thuật | Biên tập toàn diện báo cáo kỹ thuật khoa học, kịch bản bảo vệ trước hội đồng, phân tích kinh doanh đa chiều và các khuyến nghị chiến lược tiếp thị. | 100% |
| 5 | **Nguyễn Thái Học** | Kỹ sư Đánh giá & QA | Xây dựng pipeline đo lường đối sánh 3 chỉ số khoa học (**Silhouette, Calinski-Harabasz, Davies-Bouldin**) và phát triển bộ kiểm thử tự động **203 Test Cases** (`pytest`). | 100% |

---

## MỤC LỤC CHI TIẾT
1. **CHƯƠNG 1: TỔNG QUAN ĐỀ TÀI & TÍNH CẤP THIẾT**
   - 1.1. Bối cảnh thương mại điện tử & Thách thức quản trị khách hàng
   - 1.2. Hạn chế của các phương pháp truyền thống
   - 1.3. Mục tiêu và đóng góp của đề tài
2. **CHƯƠNG 2: CƠ SỞ LÝ THUYẾT & MÔ HÌNH TOÁN HỌC**
   - 2.1. Mô hình Phân tích Khách hàng RFM (Recency - Frequency - Monetary)
   - 2.2. Quy trình Xử lý & Chuẩn hóa Dữ liệu Khoa học
   - 2.3. Khảo sát 4 Thuật toán Gom cụm (Unsupervised Learning)
     - 2.3.1. Thuật toán K-Means++ (Centroid-based)
     - 2.3.2. Thuật toán Phân cụm Phân cấp Tích tụ (Hierarchical Ward's Linkage)
     - 2.3.3. Thuật toán Mô hình Hỗn hợp Gauss - GMM (Probabilistic Clustering)
     - 2.3.4. Thuật toán Gom cụm dựa trên Mật độ - DBSCAN
   - 2.4. Học có giám sát: Bộ phân lớp Gaussian Naïve Bayes (Chương 4 Đề cương AI UTH)
3. **CHƯƠNG 3: BỘ CHỈ SỐ ĐO LƯỜNG & ĐỐI SÁNH KHOA HỌC**
   - 3.1. Phương pháp Điểm uốn (Elbow Method)
   - 3.2. Hệ số Silhouette (Silhouette Coefficient)
   - 3.3. Chỉ số Calinski-Harabasz (Variance Ratio Criterion)
   - 3.4. Chỉ số Davies-Bouldin (Cluster Separation Measure)
4. **CHƯƠNG 4: THIẾT KẾ KIẾN TRÚC HỆ THỐNG & GIAO DIỆN**
   - 4.1. Kiến trúc phân tầng (Layered Architecture)
   - 4.2. Thiết kế RESTful API (FastAPI Backend)
   - 4.3. Giao diện Người dùng Apple Glassmorphism / Frosting Luxury
5. **CHƯƠNG 5: THỰC NGHIỆM, ĐỐI SÁNH & ĐÁNH GIÁ KẾT QUẢ**
   - 5.1. Dữ liệu Thực nghiệm (Dataset Online Retail)
   - 5.2. Kết quả Xác định Số Cụm K Tối Ưu
   - 5.3. Bảng Ma trận Đối sánh 4 Thuật toán
   - 5.4. Kết quả Huấn luyện & Đánh giá Bộ phân loại Naïve Bayes
   - 5.5. Chân dung Khách hàng 360° & Đề xuất Chiến lược Kinh doanh
6. **CHƯƠNG 6: KẾT LUẬN & HƯỚNG PHÁT TRIỂN**
   - 6.1. Tổng kết kết quả đạt được
   - 6.2. Hướng mở rộng đề tài

---

## CHƯƠNG 1: TỔNG QUAN ĐỀ TÀI & TÍNH CẤP THIẾT

### 1.1. Bối cảnh thương mại điện tử & Thách thức quản trị khách hàng
Trong kỷ nguyên kinh tế số và bùng nổ thương mại điện tử (E-Commerce), dữ liệu giao dịch của người tiêu dùng phát sinh liên tục với tốc độ chóng mặt. Tuy nhiên, giá trị đóng góp của các nhóm khách hàng vào tổng doanh thu của doanh nghiệp có sự phân hóa cực kỳ sâu sắc (tuân theo quy luật Pareto 80/20: 80% doanh thu đến từ 20% khách hàng trung thành).

Nếu doanh nghiệp tiếp tục áp dụng các chiến dịch tiếp thị đại trà (*Mass Marketing - "One-size-fits-all"*), hệ quả sẽ là:
* **Lãng phí chi phí marketing:** Phân bổ ngân sách khuyến mãi cho những khách hàng đã rời bỏ (Churned) hoặc không có nhu cầu.
* **Suy giảm trải nghiệm người dùng:** Gửi thông điệp quảng cáo không phù hợp, dẫn đến hiện tượng "Spam fatigue" khiến khách hàng bỏ theo dõi thương hiệu.
* **Bỏ lỡ khách hàng tiềm năng:** Không kịp thời chăm sóc đặc biệt cho nhóm khách hàng có giá trị cao (VIP/Whales).

### 1.2. Hạn chế của các phương pháp truyền thống
Trước đây, các doanh nghiệp thường phân nhóm khách hàng bằng quy tắc thủ công (Rule-based) trên bảng tính Excel, ví dụ: "Ai mua trên 1 triệu là VIP". Cách tiếp cận này có nhược điểm chí mạng:
1. **Mang tính cảm tính và phi tuyến tính:** Không thể nắm bắt được mối quan hệ tương tác đa chiều giữa thời gian giao dịch gần nhất, tần suất mua và tổng chi tiêu.
2. **Không thể mở rộng (Unscalable):** Khi tập dữ liệu lên đến hàng trăm nghìn giao dịch, việc tính toán thủ công hoàn toàn bất khả thi.
3. **Thiếu khả năng phản ứng thời gian thực:** Mỗi khi có khách hàng mới, doanh nghiệp không thể biết ngay họ thuộc chân dung nào để kích hoạt chương trình chào mừng phù hợp.

### 1.3. Mục tiêu và đóng góp của đề tài
Dự án **CustomerInsight AI** của **Nhóm 7** được xây dựng nhằm giải quyết triệt để các vấn đề trên thông qua:
* **Xây dựng động cơ đa thuật toán (Multi-algorithm Engine):** Triển khai đồng thời 4 mô hình học máy không giám sát (**K-Means++, Hierarchical Ward, Gaussian Mixture Models, DBSCAN**) để đối sánh và tìm ra cấu trúc phân cụm tối ưu nhất.
* **Đưa lý thuyết xác suất Bayes vào thực tiễn:** Ứng dụng lý thuyết học có giám sát **Gaussian Naïve Bayes** (nội dung trọng tâm trong Chương 4 Giáo trình AI UTH) nhằm phân lớp tự động tức thì cho khách hàng mới phát sinh giao dịch mà không cần chạy lại toàn bộ mô hình gom cụm.
* **Tạo dựng hệ thống Dashboard tương tác đẳng cấp:** Xây dựng ứng dụng Web hoàn chỉnh theo ngôn ngữ **Apple Glassmorphism / Frosting Luxury**, trực quan hóa không gian cụm 3 chiều tương tác xoay 360 độ và cung cấp các báo cáo hành động cụ thể cho bộ phận Marketing.

---

## CHƯƠNG 2: CƠ SỞ LÝ THUYẾT & MÔ HÌNH TOÁN HỌC

### 2.1. Mô hình Phân tích Khách hàng RFM
Mô hình RFM là phương pháp kinh điển trong khoa học tiếp thị định lượng, chuyển đổi lịch sử giao dịch mua sắm thành vector đặc trưng 3 chiều $\mathbf{x} = [R, F, M]^T$:

1. **Recency ($R$ - Độ mới của giao dịch):**  
   Số ngày tính từ giao dịch gần nhất của khách hàng đến mốc thời gian phân tích $T_{\max}$:
   $$R_i = \max(T) - T_i^{\text{last}}$$
   *Ý nghĩa:* $R$ càng nhỏ chứng tỏ khách hàng mới mua sắm gần đây, mức độ gắn kết với thương hiệu còn rất cao.

2. **Frequency ($F$ - Tần suất giao dịch):**  
   Tổng số lần phát sinh đơn hàng thành công của khách hàng trong chu kỳ quan sát:
   $$F_i = \sum_{j} \mathbb{I}(\text{Invoice}_j \in \text{Customer}_i)$$
   *Ý nghĩa:* $F$ cao biểu thị khách hàng có thói quen mua sắm định kỳ, có khả năng trở thành khách hàng thân thiết.

3. **Monetary ($M$ - Giá trị tiền tệ tích luỹ):**  
   Tổng số tiền mà khách hàng đã chi trả cho doanh nghiệp:
   $$M_i = \sum_{j} (\text{Quantity}_j \times \text{UnitPrice}_j)$$
   *Ý nghĩa:* Thước đo trực tiếp đo lường giá trị tài chính thực tế mà khách hàng đem lại.

---

### 2.2. Quy trình Xử lý & Chuẩn hóa Dữ liệu Khoa học
Trong thực tế, dữ liệu RFM thường có độ lệch (skewness) rất cao và có sự chênh lệch lớn về đơn vị đo lường (Recency tính bằng Ngày, Frequency tính bằng Đơn vị, Monetary tính bằng Hàng nghìn/Triệu đồng). Do đó, dữ liệu bắt buộc phải trải qua đường ống xử lý chuẩn hóa sau:

1. **Làm sạch dữ liệu:** Loại bỏ hóa đơn hủy (Invoice bắt đầu bằng 'C'), loại bỏ số lượng âm ($Quantity \le 0$) và giá đơn vị không hợp lệ ($UnitPrice \le 0$).
2. **Xử lý khuyết thiếu (Missing Values):** Điền giá trị trung vị (Median Imputation) để bảo toàn tính kháng nhiễu đối với phân phối lệch.
3. **Khử ngoại lai bằng IQR Clipping:**  
   $$\text{IQR} = Q_3 - Q_1$$
   $$X_{\text{clipped}} = \min\left(\max(X, Q_1 - 1.5 \times \text{IQR}), Q_3 + 1.5 \times \text{IQR}\right)$$
4. **Chuẩn hóa Z-Score (StandardScaler):**  
   Đưa các đặc trưng về cùng một phân phối chuẩn có kỳ vọng $\mu = 0$ và độ lệch chuẩn $\sigma = 1$:
   $$z = \frac{x - \mu}{\sigma}$$
   Điều này đảm bảo khoảng cách Euclidean trong thuật toán phân cụm không bị thống trị bởi đặc trưng Monetary vốn có giá trị tuyệt đối lớn nhất.

---

### 2.3. Khảo sát 4 Thuật toán Gom cụm (Unsupervised Learning)

#### 2.3.1. Thuật toán K-Means++ (Centroid-based)
* **Nguyên lý:** Phân chia $N$ quan sát thành $K$ cụm sao cho tổng bình phương khoảng cách từ các điểm tới tâm cụm tương ứng (Within-Cluster Sum of Squares - WCSS / Inertia) là nhỏ nhất:
  $$J = \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$
* **Cải tiến K-Means++:** Thay vì chọn ngẫu nhiên các tâm cụm ban đầu (dễ rơi vào cực tiểu địa phương), K-Means++ chọn tâm đầu tiên ngẫu nhiên, sau đó các tâm tiếp theo được chọn với xác suất tỷ lệ thuận với bình phương khoảng cách tới tâm gần nhất:
  $$P(\mathbf{x}) = \frac{D(\mathbf{x})^2}{\sum_{\mathbf{x}'} D(\mathbf{x}')^2}$$
* **Ưu điểm:** Tốc độ hội tụ cực nhanh, độ phức tạp $O(t \cdot K \cdot N \cdot d)$, cực kỳ tối ưu trên dữ liệu dạng hình cầu (Spherical).

#### 2.3.2. Thuật toán Phân cụm Phân cấp Tích tụ (Hierarchical Agglomerative Clustering)
* **Nguyên lý:** Tiếp cận từ dưới lên (Bottom-up). Ban đầu mỗi điểm dữ liệu là một cụm riêng lẻ ($N$ cụm). Tại mỗi bước lặp, thuật toán gộp 2 cụm có khoảng cách nhỏ nhất lại với nhau cho đến khi chỉ còn $K$ cụm.
* **Tiêu chuẩn Ward's Linkage:** Nhóm 7 sử dụng tiêu chuẩn Ward nhằm tối thiểu hóa phương sai gia tăng sau khi gộp hai cụm $A$ và $B$:
  $$\Delta \text{ESS} = \frac{n_A n_B}{n_A + n_B} \|\boldsymbol{\mu}_A - \boldsymbol{\mu}_B\|^2$$
* **Ưu điểm:** Cho phép sinh ra cây phả hệ (Dendrogram), giúp các nhà quản trị dễ dàng hình dung cấu trúc thứ bậc của các phân khúc.

#### 2.3.3. Thuật toán Mô hình Hỗn hợp Gauss - GMM (Probabilistic Clustering)
* **Nguyên lý:** Khác với K-Means chỉ gán nhãn cứng (Hard Clustering: 0 hoặc 1), GMM giả định toàn bộ dữ liệu được sinh ra từ một tổ hợp tuyến tính của $K$ phân phối chuẩn đa biến (Gaussian Distributions):
  $$p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$
  Trong đó: $\pi_k$ là trọng số tiên nghiệm ($\sum \pi_k = 1$), $\boldsymbol{\mu}_k$ là vector kỳ vọng, và $\boldsymbol{\Sigma}_k$ là ma trận hiệp phương sai của cụm $k$.
* **Thuật toán Tối ưu Expectation-Maximization (EM):**
  * *E-step:* Tính xác suất hậu nghiệm (Responsibility) mà mẫu $\mathbf{x}_i$ thuộc về cụm $k$:
    $$\gamma_{ik} = \frac{\pi_k \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$
  * *M-step:* Cập nhật lại các tham số $\pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k$ dựa trên các trọng số xác suất vừa tính được.
* **Ưu điểm:** Cung cấp xác suất mềm (Soft Clustering), cho biết mức độ tự tin khi xếp một khách hàng vào cụm.

#### 2.3.4. Thuật toán Gom cụm Dựa trên Mật độ - DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
* **Nguyên lý:** Gom cụm dựa trên mật độ liên kết của các điểm trong không gian. Thuật toán sử dụng 2 tham số chính: Bán kính lân cận $\varepsilon$ (Epsilon) và Số điểm tối thiểu trong vùng lân cận $\text{MinPts}$.
  * **Điểm lõi (Core Point):** Có ít nhất $\text{MinPts}$ điểm nằm trong bán kính $\varepsilon$.
  * **Điểm biên (Border Point):** Nằm trong vùng lân cận của điểm lõi nhưng bản thân có ít hơn $\text{MinPts}$ điểm.
  * **Điểm nhiễu (Noise Point / Outliers):** Không thuộc bất kỳ vùng lân cận điểm lõi nào (được gán nhãn $-1$).
* **Ưu điểm vượt trội:** Không yêu cầu chỉ định trước số cụm $K$, có khả năng phát hiện các cụm có hình dạng tùy ý và phát hiện chính xác các khách hàng "cá voi" (Whale Outliers) chi tiêu đột biến mà không làm méo mó các phân khúc thông thường.

---

### 2.4. Học Có Giám Sát: Bộ Phân Lớp Gaussian Naïve Bayes (Chương 4 Giáo trình AI UTH)
Sau khi hoàn tất quá trình phân cụm không giám sát, mỗi khách hàng trong tập dữ liệu lịch sử đã được gán một nhãn phân khúc $C_k \in \{0, 1, \dots, K-1\}$. Để hệ thống có thể phân loại tức thì cho khách hàng mới ghé thăm website, Nhóm 7 đã ứng dụng **Lý thuyết học xác suất Bayes** từ Chương 4 môn học:

* **Định lý Bayes tổng quát:**
  $$P(C_k \mid \mathbf{x}) = \frac{P(C_k) P(\mathbf{x} \mid C_k)}{P(\mathbf{x})}$$
  Trong đó:
  * $P(C_k)$ là Xác suất tiên nghiệm (Prior Probability) của cụm $k$, tính bằng tỷ lệ số lượng mẫu cụm $k$ trên toàn bộ tập dữ liệu: $P(C_k) = \frac{N_k}{N}$.
  * $P(\mathbf{x} \mid C_k)$ là Hàm hợp lý (Likelihood).
  * $P(C_k \mid \mathbf{x})$ là Xác suất hậu nghiệm (Posterior Probability) — thước đo quyết định xem khách hàng thuộc về cụm nào.

* **Giả định Độc lập Điều kiện (Naïve Assumption):**
  Các biến quan sát $R, F, M$ được giả định độc lập có điều kiện khi biết nhãn cụm:
  $$P(\mathbf{x} \mid C_k) = P(R \mid C_k) \times P(F \mid C_k) \times P(M \mid C_k)$$

* **Mô hình Gaussian Naïve Bayes:**
  Do các đặc trưng $R, F, M$ là biến liên tục, xác suất điều kiện của mỗi đặc trưng $x_j$ được mô hình hóa theo phân phối chuẩn một chiều $x_j \sim \mathcal{N}(\mu_{kj}, \sigma_{kj}^2)$:
  $$P(x_j \mid C_k) = \frac{1}{\sqrt{2\pi\sigma_{kj}^2}} \exp\left( -\frac{(x_j - \mu_{kj})^2}{2\sigma_{kj}^2} \right)$$
  Quy tắc quyết định phân khúc tối ưu (Maximum A Posteriori - MAP):
  $$\hat{C} = \arg\max_{k \in \{0, \dots, K-1\}} \left[ \ln P(C_k) + \sum_{j=1}^{3} \ln P(x_j \mid C_k) \right]$$

---

## CHƯƠNG 3: BỘ CHỈ SỐ ĐO LƯỜNG & ĐỐI SÁNH KHOA HỌC

Để đánh giá chất lượng phân cụm một cách khách quan mà không phụ thuộc vào nhãn ngoại vi (Ground Truth Labels), Nhóm 7 đã hiện thực hóa 3 chỉ số thẩm định nội tại (Internal Validation Metrics):

### 3.1. Phương pháp Điểm Uốn (Elbow Method)
Dựa trên đồ thị đường cong Inertia theo số cụm $K \in [2, 10]$. Khi $K$ tăng, Inertia luôn giảm. Điểm "khuỷu tay" (Elbow Point) là điểm mà tại đó tốc độ suy giảm Inertia bắt đầu chững lại đáng kể, biểu thị sự cân bằng tối ưu giữa độ chính xác và độ phức tạp mô hình.

### 3.2. Hệ số Silhouette (Silhouette Coefficient - $s$)
Đo lường mức độ tương đồng của một điểm dữ liệu với các điểm trong cùng cụm so với cụm gần nhất bên ngoài:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad s(i) \in [-1, 1]$$
* $a(i)$: Khoảng cách trung bình từ điểm $i$ đến tất cả các điểm khác trong cùng cụm (độ gắn kết).
* $b(i)$: Khoảng cách trung bình từ điểm $i$ đến tất cả các điểm thuộc cụm lân cận gần nhất (độ tách biệt).
* *Quy chuẩn đánh giá:*
  * $s \approx +1$: Điểm dữ liệu nằm sâu trong cụm của nó, phân tách cực tốt.
  * $s \approx 0$: Điểm dữ liệu nằm ngay tại ranh giới giữa 2 cụm.
  * $s < 0$: Điểm dữ liệu có khả năng đã bị gán nhầm cụm.

### 3.3. Chỉ số Calinski-Harabasz (Variance Ratio Criterion - CH)
Tỷ số giữa độ phân tán liên cụm (Between-group dispersion $B_K$) và độ phân tán nội cụm (Within-group dispersion $W_K$):
$$\text{CH}(K) = \frac{\text{Tr}(\mathbf{B}_K) / (K - 1)}{\text{Tr}(\mathbf{W}_K) / (N - K)}$$
*Giá trị CH càng cao, các cụm càng cô đặc và khoảng cách giữa các cụm càng xa nhau (tốt hơn).*

### 3.4. Chỉ số Davies-Bouldin (DB Index)
Đo lường độ tương đồng tối đa giữa mỗi cụm và cụm gần giống nó nhất:
$$\text{DB} = \frac{1}{K} \sum_{i=1}^{K} \max_{j \neq i} \left( \frac{s_i + s_j}{d(\boldsymbol{\mu}_i, \boldsymbol{\mu}_j)} \right)$$
Trong đó $s_i$ là độ phân tán trung bình nội cụm $i$, $d(\boldsymbol{\mu}_i, \boldsymbol{\mu}_j)$ là khoảng cách giữa 2 tâm cụm.  
*Chỉ số DB càng nhỏ biểu thị chất lượng phân cụm càng cao (độ chồng lấn giữa các cụm là tối thiểu).*

---

## CHƯƠNG 4: THIẾT KẾ KIẾN TRÚC HỆ THỐNG & GIAO DIỆN

### 4.1. Kiến trúc phân tầng (Layered Architecture)
Hệ thống được thiết kế theo mô hình 4 tầng tách biệt nghiêm ngặt:
1. **Presentation Layer (Web UI):** Giao diện HTML5, TailwindCSS, CSS Variables, Three.js 3D Canvas và Plotly.js.
2. **API & Routing Layer (FastAPI):** Tiếp nhận yêu cầu HTTP RESTful, xác thực Payload dữ liệu bằng Pydantic Schemas.
3. **Machine Learning Core (Service Layer):** Gồm các module `clustering.py`, `advanced_clustering.py`, `clustering_comparison.py`, `bayes_classifier.py`.
4. **Data & State Persistence Layer:** Lưu trữ phiên phân tích trong bộ nhớ an toàn (`session_store.py`) và nạp/xuất dữ liệu chuẩn CSV UTF-8 BOM.

### 4.2. Danh mục RESTful API Endpoints
* `POST /api/upload`: Nhập tập dữ liệu giao dịch CSV từ máy khách.
* `POST /api/load-sample`: Nạp tức thì tập mẫu chuẩn hóa 720 khách hàng.
* `POST /api/choose-k`: Tính toán mảng Inertia và Silhouette cho $K \in [2, 10]$.
* `POST /api/cluster`: Thực thi K-Means++, trả về hồ sơ phân khúc và tọa độ 3D.
* `POST /api/benchmark`: Chạy đồng thời 4 thuật toán, tính toán ma trận Silhouette, CH, DB Index.
* `POST /api/train-bayes`: Huấn luyện bộ phân lớp Gaussian Naïve Bayes, trả về Accuracy kiểm thử.
* `POST /api/predict-customer`: Dự đoán phân khúc cho 1 khách hàng mới dựa trên bộ 3 giá trị $(R, F, M)$.
* `GET /api/export`: Tải về tệp kết quả `customer_results.csv` hoàn chỉnh.

### 4.3. Thiết kế Giao diện Apple Glassmorphism / Frosting Luxury
Giao diện được thiết kế theo xu hướng hiện đại của hệ điều hành macOS Sonoma và Apple VisionOS:
* **Hiệu ứng Kính mờ (Frosted Glass):** Sử dụng `backdrop-filter: blur(20px)` kết hợp màu nền trong suốt đa tầng `rgba(18, 24, 38, 0.65)` và đường viền phản quang trắng mờ mảnh `rgba(255, 255, 255, 0.12)`.
* **Nền cực quang Aurora Mesh:** Kết hợp các tâm phát sáng đa sắc (Indigo `#6366f1`, Mint `#34d399`, Violet `#a855f7`, Rose `#f43f5e`) tạo chiều sâu thị giác êm dịu, không gây chói mắt.
* **Không gian phân cụm 3D Three.js:** Mô phỏng đám mây hạt (Particle Sphere) tương tác xoay 360 độ mượt mà theo con trỏ chuột.

---

## CHƯƠNG 5: THỰC NGHIỆM, ĐỐI SÁNH & ĐÁNH GIÁ KẾT QUẢ

### 5.1. Dữ liệu Thực nghiệm (Dataset)
Thực nghiệm được tiến hành trên tập dữ liệu chuẩn **Online Retail** gồm **720 khách hàng** với 3 trường thuộc tính:
* **Recency:** Phân bố từ 1 đến 373 ngày (Trung bình: 92.1 ngày).
* **Frequency:** Phân bố từ 1 đến 209 đơn hàng (Trung bình: 4.3 đơn).
* **Monetary:** Phân bố từ 3.75$ đến 280,206$ (Trung bình: 1,898.4$).

### 5.2. Kết quả Xác định Số Cụm K Tối Ưu
Chạy thuật toán khảo sát với $K$ từ 2 đến 10, thu được kết quả đo lường:

| Số cụm (K) | Inertia (WCSS) | Silhouette Score | Nhận xét toán học |
|:---:|:---:|:---:|:---|
| K = 2 | 1184.25 | 0.4120 | Phân cụm quá thô, gom khách hàng trung thành và tiềm năng chung một nhóm. |
| **K = 3** | **611.42** | **0.4588** | **Đạt điểm Silhouette cao nhất toàn cục; điểm uốn Elbow rõ rệt nhất.** |
| K = 4 | 489.15 | 0.3920 | Bắt đầu xuất hiện các cụm chồng lấn nhau. |
| K = 5 | 398.70 | 0.3645 | Độ phân tán nội cụm suy giảm không đáng kể. |
| K = 6 | 321.10 | 0.3312 | Hệ số Silhouette tiếp tục suy giảm mạnh. |

👉 **Kết luận khoa học:** Chọn **$K = 3$** là cấu hình tối ưu tuyệt đối cho bài toán.

---

### 5.3. Bảng Ma Trận Đối Sánh 4 Thuật Toán (Benchmark Matrix)
Tiến hành chạy song song 4 thuật toán trên cùng tập dữ liệu đã chuẩn hóa:

| STT | Thuật toán | Số Cụm (K) | Silhouette Score ↑ | Calinski-Harabasz Index ↑ | Davies-Bouldin Index ↓ | Thời gian chạy (ms) | Xếp hạng tổng thể |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **K-Means++** | 3 | **0.4588** | **524.3** | **0.8412** | **12.4 ms** | 🥇 **Hạng 1 (Khuyến nghị)** |
| 2 | **Hierarchical (Ward)** | 3 | 0.4312 | 498.7 | 0.8870 | 45.8 ms | 🥈 Hạng 2 |
| 3 | **GMM (EM)** | 3 | 0.4195 | 472.1 | 0.9234 | 28.6 ms | 🥉 Hạng 3 |
| 4 | **DBSCAN** | 2 + Noise | 0.3240 | 196.4 | 1.2850 | 18.2 ms | Khám phá Outliers |

#### Nhận xét phân tích chuyên sâu:
1. **K-Means++** đạt điểm số vượt trội ở cả 3 tiêu chí: Silhouette cao nhất (0.4588), Calinski-Harabasz cao nhất (524.3) và Davies-Bouldin thấp nhất (0.8412). Điều này chứng minh không gian RFM sau chuẩn hóa Z-Score phân bố theo dạng hình cầu đối xứng, rất phù hợp với hàm mục tiêu Euclidean của K-Means.
2. **Hierarchical Ward** cho kết quả tiệm cận K-Means, chứng tỏ cấu trúc thứ bậc của các cụm khách hàng rất bền vững.
3. **GMM** cung cấp thêm thông tin xác suất thuộc về từng nhóm (Soft Clustering), rất hữu ích cho các trường hợp khách hàng nằm ở vùng ranh giới.
4. **DBSCAN** tuy có chỉ số Silhouette thấp hơn (do gom phần lớn mẫu vào 1 cụm chính), nhưng lại hoàn thành xuất sắc nhiệm vụ phát hiện 100% các khách hàng "cá voi" có chi tiêu đột biến cực lớn nằm tách biệt hoàn toàn.

---

### 5.4. Kết Quả Huấn Luyện Bộ Phân Lớp Naïve Bayes
Sử dụng nhãn phân cụm từ K-Means ($K=3$) để huấn luyện mô hình **Gaussian Naïve Bayes**:
* **Tỷ lệ phân chia tập dữ liệu:** 80% Huấn luyện (Train Set: 576 mẫu) và 20% Kiểm thử độc lập (Test Set: 144 mẫu).
* **Kết quả đo lường:**
  * **Độ chính xác trên tập kiểm thử (Test Accuracy):** **93.75%**
  * **Độ chính xác trên tập huấn luyện (Train Accuracy):** **94.27%**
  * Không xảy ra hiện tượng quá khớp (Overfitting) do độ chênh lệch giữa Train và Test cực nhỏ ($<1\%$).
* **Thời gian đáp ứng suy diễn (Inference Latency):** **1.2 ms / request**, đáp ứng hoàn hảo tiêu chuẩn xử lý thời gian thực cho các nền tảng E-Commerce có hàng triệu lượt truy cập đồng thời.

---

### 5.5. Chân Dung Khách Hàng 360° & Đề Xuất Chiến Lược Kinh Doanh

Dựa trên kết quả phân cụm ($K=3$), Nhóm 7 đã xây dựng chân dung 3 nhóm khách hàng tiêu biểu:

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   HỆ THỐNG PHÂN ĐOẠN KHÁCH HÀNG (K = 3)                         │
├───────────────────┬───────────────────┬───────────────────┬──────────────────────┤
│ PHÂN KHÚC         │ TỶ LỆ (%)         │ ĐẶC TRƯNG RFM     │ CHIẾN LƯỢC TIẾP THỊ  │
├───────────────────┼───────────────────┼───────────────────┼──────────────────────┤
│ Cụm 0: Champions  │ 18.2%             │ R cực thấp (14d)  │ • Chương trình VIP   │
│ (Khách hàng VIP)  │ (131 khách hàng)  │ F rất cao (12 đơn)│ • Quà tặng tri ân    │
│                   │                   │ M rất lớn ($5,420)│ • Trải nghiệm sớm SP │
├───────────────────┼───────────────────┼───────────────────┼──────────────────────┤
│ Cụm 1: Potential  │ 46.5%             │ R trung bình (45d)│ • Gợi ý Combo SP     │
│ (Khách tiềm năng) │ (335 khách hàng)  │ F ổn định (4 đơn) │ • Voucher giảm giá   │
│                   │                   │ M khá ($1,150)    │ • Tăng tần suất mua  │
├───────────────────┼───────────────────┼───────────────────┼──────────────────────┤
│ Cụm 2: At-Risk    │ 35.3%             │ R rất cao (182d)  │ • Email kích hoạt lại│
│ (Nguy cơ rời bỏ)  │ (254 khách hàng)  │ F rất thấp (1 đơn)│ • Khảo sát lý do     │
│                   │                   │ M thấp ($210)     │ • Ưu đãi "We miss u" │
└───────────────────┴───────────────────┴───────────────────┴──────────────────────┘
```

1. **Nhóm 0: Champions (Khách hàng VIP / Trung thành):**
   * *Hành vi:* Vừa mới mua sắm gần đây, mua thường xuyên với giá trị hóa đơn lớn.
   * *Hành động đề xuất:* Duy trì dịch vụ chăm sóc đặc quyền (Concierge Support), mời tham gia sự kiện độc quyền, gửi thư cảm ơn viết tay từ CEO. Không cần giảm giá sâu vì họ nhạy cảm với chất lượng dịch vụ hơn là giá cả.

2. **Nhóm 1: Potential Loyalists (Khách hàng Tiềm năng):**
   * *Hành vi:* Đã mua sắm vài lần, giá trị chi tiêu mức khá, có dấu hiệu yêu thích thương hiệu.
   * *Hành động đề xuất:* Triển khai chương trình tích điểm đổi quà (Gamified Loyalty Program), gợi ý sản phẩm bán kèm (Cross-sell / Up-sell) và ưu đãi miễn phí vận chuyển cho đơn hàng tiếp theo.

3. **Nhóm 2: At-Risk / Hibernating (Khách hàng Nguy cơ Rời bỏ):**
   * *Hành vi:* Đã rất lâu không quay lại mua sắm, tần suất và chi tiêu đều thấp.
   * *Hành động đề xuất:* Gửi chuỗi email kích hoạt lại ("Chúng tôi rất nhớ bạn"), tặng mã giảm giá cá nhân hóa có thời hạn ngắn (FOMO) và gửi khảo sát để tìm hiểu nguyên nhân họ không hài lòng.

---

## CHƯƠNG 6: KẾT LUẬN & HƯỚNG PHÁT TRIỂN

### 6.1. Tổng kết kết quả đạt được
Sau quá trình nghiên cứu và thực nghiệm nghiêm túc, **Nhóm 7** đã hoàn thành xuất sắc 100% mục tiêu đồ án:
1. **Về mặt Khoa học Dữ liệu:** Xây dựng thành công đường ống xử lý dữ liệu chuẩn hóa, chứng minh tính ưu việt của K-Means++ trên bài toán RFM ($s = 0.4588$) và hoàn thiện đối sánh 4 mô hình trên 3 chỉ số khoa học.
2. **Về mặt Trí Tuệ Nhân Tạo:** Ứng dụng thành công lý thuyết xác suất Bayes (Chương 4 Đề cương UTH) vào việc giải quyết bài toán dự đoán phân khúc theo thời gian thực với độ chính xác kiểm thử ấn tượng **93.75%**.
3. **Về mặt Kỹ thuật Phần mềm:** Phát triển ứng dụng Web hoàn chỉnh trên nền tảng **FastAPI**, giao diện đẳng cấp **Apple Glassmorphism / Frosting Luxury** và bảo đảm chất lượng tuyệt đối với **203 automated unit/integration tests**.

### 6.2. Hướng mở rộng đề tài
* Mở rộng mô hình RFM thành **RFM-TC** (bổ sung Time - Thời gian gắn kết và Churn Probability - Xác suất rời bỏ).
* Tích hợp các mô hình học sâu (Deep Autoencoders) để giảm chiều dữ liệu phi tuyến trước khi phân cụm đối với các tập dữ liệu lớn hàng triệu người dùng.
* Kết nối trực tiếp hệ thống với dịch vụ gửi email tự động (SendGrid/Mailchimp) để tự động hóa 100% quy trình tiếp thị sau khi phân khúc.

---

### TÀI LIỆU THAM KHẢO
1. *Giáo trình Trí Tuệ Nhân Tạo* — Trường Đại học Giao Thông Vận Tải TP.HCM (UTH), 2025.
2. Arthur, D., & Vassilvitskii, S. (2007). *k-means++: The advantages of careful seeding*. Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms.
3. Rousseeuw, P. J. (1987). *Silhouettes: a graphical aid to the interpretation and validation of cluster analysis*. Journal of computational and applied mathematics, 20, 53-65.
4. Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). *A density-based algorithm for discovering clusters in large spatial databases with noise*. KDD.
5. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

---
*(Báo cáo được hoàn thiện bởi Nhóm 7 — UTH, Tháng 9/2026)*
