# TÀI LIỆU ÔN TẬP & BẢO VỆ ĐỒ ÁN TRÍ TUỆ NHÂN TẠO
## Đề tài: CustomerInsight AI — Hệ thống Phân tích & Suy diễn Hành vi Khách hàng Đa thuật toán
**Đơn vị:** Trường Đại học Giao thông Vận tải TP.HCM (UTH) — Nhóm 7

---

## MỤC LỤC
1. [Bản chất bài toán & Mô hình dữ liệu RFM](#1-bản-chất-bài-toán--mô-hình-dữ-liệu-rfm)
2. [Quy trình Tiền xử lý dữ liệu (Data Preprocessing)](#2-quy-trình-tiền-xử-lý-dữ-liệu-data-preprocessing)
3. [Các thuật toán Học máy cốt lõi (Core AI Algorithms)](#3-các-thuật-toán-học-máy-cốt-lõi-core-ai-algorithms)
   - 3.1. K-Means++ & Tối ưu số cụm K (Elbow, Silhouette)
   - 3.2. Đối sánh 4 thuật toán phân cụm (K-Means, Ward, GMM, DBSCAN)
   - 3.3. Học có giám sát: Gaussian Naïve Bayes
4. [Ánh xạ hành vi & Chân dung 5 phân khúc khách hàng](#4-ánh-xạ-hành-vi--chân-dung-5-phân-khúc-khách-hàng)
5. [Cấu trúc mã nguồn & Luồng thực thi (Architecture & Code)](#5-cấu-trúc-mã-nguồn--luồng-thực-thi-architecture--code)
6. [Bộ câu hỏi vấn đáp trọng tâm khi chấm đồ án (Q&A Cheat Sheet)](#6-bộ-câu-hỏi-vấn-đáp-trọng-tâm-khi-chấm-đồ-án-qa-cheat-sheet)

---

## 1. BẢN CHẤT BÀI TOÁN & MÔ HÌNH DỮ LIỆU RFM

### 1.1. Mục tiêu kinh doanh
Doanh nghiệp bán lẻ trực tuyến có hàng ngàn khách hàng nhưng ngân sách tiếp thị (Marketing/Voucher/Chăm sóc) có hạn. Nếu đối xử với mọi khách hàng như nhau sẽ gây lãng phí chi phí. 
-> **Mục tiêu của AI**: Tự động nhóm khách hàng có hành vi tương đồng vào cùng một phân khúc, từ đó cá nhân hóa chiến lược chăm sóc và phân loại ngay lập tức khi có khách hàng mới.

### 1.2. Mô hình định lượng RFM là gì?
Hệ thống sử dụng bộ 3 chỉ số kinh điển trong thương mại điện tử:
1. **Recency ($R$) — Tính gần đây**: 
   - Số ngày kể từ lần giao dịch gần nhất của khách hàng đến mốc thời gian khảo sát.
   - *Quy luật*: $R$ càng nhỏ -> Khách hàng vừa mới mua gần đây -> Đang hoạt động tích cực. $R$ càng lớn -> Khách lâu ngày không quay lại -> Có nguy cơ rời bỏ (Churn).
2. **Frequency ($F$) — Tần suất**:
   - Tổng số đơn hàng hoặc số lần phát sinh giao dịch của khách hàng.
   - *Quy luật*: $F$ càng cao -> Khách hàng càng quen thuộc và gắn bó với doanh nghiệp.
3. **Monetary ($M$) — Doanh số / Tiền tệ**:
   - Tổng số tiền khách hàng đã chi tiêu tích lũy.
   - *Quy luật*: $M$ càng cao -> Khách hàng mang lại giá trị tài chính lớn cho doanh nghiệp.

---

## 2. QUY TRÌNH TIỀN XỬ LÝ DỮ LIỆU (DATA PREPROCESSING)
File phụ trách: `src/preprocessing.py`

Dữ liệu thô thực tế không bao giờ đưa thẳng vào thuật toán AI được. Dự án thực hiện quy trình 4 bước chuẩn khoa học:

### Bước 1: Làm sạch dữ liệu (Data Cleaning)
- Loại bỏ các dòng có mã khách hàng trống (`CustomerID is Null`).
- Loại bỏ các đơn hàng hoàn/hủy có số lượng hoặc đơn giá âm (`Quantity <= 0`, `UnitPrice <= 0`).

### Bước 2: Khử ngoại lai bằng IQR (Interquartile Range)
- *Tại sao phải làm?* Trong bán lẻ, có những khách mua sỉ (mua hàng ngàn sản phẩm, chi hàng trăm triệu) hoặc có khách bị lỗi giao dịch. Những điểm dị biệt (Outliers) này nếu giữ lại sẽ kéo lệch tâm cụm K-Means, làm hỏng toàn bộ kết quả phân nhóm.
- *Công thức toán*:
  $$IQR = Q_3 - Q_1$$
  $$\text{Ngưỡng dưới} = Q_1 - 1.5 \times IQR, \quad \text{Ngưỡng trên} = Q_3 + 1.5 \times IQR$$
  *(Dữ liệu nằm ngoài khoảng này được xem là ngoại lai và bị loại bỏ).*

### Bước 3: Khắc phục lệch phân phối bằng Log-Transform
- *Tại sao phải làm?* Dữ liệu tiền bạc ($M$) và tần suất ($F$) luôn bị hiện tượng **Lệch phải nặng (Right-skewed)**: đa số khách hàng mua ít tiền, chỉ một lượng nhỏ khách chi nhiều tiền. Nếu dùng trực tiếp, khoảng cách Euclid trong K-Means sẽ bị méo mó.
- *Giải pháp*: Biến đổi theo hàm logarit tự nhiên:
  $$x_{\text{log}} = \ln(1 + x)$$
  *(Cộng 1 để phòng trường hợp $x = 0$, đảm bảo $\ln(1) = 0$, không bị lỗi chia cho 0 hoặc log số âm).* Biến đổi này ép đồ thị co về dạng phân phối gần chuẩn (hình chuông Gauss).

### Bước 4: Chuẩn hóa MinMax Scaler
- *Tại sao phải làm?* $R$ tính bằng ngày (1 - 365), $F$ tính bằng lần (1 - 50), còn $M$ tính bằng triệu đồng (100.000 - 100.000.000). Nếu để nguyên, biến $M$ sẽ áp đảo hoàn toàn $R$ và $F$ khi tính khoảng cách.
- *Công thức toán*:
  $$x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \in [0, 1]$$
  Đưa cả 3 chiều về cùng hệ quy chiếu công bằng từ 0 đến 1.

---

## 3. CÁC THUẬT TOÁN HỌC MÁY CỐT LÕI (CORE AI ALGORITHMS)

### 3.1. K-Means++ & Tối ưu hóa số cụm K
File phụ trách: `src/clustering.py`

#### Bản chất thuật toán K-Means++
1. **Khởi tạo thông minh (K-Means++)**: Thay vì chọn ngẫu nhiên các tâm cụm (dễ rơi vào cực tiểu địa phương), K-Means++ chọn tâm đầu tiên ngẫu nhiên, sau đó các tâm tiếp theo được chọn với xác suất tỷ lệ thuận với khoảng cách bình phương tới tâm gần nhất. Điều này giúp các tâm ban đầu nằm cách xa nhau.
2. **Gán cụm**: Từng điểm dữ liệu $x_i$ được gán vào cụm $k$ có khoảng cách Euclid ngắn nhất tới tâm $\mu_k$:
   $$c^{(i)} = \arg\min_k \|x_i - \mu_k\|^2$$
3. **Cập nhật tâm cụm**: Tính lại tọa độ tâm cụm bằng trung bình cộng tất cả các điểm thuộc cụm đó:
   $$\mu_k = \frac{1}{|S_k|} \sum_{x \in S_k} x$$
4. **Điều kiện dừng**: Lặp lại bước 2 và 3 cho đến khi tọa độ các tâm không còn thay đổi hoặc đạt số vòng lặp tối đa (`max_iter = 300`).

#### Hai phương pháp khoa học để chọn K
Hệ thống chạy thử nghiệm $K$ từ 2 đến 10:
- **Phương pháp Khuỷu tay (Elbow Method)**:
  - Tính tổng bình phương khoảng cách trong cụm (Inertia / WCSS):
    $$\text{WCSS} = \sum_{k=1}^K \sum_{x \in S_k} \|x - \mu_k\|^2$$
  - Khi $K$ tăng, WCSS luôn giảm. Điểm gập khúc (Elbow) mà tại đó tốc độ giảm chậm lại rõ rệt chính là điểm $K$ cân bằng giữa độ nén và độ phức tạp.
- **Hệ số Silhouette Score**:
  - Đo lường mức độ hợp lý của việc gán điểm dữ liệu vào cụm của nó:
    $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
    *(Trong đó $a(i)$ là khoảng cách trung bình đến các điểm cùng cụm; $b(i)$ là khoảng cách trung bình đến các điểm ở cụm gần nhất khác).*
  - $s(i) \in [-1, 1]$. Điểm càng gần +1 nghĩa là phân cụm càng rõ nét và các cụm tách bạch nhau.
  - **Kết luận thực nghiệm**: Tại $K = 5$, hệ số Silhouette đạt đỉnh cao và WCSS đạt độ gấp khúc tự nhiên.

---

### 3.2. Đối sánh 4 thuật toán phân cụm (Clustering Benchmarking)
File phụ trách: `src/clustering_comparison.py`

| Thuật toán | Cơ chế hoạt động | Ưu điểm | Nhược điểm |
|---|---|---|---|
| **K-Means** | Phân hoạch dựa trên tâm (Centroid-based) | Chạy cực nhanh, dễ diễn giải kinh doanh. | Chỉ tìm cụm dạng hình cầu lồi. |
| **Hierarchical (Ward)** | Phân cụm thứ bậc kết tụ (Agglomerative), tối thiểu hóa phương sai khi gộp cụm. | Kết quả ổn định 100%, không phụ thuộc tâm ban đầu. | Độ phức tạp tính toán lớn ($O(n^3)$), tốn RAM. |
| **GMM (Gaussian Mixture)** | Phân cụm xác suất mềm (Soft Clustering) dùng thuật toán EM. | Mỗi khách có thể thuộc về nhiều cụm theo % xác suất. | Nhạy cảm với khởi tạo ban đầu. |
| **DBSCAN** | Phân cụm dựa trên mật độ (Density-based) với $\epsilon$ và $\text{MinPts}$. | Tự động phát hiện ngoại lai, tìm hình dạng bất kỳ. | Khó chọn $\epsilon$; kém hiệu quả khi mật độ không đều. |

#### 3 Thước đo khoa học để so sánh:
1. **Silhouette Score**: Đo độ tách biệt giữa các cụm (càng **cao** càng tốt).
2. **Calinski-Harabasz (CH Index)**: Tỷ số phương sai liên cụm và nội cụm (càng **cao** càng tốt).
3. **Davies-Bouldin (DB Index)**: Tỷ lệ độ phân tán nội cụm so với khoảng cách giữa các tâm (càng **thấp** càng tốt).

---

### 3.3. Học có giám sát: Phân loại bằng Gaussian Naïve Bayes
File phụ trách: `src/bayes_classifier.py`

#### Tại sao cần Naïve Bayes sau khi đã có K-Means?
- K-Means là học không giám sát: dùng để **tìm ra quy luật và dán nhãn** ban đầu cho dữ liệu cũ.
- Khi có khách hàng mới, ta cần **xác định ngay họ thuộc cụm nào trong mili-giây** để gửi voucher phù hợp mà không cần chạy lại toàn bộ thuật toán K-Means cho toàn bộ cơ sở dữ liệu.

#### Công thức toán học
Định lý Bayes:
$$P(C_k \mid X) = \frac{P(C_k) \cdot P(X \mid C_k)}{P(X)} \propto P(C_k) \prod_{j=1}^d P(x_j \mid C_k)$$

- $P(C_k)$: Xác suất tiên nghiệm (tỷ lệ số khách hàng thuộc cụm $k$ trong lịch sử).
- $P(x_j \mid C_k)$: Hàm mật độ xác suất của đặc trưng $x_j$ (với $j \in \{R, F, M\}$) theo phân phối chuẩn Gauss:
  $$P(x_j \mid C_k) = \frac{1}{\sqrt{2\pi \sigma_{kj}^2}} \exp\left(-\frac{(x_j - \mu_{kj})^2}{2\sigma_{kj}^2}\right)$$

#### Đánh giá mô hình & Chống Overfitting:
- Chia tập dữ liệu: **80% Train, 20% Test** (`test_size=0.2, random_state=42`).
- **Train Accuracy**: ~94.4% | **Test Accuracy**: ~93.1%.
- Mức chênh lệch $| \text{Train} - \text{Test} | \approx 1.3\% < 5\%$ chứng minh mô hình tổng quát hóa tuyệt vời, **hoàn toàn không bị Overfitting**.

---

## 4. ÁNH XẠ HÀNH VI & CHÂN DUNG 5 PHÂN KHÚC KHÁCH HÀNG
File phụ trách: `src/profiling.py`

| Mã cụm | Tên phân khúc | Đặc điểm RFM thực tế | Chiến lược Marketing / Hành động |
|---|---|---|---|
| **Cụm 0** | **Khách hàng VIP / Kim cương** | $R$ rất thấp (vừa mua), $F$ cực cao, $M$ cực lớn. | Tri ân đặc biệt, phục vụ ưu tiên 1:1, tặng quà sinh nhật cao cấp. |
| **Cụm 1** | **Khách hàng Thân thiết (Loyal)** | $R$ thấp, $F$ đều đặn, $M$ ở mức khá/tốt. | Tích điểm thăng hạng, khuyến khích giới thiệu bạn bè (Referral). |
| **Cụm 2** | **Khách hàng Tiềm năng (Potential)** | Mới mua gần đây ($R$ thấp), nhưng tần suất $F$ và tiền $M$ chưa cao. | Gửi email hướng dẫn sử dụng, tặng voucher cho đơn hàng thứ 2. |
| **Cụm 3** | **Khách hàng Nguy cơ rời bỏ (At Risk)** | Đã từng mua nhiều ($F, M$ khá) nhưng rất lâu chưa quay lại ($R$ lớn). | Chiến dịch "Chúng tôi nhớ bạn" (Win-back), gọi điện khảo sát, tặng voucher giảm sốc. |
| **Cụm 4** | **Khách hàng Ngủ đông / Mất mát (Lost)** | $R$ rất cao (quá lâu không mua), $F = 1$, $M$ rất thấp. | Giảm chi phí tiếp thị trực tiếp; chỉ gửi email tự động vào các dịp lễ lớn. |

---

## 5. BỘ CÂU HỎI VẤN ĐÁP TRỌNG TÂM KHI CHẤM ĐỒ ÁN (Q&A CHEAT SHEET)

### Câu 1: Tại sao đồ án của em phải kết hợp cả Học không giám sát (Clustering) và Học có giám sát (Naïve Bayes)?
> **Trả lời**: "Thưa thầy/cô, dữ liệu khách hàng ban đầu chưa có bất kỳ nhãn nào (Unlabeled). Em dùng **K-Means** (học không giám sát) để gom các khách hàng tương đồng vào cụm tự nhiên và dán nhãn kinh doanh. Sau khi dữ liệu đã có nhãn chuẩn, em huấn luyện **Gaussian Naïve Bayes** (học có giám sát) để khi có một khách hàng mới xuất hiện, hệ thống có thể tính toán xác suất và phân loại họ ngay lập tức trong vòng chưa tới 2ms mà không cần gom cụm lại từ đầu."

### Câu 2: Tại sao em lại chọn $K = 5$? Nếu em chọn $K = 3$ hoặc $K = 8$ thì sao?
> **Trả lời**: "Dạ thưa thầy/cô, việc chọn $K=5$ dựa trên cả 2 yếu tố:
> 1. **Về mặt kỹ thuật**: Tại $K=5$, biểu đồ Elbow xuất hiện điểm gấp khúc giảm quán tính rõ rệt, và điểm Silhouette Score đạt mức cao ổn định (~0.42). Nếu chọn $K=3$ thì quá khái quát, không tách được khách tiềm năng và khách ngủ đông; nếu chọn $K=8$ thì cụm bị phân mảnh, nhiều cụm bị trùng lặp ý nghĩa và WCSS không giảm thêm đáng kể.
> 2. **Về mặt kinh doanh**: $K=5$ tương ứng hoàn hảo với 5 phân khúc kinh điển trong thực tế bán lẻ: VIP, Thân thiết, Tiềm năng, Nguy cơ rời bỏ và Ngủ đông."

### Câu 3: Tại sao dữ liệu phải đi qua Log-Transform $\ln(1+x)$ trước khi phân cụm?
> **Trả lời**: "Vì trong thương mại điện tử, số tiền chi tiêu ($M$) và số lần mua ($F$) luôn bị lệch phải rất mạnh (phần lớn khách mua ít tiền, một số ít khách chi rất nhiều). Nếu tính khoảng cách Euclid trực tiếp, các điểm giá trị lớn sẽ làm biến dạng tâm cụm. Phép biến đổi Log đưa phân phối dữ liệu về gần phân phối chuẩn Gauss, giúp thuật toán K-Means hoạt động ổn định và chính xác nhất."

### Câu 4: Trong 4 thuật toán ở trang Đối sánh (Benchmark), thuật toán nào tốt nhất và tại sao?
> **Trả lời**: "Thưa thầy/cô, **K-Means** và **Hierarchical Ward** cho kết quả tốt nhất về độ tách biệt và chỉ số Silhouette Score (~0.42). Tuy nhiên, K-Means vượt trội hơn hẳn về thời gian thực thi (Runtime chỉ ~5ms so với ~80ms của Hierarchical). GMM phù hợp cho phân cụm mềm nhưng ranh giới cụm kém rõ ràng hơn. DBSCAN không tối ưu cho tập dữ liệu RFM này vì mật độ phân bố giữa khách VIP và khách thường chênh lệch nhau rất lớn."

### Câu 5: Làm sao em chứng minh mô hình Naïve Bayes của em không bị học vẹt (Overfitting)?
> **Trả lời**: "Em đã chia tập dữ liệu thành 80% Train và 20% Test độc lập. Kết quả đánh giá cho thấy độ chính xác trên tập Train đạt **~94.4%** và trên tập Test đạt **~93.1%**. Độ chênh lệch giữa Train và Test chỉ khoảng **1.3%** (< 5%), điều này chứng minh mô hình có khả năng tổng quát hóa cực kỳ tốt trên dữ liệu mới và không bị Overfitting."
