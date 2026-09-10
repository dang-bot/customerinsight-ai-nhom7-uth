# MASTER PROMPT & TÀI LIỆU NGỮ CẢNH (CONTEXT PACKAGE)
## DÀNH ĐỂ NẠP CHO AI (CHATGPT / CLAUDE / GEMINI) SOẠN BÁO CÁO TOÀN DIỆN

Bạn là một chuyên gia cao cấp về Trí Tuệ Nhân Tạo (AI) và Khoa học Dữ liệu (Data Science), đồng thời là giảng viên học thuật có nhiều năm kinh nghiệm hướng dẫn đồ án tốt nghiệp đại học tại Việt Nam.

Nhiệm vụ của bạn là: Dựa vào toàn bộ thông tin dự án, cấu trúc kỹ thuật, công thức toán học và số liệu thực nghiệm chi tiết dưới đây, hãy soạn thảo một BẢN BÁO CÁO BÀI TẬP LỚN MÔN HỌC HOÀN CHỈNH, CHUYÊN NGHIỆP, HỌC THUẬT VÀ DÀI TẬP (tương đương 30 - 40 trang luận văn chuẩn quy cách đại học).

================================================================================
1. THÔNG TIN HÀNH CHÍNH & THÀNH VIÊN ĐỀ TÀI
================================================================================
- Tên trường: TRƯỜNG ĐẠI HỌC GIAO THÔNG VẬN TẢI TP.HCM (UTH)
- Viện: Viện Công nghệ Thông tin
- Bộ môn: Trí tuệ Nhân tạo (Mã học phần: 121033)
- Học kỳ: HK Hè — Năm học 2025 - 2026
- Tên đề tài: "HỆ THỐNG PHÂN KHÚC & DỰ ĐOÁN HÀNH VI KHÁCH HÀNG THÔNG MINH (CUSTOMERINSIGHT AI: MULTI-ALGORITHM CLUSTERING & REAL-TIME GAUSSIAN NAÏVE BAYES INFERENCE)"
- Repository GitHub chính thức của nhóm: https://github.com/dang-bot/customerinsight-ai-nhom7-uth.git
- Đơn vị thực hiện: NHÓM 7 (Gồm 5 thành viên):
  1. BÙI ĐĂNG TOÀN TỈNH — Trưởng nhóm kỹ thuật ML
     Nhiệm vụ: Nghiên cứu toán học & tích hợp 3 thuật toán gom cụm nâng cao: Hierarchical (Ward's Linkage), Gaussian Mixture Models (GMM/EM) và DBSCAN xử lý ngoại lai (Outliers).
  2. ĐẶNG HOÀNG TRUNG — Kỹ sư Mô hình Học máy
     Nhiệm vụ: Ứng dụng lý thuyết xác suất Bayes (Chương 4 Đề cương AI UTH), xây dựng bộ phân lớp Gaussian Naïve Bayes dự đoán phân khúc tức thì cho khách hàng mới (<2ms).
  3. ĐẶNG ĐÌNH KHANG — Kỹ sư Fullstack & UI/UX
     Nhiệm vụ: Thiết kế toàn bộ kiến trúc FastAPI RESTful Endpoints, đồng bộ State dữ liệu và phát triển giao diện Web Dashboard theo phong cách Apple Glassmorphism / Frosting Luxury.
  4. NGUYỄN QUANG HIẾU — Kỹ sư Báo cáo Học thuật
     Nhiệm vụ: Biên tập toàn diện báo cáo kỹ thuật khoa học, kịch bản bảo vệ trước hội đồng, phân tích nghiệp vụ thương mại điện tử và các chiến lược tiếp thị đa chiều.
  5. NGUYỄN THÁI HỌC — Kỹ sư Đánh giá & QA
     Nhiệm vụ: Xây dựng pipeline đo lường đối sánh 3 chỉ số khoa học (Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Index) và quản trị bộ kiểm thử tự động 203 Test Cases (pytest).

================================================================================
2. TỔNG QUAN BÀI TOÁN & TÍNH ĐỘC ĐÁO CỦA DỰ ÁN
================================================================================
- Bối cảnh: Doanh nghiệp bán lẻ hiện đại đối mặt với sự phân hóa doanh thu sâu sắc (Quy luật Pareto: 80% doanh thu đến từ 20% khách hàng trung thành). Phương pháp marketing đại trà (Mass Marketing) gây lãng phí lớn, còn phương pháp gom cụm thủ công trên Excel không thể mở rộng và mang tính cảm tính.
- Đóng góp độc quyền của Nhóm 7 (vượt trội so với các đề tài K-Means thông thường):
  1. Động cơ gom cụm đa mô hình (Multi-algorithm Engine): Không chỉ dùng K-Means đơn lẻ, nhóm xây dựng "Đấu trường Đối sánh" (Arena Benchmark) chạy đồng thời 4 thuật toán:
     • K-Means++ (Centroid-based)
     • Hierarchical Agglomerative Clustering (Connectivity-based với tiêu chuẩn Ward's Linkage)
     • Gaussian Mixture Models - GMM (Probabilistic-based với giải thuật EM)
     • DBSCAN (Density-based gom cụm theo mật độ và phát hiện điểm nhiễu)
  2. Đánh giá đa chiều trên 3 thước đo nội tại khoa học:
     • Silhouette Score (Độ tách biệt và gắn kết nội cụm)
     • Calinski-Harabasz Index (Tỷ số phương sai liên cụm / nội cụm)
     • Davies-Bouldin Index (Độ tương đồng và chồng lấn giữa các cụm)
  3. Ứng dụng Học có giám sát (Gaussian Naïve Bayes - Chương 4 Giáo trình AI UTH):
     Sau khi gom cụm không giám sát để gắn nhãn chân dung khách hàng lịch sử, mô hình Gaussian Naïve Bayes được huấn luyện để đóng vai trò "Trình suy diễn thời gian thực" (Real-time Inference Engine). Khi có một khách hàng mới nhập vào các chỉ số (Recency, Frequency, Monetary), hệ thống tính toán xác suất hậu nghiệm P(Cluster_k | R,F,M) và đưa ra phân khúc cùng độ tin cậy tức thì trong <2ms mà không phải chạy lại K-Means toàn cục.
  4. Giao diện Người dùng Apple Glassmorphism / Frosting Luxury:
     Thiết kế đẳng cấp với nền Aurora Mesh Gradient, các panel kính mờ trong suốt (backdrop-filter: blur(20px)) và không gian 3D Three.js Particle Cloud tương tác xoay 360 độ theo chuyển động chuột.
  5. Chất lượng phần mềm chuẩn công nghiệp:
     Được bảo đảm bởi bộ kiểm thử tự động 203 Automated Unit & Integration Tests đạt tỷ lệ Passed 100%.

================================================================================
3. CƠ SỞ TOÁN HỌC & CÔNG THỨC CHI TIẾT
================================================================================
Hãy trình bày đầy đủ, chi tiết từng công thức toán học dưới dạng LaTeX:

A. MÔ HÌNH RFM:
- Vector đặc trưng: x = [R, F, M]^T
- Recency: R_i = max(T) - T_i_last (Khoảng cách ngày từ lần mua gần nhất)
- Frequency: F_i = Tổng số hóa đơn phát sinh thành công
- Monetary: M_i = Tổng doanh thu chi tiêu tích luỹ

B. TIỀN XỬ LÝ & CHUẨN HÓA DỮ LIỆU:
- Làm sạch hóa đơn hủy (Invoice bắt đầu bằng 'C'), loại bỏ Quantity <= 0 và UnitPrice <= 0.
- Xử lý khuyết thiếu: Median Imputation.
- Khử ngoại lai bằng IQR Clipping:
  IQR = Q3 - Q1
  X_clipped = min(max(X, Q1 - 1.5 * IQR), Q3 + 1.5 * IQR)
- Chuẩn hóa Z-Score (StandardScaler):
  z = (x - mu) / sigma
  Đưa về phân phối chuẩn có kỳ vọng bằng 0, độ lệch chuẩn bằng 1 để khoảng cách Euclidean không bị chi phối bởi Monetary.

C. 4 THUẬT TOÁN PHÂN CỤM:
1. K-Means++:
   - Hàm mục tiêu cực tiểu WCSS (Inertia):
     J = sum_{k=1}^K sum_{x_i in C_k} ||x_i - mu_k||^2
   - Xác suất chọn tâm ban đầu K-Means++:
     P(x) = D(x)^2 / sum_{x'} D(x')^2
2. Hierarchical (Ward's Linkage):
   - Phương sai gia tăng tối thiểu khi gộp 2 cụm A và B:
     Delta ESS = (n_A * n_B) / (n_A + n_B) * ||mu_A - mu_B||^2
3. Gaussian Mixture Models (GMM):
   - Phân phối kết hợp: p(x) = sum_{k=1}^K pi_k * N(x | mu_k, Sigma_k)
   - E-step (Tính trách nhiệm hậu nghiệm):
     gamma_{ik} = [pi_k * N(x_i | mu_k, Sigma_k)] / [sum_{j=1}^K pi_j * N(x_i | mu_j, Sigma_j)]
   - M-step (Cập nhật tham số trọng số pi, kỳ vọng mu, hiệp phương sai Sigma).
4. DBSCAN:
   - Tham số: Bán kính epsilon và MinPts.
   - Định nghĩa điểm lõi (Core point), điểm biên (Border point) và điểm nhiễu ngoại lai (Noise point, label = -1).

D. PHÂN LỚP GAUSSIAN NAÏVE BAYES (CHƯƠNG 4 GIÁO TRÌNH AI):
- Định lý Bayes:
  P(C_k | x) = [P(C_k) * P(x | C_k)] / P(x)
- Giả định độc lập điều kiện:
  P(x | C_k) = P(R | C_k) * P(F | C_k) * P(M | C_k)
- Hàm mật độ xác suất Gauss liên tục:
  P(x_j | C_k) = (1 / sqrt(2 * pi * sigma_{kj}^2)) * exp(-(x_j - mu_{kj})^2 / (2 * sigma_{kj}^2))
- Quy tắc quyết định phân khúc tối ưu (MAP):
  C_hat = argmax_{k} [ ln P(C_k) + sum_{j=1}^3 ln P(x_j | C_k) ]

E. BỘ 3 CHỈ SỐ ĐO LƯỜNG NỘI TẠI:
1. Silhouette Coefficient (s):
   s(i) = [b(i) - a(i)] / max(a(i), b(i)), s in [-1, 1]
   (a(i): khoảng cách trung bình nội cụm; b(i): khoảng cách trung bình cụm lân cận gần nhất).
2. Calinski-Harabasz Index (CH):
   CH = [Tr(B_K) / (K - 1)] / [Tr(W_K) / (N - K)] (Càng lớn càng tốt).
3. Davies-Bouldin Index (DB):
   DB = (1 / K) * sum_{i=1}^K max_{j != i} [(s_i + s_j) / d(mu_i, mu_j)] (Càng bé càng tốt).

================================================================================
4. DỮ LIỆU THỰC NGHIỆM & KẾT QUẢ ĐO LƯỜNG CHI TIẾT
================================================================================
- Tập dữ liệu: Online Retail chuẩn hóa (N = 720 khách hàng tiêu biểu).
- Kết quả khảo sát chọn K tối ưu (K từ 2 đến 10):
  • K = 2: Inertia = 1184.25, Silhouette = 0.4120
  • K = 3: Inertia = 611.42, Silhouette = 0.4588 (ĐẠT ĐỈNH TOÀN CỤC & ĐIỂM UỐN ELBOW RÕ RỆT NHẤT)
  • K = 4: Inertia = 489.15, Silhouette = 0.3920
  • K = 5: Inertia = 398.70, Silhouette = 0.3645
  => K = 3 là cấu hình tối ưu tuyệt đối.

- Bảng kết quả Ma trận đối sánh 4 thuật toán (Benchmark):
  1. K-Means++: K=3 | Silhouette: 0.4588 (Cao nhất) | CH Index: 524.3 (Cao nhất) | DB Index: 0.8412 (Thấp nhất) | Thời gian: 12.4ms => Xếp hạng 1 (Khuyến nghị triển khai).
  2. Hierarchical (Ward): K=3 | Silhouette: 0.4312 | CH Index: 498.7 | DB Index: 0.8870 | Thời gian: 45.8ms => Xếp hạng 2.
  3. GMM (EM): K=3 | Silhouette: 0.4195 | CH Index: 472.1 | DB Index: 0.9234 | Thời gian: 28.6ms => Xếp hạng 3.
  4. DBSCAN: 2 cụm + 1 cụm Nhiễu (-1) | Silhouette: 0.3240 | CH Index: 196.4 | DB Index: 1.2850 | Thời gian: 18.2ms => Phát hiện 100% khách hàng chi tiêu đột biến (Whales).

- Kết quả kiểm định mô hình Gaussian Naïve Bayes:
  • Tỷ lệ phân chia: Train 80% (576 mẫu), Test 20% (144 mẫu).
  • Độ chính xác trên tập kiểm thử (Test Accuracy): 93.75%
  • Độ chính xác trên tập huấn luyện (Train Accuracy): 94.27% (Không bị Overfitting).
  • Thời gian phản hồi suy diễn (Inference Latency): 1.2ms / khách hàng mới.

- Chân dung 3 cụm khách hàng (Segment 360°) & Chiến lược kinh doanh:
  1. Cụm 0: Champions (Khách hàng VIP) - 18.2% (131 KH):
     • Đặc trưng: Recency cực thấp (14 ngày), Frequency rất cao (12 đơn), Monetary rất lớn ($5,420).
     • Chiến lược: Dịch vụ chăm sóc riêng biệt (Concierge), tri ân đặc quyền, trải nghiệm trước sản phẩm mới.
  2. Cụm 1: Potential Loyalists (Khách hàng tiềm năng) - 46.5% (335 KH):
     • Đặc trưng: Recency trung bình (45 ngày), Frequency ổn định (4 đơn), Monetary khá ($1,150).
     • Chiến lược: Khuyến khích tích điểm thưởng (Loyalty Program), đề xuất sản phẩm bán kèm (Cross-sell/Up-sell), voucher miễn phí vận chuyển.
  3. Cụm 2: At-Risk / Hibernating (Khách hàng nguy cơ rời bỏ) - 35.3% (254 KH):
     • Đặc trưng: Recency rất cao (182 ngày không quay lại), Frequency thấp (1 đơn), Monetary thấp ($210).
     • Chiến lược: Gửi chuỗi email chăm sóc kích hoạt lại ("Chúng tôi nhớ bạn"), mã giảm giá cá nhân hóa có thời hạn ngắn, khảo sát phản hồi dịch vụ.

================================================================================
5. YÊU CẦU CẤU TRÚC BÁO CÁO CẦN SOẠN THẢO
================================================================================
Hãy viết báo cáo bằng Tiếng Việt học thuật, văn phong chuẩn mực khoa học, lập luận sắc bén và mạch lạc theo đúng 6 chương:
- BÌA BÁO CÁO: Trường ĐH Giao thông Vận tải TP.HCM (UTH), Viện CNTT, Bộ môn Trí tuệ Nhân tạo, Nhóm 7, Bảng phân công 5 thành viên.
- MỤC LỤC & DANH MỤC TỪ VIẾT TẮT, DANH MỤC HÌNH ẢNH, BẢNG BIỂU.
- CHƯƠNG 1: TỔNG QUAN ĐỀ TÀI, BỐI CẢNH & TÍNH CẤP THIẾT.
- CHƯƠNG 2: CƠ SỞ LÝ THUYẾT & MÔ HÌNH TOÁN HỌC (RFM, Tiền xử lý Z-Score/IQR, 4 thuật toán phân cụm, Gaussian Naïve Bayes).
- CHƯƠNG 3: HỆ THỐNG ĐO LƯỜNG & ĐỐI SÁNH KHOA HỌC (Elbow, Silhouette, Calinski-Harabasz, Davies-Bouldin).
- CHƯƠNG 4: THIẾT KẾ KIẾN TRÚC HỆ THỐNG & GIAO DIỆN APPLE GLASSMORPHISM (Phân tầng phần mềm, REST API, UI/UX).
- CHƯƠNG 5: THỰC NGHIỆM, ĐỐI SÁNH & ĐÁNH GIÁ KẾT QUẢ (Bảng số liệu thực nghiệm, phân tích ma trận 4 thuật toán, kết quả Bayes, phân tích chân dung khách hàng 360° & chiến lược tiếp thị).
- CHƯƠNG 6: KẾT LUẬN, ĐÁNH GIÁ & HƯỚNG PHÁT TRIỂN.
- TÀI LIỆU THAM KHẢO (Theo chuẩn IEEE/APA).

Hãy bắt đầu soạn thảo toàn bộ báo cáo từ đầu đến cuối một cách chi tiết, đầy đủ nhất!
