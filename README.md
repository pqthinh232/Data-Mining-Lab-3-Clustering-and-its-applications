
# LAB 3: Khai thác dữ liệu và ứng dụng (Data Mining)

## Đề tài: Automatic Data Curation for Self-Supervised Learning: A Clustering-Based Approach

Mục tiêu của dự án này là tái hiện và đánh giá thuật toán **Hierarchical K-means (HK-means)** kết hợp với **Resampling** để tự động cân bằng dữ liệu (Data Curation) từ các nguồn dữ liệu thô (long-tail), hỗ trợ cho các mô hình học tự giám sát (Self-supervised Learning).

---

## 👥 Thành viên thực hiện
* **Lê Quốc Thiện** - MSSV: `23127481`
* **Phạm Quang Thịnh** - MSSV: `23127485`

---

## 📂 Cấu trúc thư mục dự án

```text
root/
├── data/               # Thư mục chứa dữ liệu thực nghiệm (Cần tải về máy)
├── docs/               # Tài liệu báo cáo chi tiết
│   └── Report.pdf      # File báo cáo đồ án (PDF)
├── notebooks/          # Các file Jupyter Notebook thực nghiệm chính
│   ├── 01_main_experiments.ipynb  # Thực nghiệm trên dữ liệu mô phỏng 2D
│   ├── 02_ablation_study.ipynb   # Các nghiên cứu thăm dò (Ablation Study)
│   ├── 03_new_dataset.ipynb      # Thực nghiệm trên bộ dữ liệu CIFAR-10 Long-tail
│   └── grid_test_results_cifar10.csv # Kết quả Grid Search lưu trữ
├── paper/              # Bài báo khoa học gốc
│   └── 2717_Automatic_Data_Curation_f.pdf
├── src/                # Mã nguồn tự triển khai và tối ưu bởi nhóm
│   ├── metrics.py      # Các độ đo: KL Divergence, NMI, ACC, ARI...
│   ├── model.py        # Định nghĩa các lớp mô hình và Linear Probing
│   ├── utils.py        # Các hàm bổ trợ xử lý dữ liệu
│   └── visualization.py # Công cụ trực quan hóa (Voronoi, KDE 3D...)
├── src_author/         # Mã nguồn gốc của tác giả (Dùng để đối chứng)
│   ├── clusters.py
│   ├── distributed_kmeans_gpu.py # K-means phân tán trên GPU
│   ├── hierarchical_kmeans_gpu.py
│   └── hierarchical_sampling.py
└── requirements.txt    # Danh sách các thư viện cần thiết

```

---

## 🛠 Hướng dẫn cài đặt

Để đảm bảo môi trường thực thi ổn định và tương thích, vui lòng cài đặt các thư viện cần thiết thông qua tệp `requirements.txt`.

1. **Tạo môi trường ảo (Khuyến khích):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Trên Linux/Mac
   # hoặc
   .\venv\Scripts\activate   # Trên Windows
   ```

2. **Cài đặt thư viện:**
   ```bash
   pip install -r requirements.txt
   ```

*Danh sách thư viện chính bao gồm: `torch` (>=1.12.0), `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`,...*

---

## 📥 Hướng dẫn thiết lập dữ liệu

Do kích thước dữ liệu lớn (bao gồm các file `.npy` chứa embedding đã trích xuất bằng DINOv2), nhóm không upload trực tiếp lên repository. Vui lòng làm theo các bước sau:

1. **Truy cập liên kết Drive:** [Tải folder Data tại đây](https://drive.google.com/drive/folders/1eIYaKHB-kYmHZWz6YOoNU4w8RfOUOhtR?usp=sharing)
2. **Tải xuống thư mục `data`**.
3. **Đặt thư mục đã tải vào thư mục gốc** của dự án sao cho cấu trúc là `root/data/`.

---

## 🚀 Giới thiệu các thành phần thực nghiệm

### 1. Các Notebook chính (`notebooks/`)
*   **`01_main_experiments.ipynb`**: Tái hiện thí nghiệm trên không gian 2D. Chứng minh khả năng "làm phẳng" (flattening) phân phối dữ liệu của HK-means để đạt tới phân phối đều (uniform distribution).
*   **`02_ablation_study.ipynb`**: Nghiên cứu sự ảnh hưởng của các siêu tham số như chiến lược chọn mẫu từ tâm cụm (Closest, Furthest, Median) và ngân sách lấy mẫu lại ($r_t$).
*   **`03_new_dataset.ipynb`**: Áp dụng quy trình Curation lên bộ dữ liệu **CIFAR-10 Long-tail**. Đây là phần thực nghiệm thay thế cho bộ dữ liệu 743M ảnh của tác giả, nhằm đánh giá hiệu quả thuật toán trong điều kiện tài nguyên hạn chế.

### 2. File thuật toán quan trọng
*   **`src/metrics.py`**: Chứa hàm tính toán **KL Divergence** giữa phân phối tâm cụm và phân phối đều, đây là chỉ số cốt lõi để đánh giá độ thành công của việc curation.
*   **`src/visualization.py`**: Chứa các hàm vẽ bản đồ Voronoi và mật độ KDE 3D, giúp quan sát trực quan sự thay đổi của các centroid qua từng cấp độ phân cấp.
*   **`src_author/`**: Chứa cài đặt gốc của tác giả sử dụng **PyTorch GPU**. Nhóm sử dụng phần này để chạy đối chứng (Cross-validation) nhằm đảm bảo tính tái hiện (reproducibility) của thuật toán.

---
*Dự án được thực hiện cho môn học Khai thác dữ liệu và ứng dụng - Trường ĐH Khoa học Tự nhiên, ĐHQG-HCM.*