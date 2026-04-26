import numpy as np
from sklearn.neighbors import KernelDensity

def calculate_kl_divergence(points, L=3, step=0.02, bandwidth=0.5):
    """
    Hàm tính KL chuẩn 100% theo logic của tác giả.
    Sử dụng tham số 'step' để khớp với lệnh gọi trong Notebook.
    """
    # 1. Tạo lưới tọa độ
    x_range = np.arange(-L, L, step)
    y_range = np.arange(-L, L, step)
    gx, gy = np.meshgrid(x_range, y_range)
    grid = np.vstack([gx.ravel(), gy.ravel()]).T

    # 2. Mật độ đều trên Omega
    d_u = 1 / (4 * L**2)
    
    # 3. Ước lượng KDE (Dùng sklearn đúng như tác giả)
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian').fit(points)
    
    # 4. Tính mật độ f và chuẩn hóa
    # kde.score_samples trả về log-density
    f = np.exp(kde.score_samples(grid))
    # Chuẩn hóa để tích phân trên lưới xấp xỉ bằng 1
    f_norm = f * 1 / (step**2 * f.sum())
    
    # 5. Công thức KL chuẩn của tác giả: -sum( P * log(U/P) * delta )
    # Tránh lỗi log(0) bằng cách cộng thêm epsilon 1e-12
    KL = -(f_norm * np.log(d_u / (f_norm + 1e-12))).sum() * (step**2)
    
    return KL

import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

def calculate_acc(y_true, y_pred):
    """
    Mục đích: Tính toán độ chính xác phân cụm (Clustering Accuracy) bằng cách sử dụng 
    thuật toán Hungarian (vòng lặp tối ưu) để khớp nhãn cụm với nhãn thực tế.

    Tham số:
    - y_true (array-like): Nhãn thực tế (Ground truth labels).
    - y_pred (array-like): Nhãn cụm dự đoán (Predicted cluster assignments).

    Giá trị trả về:
    - acc (float): Giá trị Accuracy trong khoảng [0, 1].
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    
    # 1. Xác định kích thước ma trận chi phí (Cost Matrix)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    
    # 2. Xây dựng ma trận hiệp biến (Contingency matrix)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
        
    # 3. Sử dụng thuật toán Hungarian (linear_sum_assignment) 
    # để tìm cách gán nhãn sao cho tổng số mẫu khớp là lớn nhất.
    # Lưu ý: linear_sum_assignment tìm chi phí nhỏ nhất, nên ta lấy Max - w.
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    
    # 4. Tính toán Accuracy
    acc = sum([w[i, j] for i, j in zip(row_ind, col_ind)]) / y_pred.size
    return acc

def calculate_nmi(y_true, y_pred):
    """
    Mục đích: Tính toán Normalized Mutual Information (NMI), đo lường sự tương quan 
    giữa nhãn thực tế và nhãn cụm. Độ đo này bất biến với hoán vị nhãn.

    Tham số:
    - y_true (array-like): Nhãn thực tế.
    - y_pred (array-like): Nhãn cụm dự đoán.

    Giá trị trả về:
    - nmi (float): Giá trị NMI trong khoảng [0, 1].
    """
    return normalized_mutual_info_score(y_true, y_pred)

# Bạn có thể thêm hàm tính Entropy vào đây để dùng chung trong Notebook
def calculate_entropy(y_labels):
    """
    Mục đích: Tính toán Shannon Entropy của phân phối các lớp để đo độ cân bằng.
    Entropy càng cao chứng tỏ dữ liệu càng cân bằng.
    """
    _, counts = np.unique(y_labels, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log(probs + 1e-12))