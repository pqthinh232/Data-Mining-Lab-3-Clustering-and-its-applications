from scipy.stats import gaussian_kde
from sklearn.metrics import normalized_mutual_info_score as nmi
from scipy.optimize import linear_sum_assignment
import numpy as np

def calculate_kl_divergence(centroids, low=-3, high=3, grid_size=100):
    """
    Mục đích: Đo độ đồng đều của các tâm cụm so với phân phối đều (Uniform) bằng phương pháp KDE.

    Tham số:
    ----------
    centroids : numpy.ndarray (n_clusters, 2) - Tọa độ các tâm cụm.
    low, high : float - Giới hạn phạm vi của lưới tọa độ.
    grid_size : int - Độ phân giải của lưới tính mật độ.

    Giá trị trả về:
    ----------
    kl_div : float - Giá trị KL-Divergence (số càng nhỏ thì tâm cụm càng phân bố đều).
    """


    # 1. Tạo lưới grid 2D
    x_grid, y_grid = np.mgrid[low:high:complex(grid_size), low:high:complex(grid_size)]
    positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
    
    # 2. Ước lượng mật độ hạt nhân (KDE) của centroids
    kernel = gaussian_kde(centroids.T)
    # Tính mật độ tại các điểm trên lưới
    f = np.reshape(kernel(positions), x_grid.shape)
    
    # Tránh giá trị 0/âm để tính log, sau đó chuẩn hóa thành phân phối xác suất
    f = np.clip(f, 1e-10, None)
    f /= np.sum(f) 
    
    # 3. Tạo phân phối đều (Uniform) trên lưới grid
    u = np.ones_like(f) / (grid_size * grid_size)
    
    # 4. Tính KL Divergence: D_KL(U || F)
    # đo độ lệch của F so với U
    kl_div = np.sum(u * np.log(u / f))
    return kl_div

def calculate_acc(y_true, y_pred):
    """
    Mục đích: Tính độ chính xác phân cụm bằng thuật toán Hungarian để khớp nối nhãn tối ưu.

    Tham số:
    ----------
    y_true : numpy.ndarray - Nhãn thực tế của dữ liệu.
    y_pred : numpy.ndarray - Nhãn dự đoán từ thuật toán phân cụm.

    Giá trị trả về:
    ----------
    acc : float - Chỉ số Accuracy sau khi khớp nhãn (từ 0 đến 1).
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size

def calculate_nmi(y_true, y_pred):
    """
    Mục đích: Tính Normalized Mutual Information.
    """
    return nmi(y_true, y_pred)
