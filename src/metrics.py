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