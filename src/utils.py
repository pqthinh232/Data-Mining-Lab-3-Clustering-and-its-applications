import numpy as np
from sklearn.datasets import make_blobs
import pickle
import os

import numpy as np

def generate_simulated_data(random_state=42):
    """
    Tái hiện chính xác bộ dữ liệu mô phỏng trong Mục 4.1 của bài báo.
    Tổng cộng: 9000 điểm.
    - Cụm 1: 7000 điểm (Mật độ cực cao - Head)
    - Cụm 2: 1000 điểm (Mật độ trung bình)
    - Cụm 3: 500 điểm (Mật độ thấp)
    - Uniform: 500 điểm (Dữ liệu nền - Tail/Noise)
    """
    # Sử dụng Generator với seed để đảm bảo tính tái hiện
    rng = np.random.default_rng(random_state)
    
    data = np.concatenate([
        # Cụm Gauss 1: Trung tâm [-1, -1], độ lệch chuẩn 0.5
        rng.standard_normal((7000, 2)) * 0.5 + [-1, -1],
        
        # Cụm Gauss 2: Trung tâm [1, -1], độ lệch chuẩn 0.5
        rng.standard_normal((1000, 2)) * 0.5 + [1, -1],
        
        # Cụm Gauss 3: Trung tâm [0, 1], độ lệch chuẩn 0.5
        rng.standard_normal((500, 2)) * 0.5 + [0, 1],
        
        # Dữ liệu phân phối đều trong khoảng [-3, 3]
        # (rng.random - 0.5) * 6 đưa dải [0, 1] về [-3, 3]
        (rng.random((500, 2)) - 0.5) * 6,
    ])
    
    return data


def load_cifar10_longtail(path, imbalance_ratio=0.01):
    """
    path: Đường dẫn đến folder 'cifar-10-batches-py'
    imbalance_ratio: Tỷ lệ giữa lớp ít nhất và lớp nhiều nhất (0.01 = 1%)
    """
    def unpickle(file):
        with open(file, 'rb') as fo:
            return pickle.load(fo, encoding='bytes')

    # Load toàn bộ 5 batch training
    x_all, y_all = [], []
    for i in range(1, 6):
        d = unpickle(os.path.join(path, f'data_batch_{i}'))
        x_all.append(d[b'data'])
        y_all.append(d[b'labels'])
    
    X = np.vstack(x_all).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    Y = np.concatenate(y_all)

    # Tạo phân phối Long-tail (Exponential decay)
    n_classes = 10
    img_max = 5000
    selected_indices = []
    
    for cls_idx in range(n_classes):
        # Tính số lượng ảnh giữ lại cho lớp này
        num_imgs = int(img_max * (imbalance_ratio ** (cls_idx / (n_classes - 1.0))))
        cls_indices = np.where(Y == cls_idx)[0]
        np.random.shuffle(cls_indices)
        selected_indices.append(cls_indices[:num_imgs])
        print(f"Lớp {cls_idx}: giữ lại {num_imgs} ảnh")

    sel_idx = np.concatenate(selected_indices)
    return X[sel_idx], Y[sel_idx]

def load_cifar10_test(path):
    """
    Mục đích: Load tập dữ liệu Test chuẩn của CIFAR-10 (10.000 ảnh cân bằng).
    Tham số: path - đường dẫn đến thư mục 'cifar-10-batches-py'
    """
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            return pickle.load(fo, encoding='bytes')

    # File test mặc định của CIFAR-10 tên là 'test_batch'
    test_data = unpickle(os.path.join(path, 'test_batch'))
    
    X_test = test_data[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    Y_test = np.array(test_data[b'labels'])
    
    return X_test, Y_test