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

def load_cifar10_kaggle(path):
    """Đọc dữ liệu CIFAR-10 từ folder bạn đã tải từ Kaggle (dạng pickle)"""
    # CIFAR-10 Kaggle thường có các file data_batch_1, ..., test_batch
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    # Ví dụ load test_batch
    data_file = os.path.join(path, 'test_batch')
    d = unpickle(data_file)
    x = d[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.array(d[b'labels'])
    return x, y