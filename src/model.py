import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


def hierarchical_kmeans_resampling(X, k_list, T, m, r_t_list, num_init=10, random_state=42):
    """
    Bản cập nhật KHỚP 100% LOGIC với code gốc của Meta FAIR.
    """
    # 1. Xử lý r_t_list linh hoạt
    if isinstance(r_t_list, int):
        r_t_list = [r_t_list] * T  
        
    current_input = X.astype(np.float64) 
    centroids = None
    
    for t in range(T):
        k_t = k_list[t]
        r_t = r_t_list[t]
        print(f"--- Level {t+1}/{T} (k={k_t}, r_t={r_t}) ---")
        
        # --- BƯỚC 1: Initial K-means (Tác giả chạy 1 lần khởi tạo lớn) ---
        km = KMeans(n_clusters=k_t, init='k-means++', n_init=num_init, max_iter=50, tol=0, random_state=random_state)
        km.fit(current_input)
        centroids = km.cluster_centers_
        labels = km.labels_
        
        # --- BƯỚC 2: Resampling-kmeans (Vòng lặp làm phẳng) ---
        if m > 0 and r_t > 0:
            for s_step in range(m):
                # 2.1 Lấy mẫu 'closest' theo logic tác giả
                sampled_points = []
                for i in range(k_t):
                    cluster_pts = current_input[labels == i]
                    if len(cluster_pts) == 0: continue
                    
                    # Tính khoảng cách tới centroid hiện tại
                    dists = np.linalg.norm(cluster_pts - centroids[i], axis=1)
                    # Lấy r_t điểm gần nhất
                    idx = np.argsort(dists)[:min(len(cluster_pts), r_t)]
                    sampled_points.append(cluster_pts[idx])
                
                R = np.vstack(sampled_points)
                
                # 2.2 CHỖ SỬA QUAN TRỌNG: Chạy K-means trên R nhưng KHÔNG dùng warm start
                # Tác giả khởi tạo lại (k-means++) để phá vỡ cấu trúc cũ
                km_R = KMeans(n_clusters=k_t, init='k-means++', n_init=num_init, max_iter=50, tol=0, random_state=random_state)
                km_R.fit(R)
                centroids = km_R.cluster_centers_
                
                # 2.3 Gán nhãn lại cho toàn bộ tập input để vòng sau lấy mẫu chuẩn hơn
                labels = km_R.predict(current_input)
        
        # Centroids của tầng này là input cho tầng sau
        current_input = centroids
        
    return centroids



def kmeans_with_power_s_stable(X, k, s_target, n_iters=20, random_state=42):
    """
    Sử dụng kỹ thuật Annealing: Tăng dần s để Centroids di chuyển ổn định.
    Giúp thanh màu đỏ thấp dần đúng theo bài báo.
    """
    # 1. Bắt đầu từ K-means tiêu chuẩn (s=2)
    km = KMeans(n_clusters=k, init='k-means++', n_init=1, random_state=random_state).fit(X)
    centroids = km.cluster_centers_

    # Danh sách các bước tăng s (ví dụ target 256 thì đi qua 4, 16, 64)
    s_steps = [4, 16, 64, 256]
    actual_steps = [s for s in s_steps if s <= s_target]

    for s in actual_steps:
        for _ in range(10): # Chạy 20 vòng lặp cho mỗi mức s
            dists = cdist(X, centroids, metric='euclidean')
            labels = np.argmin(dists, axis=1)
            
            new_centroids = []
            for i in range(k):
                cp = X[labels == i]
                if len(cp) > 0:
                    # IRLS Update với Log-sum-exp để chống lộn xộn
                    c = cp.mean(axis=0)
                    for _ in range(3):
                        d = np.linalg.norm(cp - c, axis=1) + 1e-6
                        log_w = (s - 2) * np.log(d)
                        w = np.exp(log_w - np.max(log_w)) # Ổn định hóa
                        w /= (w.sum() + 1e-12)
                        c = np.average(cp, axis=0, weights=w)
                    new_centroids.append(c)
                else:
                    new_centroids.append(centroids[i])
            centroids = np.array(new_centroids)
            
    return centroids
