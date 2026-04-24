import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def hierarchical_kmeans_resampling(X, k_list, T, m, r_t_list, num_init=10, random_state=42):
    """
    Bản cập nhật khớp logic 100% với hierarchical_kmeans_with_resampling của Meta.
    r_t_list: Danh sách r_t cho từng tầng, ví dụ [10, 5, 1]
    """
    if isinstance(r_t_list, int):
        r_t_list = [r_t_list] * T  
    current_input = X
    centroids = None
    
    for t in range(T):
        k_t = k_list[t]
        r_t = r_t_list[t] # Lấy r_t riêng cho từng tầng
        
        # --- BƯỚC 1: Initial K-means (Tác giả gọi là 'Initial kmeans') ---
        # Chạy với n_init cao để ổn định
        km = KMeans(n_clusters=k_t, init='k-means++', n_init=num_init, random_state=random_state)
        km.fit(current_input)
        centroids = km.cluster_centers_
        labels = km.labels_
        
        # --- BƯỚC 2: Resampling loop (Tác giả gọi là 'Resampling-kmeans') ---
        if m > 0 and r_t > 0:
            for _ in range(m):
                sampled_points = []
                for i in range(k_t):
                    cluster_pts = current_input[labels == i]
                    if len(cluster_pts) == 0: continue
                    
                    # Logic 'closest' strategy của tác giả:
                    dists = np.linalg.norm(cluster_pts - centroids[i], axis=1)
                    # Sắp xếp và lấy r_t điểm gần nhất
                    idx = np.argsort(dists)[:min(len(cluster_pts), r_t)]
                    sampled_points.append(cluster_pts[idx])
                
                R = np.vstack(sampled_points)
                
                # Chạy K-means trên tập R (Tác giả dùng n_init=num_init ở đây luôn)
                km_R = KMeans(n_clusters=k_t, init=centroids, n_init=1, random_state=random_state)
                km_R.fit(R)
                centroids = km_R.cluster_centers_
                
                # Gán nhãn lại cho toàn bộ input của level này (Rất quan trọng)
                labels = km_R.predict(current_input)
        
        # Chuẩn bị cho level tiếp theo: X_next = Centroids_prev
        current_input = centroids
        
    return centroids


import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

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
        for _ in range(10): # Chạy 10 vòng lặp cho mỗi mức s
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
