import numpy as np
from sklearn.cluster import KMeans

def hierarchical_kmeans_resampling(X, k_list, T=3, m=10, r_t=5, random_state=42):
    """
    Mục đích: Thực hiện Algorithm 1 (Hierarchical K-means with Resampling) để tìm các tâm cụm phân phối đều.

    Tham số:
    ----------
    X : numpy.ndarray - Dữ liệu đầu vào (n_samples, n_features).
    k_list : list[int] - Danh sách số cụm cho mỗi tầng (độ dài bằng T).
    T, m, r_t : int - Số tầng, số lần lặp resampling và số điểm lấy mẫu mỗi cụm.
    random_state : int - Hạt giống để cố định kết quả.

    Giá trị trả về:
    ----------
    dict - Chứa centroids/labels theo từng tầng và kết quả phân cụm cuối cùng trên X.
    """


    assert len(k_list) == T, "k_list phải có độ dài = T"

    # Lưu toàn bộ hierarchy
    all_centroids = []   # C_t
    all_labels = []      # L_t

    # Level 1: I = X
    current_input = X

    for t in range(T):
        print(f"--- Level {t+1}/{T} ---")
        k_t = k_list[t]

        # K-means trên I
        kmeans = KMeans(
            n_clusters=k_t,
            init='k-means++',
            n_init=1,
            random_state=random_state
        )
        kmeans.fit(current_input)
        centroids = kmeans.cluster_centers_

        # Assign cluster trên I
        labels = kmeans.labels_

        # Resampling loop
        for s in range(m):

            resampled_points = []

            for i in range(k_t):
                cluster_points = current_input[labels == i]

                if len(cluster_points) == 0:
                    continue

                # Tính khoảng cách tới centroid
                distances = np.linalg.norm(
                    cluster_points - centroids[i], axis=1
                )

                # Lấy r_t điểm gần nhất
                n_sample = min(r_t, len(cluster_points))
                idx = np.argsort(distances)[:n_sample]

                resampled_points.append(cluster_points[idx])

            # Tạo tập R
            R = np.vstack(resampled_points)

            # K-means trên R để update centroid
            kmeans = KMeans(
                n_clusters=k_t,
                init=centroids,  
                n_init=1,
                random_state=random_state
            )
            kmeans.fit(R)

            centroids = kmeans.cluster_centers_

            # Re-assign lại trên I 
            labels = kmeans.predict(current_input)

        # Lưu kết quả level t
        all_centroids.append(centroids)
        all_labels.append(labels)

        # chuẩn bị level tiếp theo
        current_input = centroids  

    # gán label cho toàn bộ X theo level cuối
    final_kmeans = KMeans(
        n_clusters=k_list[-1],
        init=all_centroids[-1],
        n_init=1
    )
    final_kmeans.fit(all_centroids[-1])

    final_labels = final_kmeans.predict(X)

    return {
        "centroids_per_level": all_centroids,
        "labels_per_level": all_labels,
        "final_centroids": all_centroids[-1],
        "final_labels": final_labels
    }