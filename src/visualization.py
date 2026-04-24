import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.stats import gaussian_kde
import numpy as np

def plot_paper_style_fig3a(X, all_centroids):
    group1 = ['data', 'k-means', 'k-means, s=4', 'k-means, s=64', 'k-means, s=256']
    group2 = ['ours, 2-level', 'ours, 3-level', 'ours, 3-level w/ resamp.', 'dbscan', 'agglomerative']
    
    # Thiết lập lưới tọa độ cho KDE
    grid_size = 50
    x_g, y_g = np.mgrid[-3.5:3.5:complex(grid_size), -3.5:3.5:complex(grid_size)]
    pos = np.vstack([x_g.ravel(), y_g.ravel()])
    
    z_max = gaussian_kde(X.T)(pos).max()

    fig = plt.figure(figsize=(20, 16))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    def draw_group(methods, row_start_idx):
        for i, name in enumerate(methods):
            # --- VẼ VORONOI (Hàng 1 và 3) ---
            ax_vor = fig.add_subplot(4, 5, row_start_idx + i + 1)
            if name == 'data':
                ax_vor.scatter(X[:, 0], X[:, 1], s=0.8, alpha=0.3, color='tab:blue')
            else:
                centroids = all_centroids.get(name)
                if centroids is not None:
                    vor = Voronoi(centroids)
                    voronoi_plot_2d(vor, ax=ax_vor, show_vertices=False, 
                                    line_colors='black', line_width=0.4, 
                                    point_size=1, point_color='blue')
            
            ax_vor.set_title(name, fontsize=12)
            ax_vor.set_xlim(-3.2, 3.2); ax_vor.set_ylim(-3.2, 3.2)
            ax_vor.set_xticks([]); ax_vor.set_yticks([])
            ax_vor.set_aspect('equal')

            # --- VẼ KDE 3D (Hàng 2 và 4) ---
            ax_kde = fig.add_subplot(4, 5, row_start_idx + i + 6, projection='3d')
            source = X if name == 'data' else all_centroids.get(name)
            
            if source is not None:
                kernel = gaussian_kde(source.T)
                z = np.reshape(kernel(pos), x_g.shape)
                # Vẽ surface
                surf = ax_kde.plot_surface(x_g, y_g, z, cmap='coolwarm', 
                                           edgecolor='none', alpha=0.8, antialiased=True)
                
                ax_kde.set_zlim(0, z_max)
                ax_kde.grid(True, linestyle='--', alpha=0.5)
                # Ẩn các con số trên trục để sạch sẽ nhưng giữ lưới
                ax_kde.set_xticklabels([]); ax_kde.set_yticklabels([]); ax_kde.set_zticklabels([])
                # Góc nhìn chuẩn
                ax_kde.view_init(elev=30, azim=-60)
                
                # Làm cho các mặt phẳng (panes) trong suốt hoặc màu xám nhẹ giống báo
                ax_kde.xaxis.pane.set_edgecolor('gainsboro')
                ax_kde.yaxis.pane.set_edgecolor('gainsboro')
                ax_kde.zaxis.pane.set_edgecolor('gainsboro')
                ax_kde.xaxis.pane.fill = False
                ax_kde.yaxis.pane.fill = False
                ax_kde.zaxis.pane.fill = False

    # Vẽ cụm 1 (5 cột đầu) vào 2 hàng đầu
    draw_group(group1, 0)
    # Vẽ cụm 2 (5 cột sau) vào 2 hàng cuối
    draw_group(group2, 10)

    plt.tight_layout()
    plt.show()