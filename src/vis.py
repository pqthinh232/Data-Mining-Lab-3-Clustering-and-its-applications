import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

def plot_voronoi_and_kde(X, centroids, title, ax_voronoi, ax_kde):
    # 1. Voronoi Diagram
    ax_voronoi.scatter(X[:,0], X[:,1], s=1, color='lightblue', alpha=0.2)
    vor = Voronoi(centroids)
    voronoi_plot_2d(vor, ax=ax_voronoi, show_vertices=False, line_colors='black', line_width=0.5, point_size=2)
    ax_voronoi.set_title(title)
    ax_voronoi.set_xlim(-3, 3); ax_voronoi.set_ylim(-3, 3)

    # 2. KDE 3D Surface
    grid_size = 50
    x, y = np.mgrid[-3:3:complex(grid_size), -3:3:complex(grid_size)]
    pos = np.vstack([x.ravel(), y.ravel()])
    kernel = gaussian_kde(centroids.T)
    z = np.reshape(kernel(pos), x.shape)
    
    ax_kde.plot_surface(x, y, z, cmap='coolwarm', edgecolor='none')
    ax_kde.set_xticks([]); ax_kde.set_yticks([]); ax_kde.set_zticks([])